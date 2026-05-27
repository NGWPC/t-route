#!/usr/bin/env python3
"""Tier C -- parallel CONUS-scale benchmark (single-shot diagnostic).

Runs the full ~1.1 M-flowpath NHF case once, in parallel (cpu_pool=8), to
expose the regime the kernel-dominated serial Tier A run cannot reach:
parallel scaling, real memory footprint, and load-imbalance hotpaths.

Single run by design -- CONUS is expensive. Always captures wall / CPU /
peak RSS and t-route's own phase breakdown. The `--profile` mode adds a
hotpath profile:

  --profile cprofile  (default)  cProfile of the main process -> .pstats +
                                 top-cumulative hotpaths printed inline.
                                 (Misses joblib worker processes.)
  --profile pyspy                py-spy sampling profiler attached to the
                                 run incl. subprocesses + native frames ->
                                 flamegraph SVG. Needs sudo on macOS.
  --profile none                 clean timing only.

Usage:
    python benchmark/bench_conus.py
    python benchmark/bench_conus.py --profile pyspy
    python benchmark/bench_conus.py --profile none
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pstats
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil
import yaml

BENCH_DIR = Path(__file__).resolve().parent
CONFIG = BENCH_DIR / "conus.yaml"
DATA_DIR = BENCH_DIR / "data" / "conus"
RESULTS_DIR = BENCH_DIR / "results"

TIMING_KEYS = ("Network graph construction", "Forcing array construction",
               "Routing computations", "Output writing",
               "Total execution time")


def resolved_config(output_dir: Path) -> Path:
    """Render conus.yaml with absolute data paths + an isolated output dir."""
    if not (DATA_DIR / "MANIFEST.json").exists():
        sys.exit("ERROR: CONUS dataset missing. Run: "
                 "python benchmark/prep_conus.py")
    cfg = yaml.safe_load(CONFIG.read_text())
    sp = cfg["network_topology_parameters"]["supernetwork_parameters"]
    sp["geo_file_path"] = str((BENCH_DIR / sp["geo_file_path"]).resolve())
    fp = cfg["compute_parameters"]["forcing_parameters"]
    fp["qlat_input_folder"] = str((BENCH_DIR / fp["qlat_input_folder"]).resolve())
    cfg["output_parameters"]["stream_output"]["stream_output_directory"] = str(output_dir)
    path = output_dir / "_resolved.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _rusage_metrics(ru, wall: float) -> dict:
    rss = ru.ru_maxrss if sys.platform == "darwin" else ru.ru_maxrss * 1024
    return {"wall_s": wall, "cpu_s": ru.ru_utime + ru.ru_stime,
            "rss_mb": rss / (1024 * 1024)}


class _MemSampler:
    """Background thread tracking the peak total RSS of a process tree.

    os.wait4 only reports the *main* process's peak RSS; a parallel run's
    real footprint is main + all joblib worker processes. This polls the
    whole tree (psutil) and keeps the peak of the summed RSS.
    """

    def __init__(self, root_pid: int, interval: float = 0.5):
        self.root_pid = root_pid
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        try:
            root = psutil.Process(self.root_pid)
        except psutil.Error:
            return
        while not self._stop.is_set():
            total = 0
            try:
                for p in [root, *root.children(recursive=True)]:
                    try:
                        total += p.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            self.peak_bytes = max(self.peak_bytes, total)
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2)
        return self.peak_bytes / (1024 * 1024)


def run(config: Path, log: Path, mode: str, pstats_path: Path,
        flame_path: Path) -> dict:
    """Run the CONUS case once under the chosen profiling mode."""
    routing = [sys.executable, "-m", "nwm_routing", "-V5", "-f", str(config)]
    if mode == "cprofile":
        cmd = [sys.executable, "-m", "cProfile", "-o", str(pstats_path),
               "-m", "nwm_routing", "-V5", "-f", str(config)]
    else:
        cmd = routing

    with open(log, "wb") as fh:
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        sampler = _MemSampler(proc.pid)
        sampler.start()
        spy = None
        if mode == "pyspy":
            spy = subprocess.Popen(
                ["sudo", "py-spy", "record", "--pid", str(proc.pid),
                 "--subprocesses", "--native", "--rate", "100",
                 "-o", str(flame_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _, status, ru = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - t0
        peak_tree_mb = sampler.stop()
        if spy is not None:
            spy.wait()
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError(f"nwm_routing failed (exit {proc.returncode}):\n{tail}")
    m = _rusage_metrics(ru, wall)
    m["peak_tree_rss_mb"] = peak_tree_mb
    return m


def parse_timing(log: Path) -> list[str]:
    """Extract t-route's own phase breakdown from the run log."""
    out = []
    for line in log.read_text(errors="replace").splitlines():
        for key in TIMING_KEYS:
            if key in line:
                out.append(key + ": " + line.split(key, 1)[1].lstrip(": ").strip())
    return out


def top_hotpaths(pstats_path: Path, n: int = 20) -> None:
    """Print the top-N cumulative-time functions from a cProfile run."""
    st = pstats.Stats(str(pstats_path))
    st.sort_stats("cumulative")
    print(f"\ntop {n} hotpaths (cumulative time, main process):")
    st.print_stats(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=("cprofile", "pyspy", "none"),
                    default="cprofile", help="hotpath profiling mode")
    ap.add_argument("--label", default=None, help="label for saved artifacts")
    ap.add_argument("--json", action="store_true", help="write results/<label>.conus.json")
    args = ap.parse_args()

    label = args.label or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    pstats_path = RESULTS_DIR / f"{label}.conus.pstats"
    flame_path = RESULTS_DIR / f"{label}.conus.svg"

    work = Path(tempfile.mkdtemp(prefix="bench_conus_"))
    try:
        out_dir = work / "out"
        out_dir.mkdir()
        config = resolved_config(out_dir)
        log = out_dir / "run.log"

        print(f"CONUS-scale run (cpu_pool=8, profile={args.profile}) -- "
              f"this is a single expensive run ...")
        m = run(config, log, args.profile, pstats_path, flame_path)

        tree_gb = m["peak_tree_rss_mb"] / 1024
        print(f"\nCONUS end-to-end (single run)")
        print(f"  wall          {m['wall_s']:9.1f} s")
        print(f"  cpu           {m['cpu_s']:9.1f} s   "
              f"({m['cpu_s'] / m['wall_s']:.2f}x wall -- parallel utilization)")
        print(f"  peak RSS      {tree_gb:9.2f} GB  (whole process tree)")
        print(f"  peak RSS      {m['rss_mb'] / 1024:9.2f} GB  (main process only)")

        timing = parse_timing(log)
        if timing:
            print("\nt-route phase breakdown:")
            for line in timing:
                print(f"  {line}")

        if args.profile == "cprofile" and pstats_path.exists():
            top_hotpaths(pstats_path)
            print(f"\npstats -> {pstats_path}  (open with: snakeviz {pstats_path})")
        elif args.profile == "pyspy":
            if flame_path.exists():
                print(f"\nflamegraph -> {flame_path}")
            else:
                print("\nNOTE: py-spy produced no flamegraph -- on macOS it "
                      "needs sudo. Re-run `--profile pyspy` with sudo available.")

        if m["wall_s"] > 0:
            print("\nRESULTS.md CONUS row:")
            note = {"cprofile": "cProfile", "pyspy": "py-spy",
                    "none": "clean"}[args.profile]
            print(f"| `{_git_sha()}` | {m['wall_s']:.1f} | {m['cpu_s']:.1f} | "
                  f"{m['peak_tree_rss_mb'] / 1024:.2f} | {note} |")

        if args.json:
            rec = {"label": label, "git_sha": _git_sha(),
                   "utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                   "profile": args.profile, "metrics": m, "timing": timing}
            (RESULTS_DIR / f"{label}.conus.json").write_text(json.dumps(rec, indent=2))
            print(f"json -> {RESULTS_DIR / (label + '.conus.json')}")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BENCH_DIR,
            text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
