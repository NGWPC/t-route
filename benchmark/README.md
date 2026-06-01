# `benchmark/` t-route performance benchmarks

This directory holds the performance harness for the routing pipeline:
configs, bench drivers, golden output for correctness gates, and the
PNGs that ship with `RESULTS.md`. All measurements are produced inside
the project's devcontainer (`docker/Dockerfile.dev`, Rocky Linux 9)
so they are reproducible on any host that runs Docker.

## What this folder contains

| File | Purpose |
|---|---|
| `RESULTS.md` | Executive summary plus per-change technical writeup with bar charts. **Start here.** |
| `nhf_subset_ohio.yaml` | Tier A config: 1-day, ~11 k flowpaths, single worker. Used for correctness gates and kernel-dominated wall measurement. |
| `conus.yaml` | Tier C config: full CONUS NHF (1.1 M flowpaths), 8 workers, 24 timesteps. Used for production-scale wall measurement. |
| `bench_e2e.py` | Tier A driver. Runs the full `nwm_routing` CLI, measures wall/CPU/RSS, compares output to `golden/` (PASS / FAIL gate). |
| `bench_conus.py` | Tier C driver. Single CONUS run with `--profile {cprofile,pyspy,none}` for hot-path analysis. |
| `bench_kernel.py` | Tier B microbenchmark. Replays harvested `compute_network_structured()` calls so the MC kernel can be timed in isolation, without the Python pipeline around it. |
| `harvest_kernel_inputs.py` | Records the kernel inputs from a real Tier A run into `data/kernel_calls.pkl`, so the kernel bench can replay them deterministically. |
| `prep_ohio_data.py`, `prep_conus.py` | Build the Tier A and Tier C input data from the NHF v1.1.4 CONUS geopackage. |
| `sweep_max_loop_size.py` | Runs Tier A across a sweep of `max_loop_size` values, captures wall/CPU/RSS per point, writes `results/max_loop_size_sweep.json`. Backs the operational deployment recommendation on chunk sizing. |
| `plot_max_loop_size.py` | Renders the sweep JSON to `figures/max_loop_size_sweep.png`. |
| `data/`, `golden/` | Input geopackages and reference output netCDFs (gitignored; build locally). |
| `results/` | Per-run JSON metric files (`{label}.json`, `{label}.kernel.json`, `{label}.conus.json`, `max_loop_size_sweep.json`). |
| `figures/` | PNG bar charts embedded in `RESULTS.md`. Regenerate via `python benchmark/generate_figures.py`. |
| `generate_figures.py` | Builds the bar charts in `figures/` from the measured numbers (constants at the top). |

## High-level summary

The optimizations drop **CONUS wall time from 297.2 s to 131.7 s
(2.26x speedup)** while reducing CPU time by 1.68x, pushing worker
utilization from 1.40x to 1.88x of 8 cores, and cutting peak
tree-RSS (sum across the main process plus 8 workers) by 3.50x
(100.7 GB to 28.7 GB). Tier A wall improves 1.22x and the isolated
MC-kernel replay (Tier B) improves 1.31x. Output is bit-identical
to a golden saved with the optimized build on the correctness gate. All
numbers are devcontainer measurements with `MALLOC_ARENA_MAX=2`.
See `RESULTS.md` for the per-change breakdown.

The work is grouped into three tracks:

1. **Kernel-level** (`src/kernel/muskingum/`):
   `-O3 -funroll-loops` build (with optional `TROUTE_NATIVE=1`
   for host-specific `-mcpu=native`/`-march=native` tuning); hoisted
   loop-invariant transcendentals; strength-reduced powers; common
   subexpression elimination (CSE) on the upstream-weighted sum.
2. **Routing-side** (`src/troute-routing/troute/routing/compute.py`):
   eliminated per-cluster deepcopy; consolidated 6+ per-cluster
   `.reindex` calls into one extended-index `pd.api.extensions.take`;
   per-cluster fast-path guards; `.to_numpy(copy=False)` migration.
3. **Graph construction** (`src/troute-network/troute/`): vectorized
   `_discretize_links`, `extract_connections`, and the two
   `groupby.apply(list).to_dict()` calls in
   `crosswalk_nex_flowpath_poi`.

## Reproducing the results

Everything runs inside the devcontainer. The compiled core (Fortran
plus Cython extensions) is built by `compiler.sh` during the Docker
image build.

### Build the devcontainer image

```bash
docker build --target dev -f docker/Dockerfile.dev \
  --build-arg TROUTE_NATIVE=1 \
  -t troute-dev:bench .
```

`TROUTE_NATIVE=1` enables host-specific `-mcpu=native` /
`-march=native` arch tuning on the MC Fortran kernel. The
benchmark numbers in `RESULTS.md` were taken with this flag.
Omit it (the project default) for a portable build safe to run
on a different CPU than the build host -- the right choice for
shipping container images or conda packages across heterogeneous
clusters, at a small wall-time cost on the kernel.

The build produces an image with the t-route source compiled in
place under `/t-route` and the Python venv at `/opt/venv` already
on `PATH`. The bench commands below assume you launch a container
from that image with the bind mounts shown.

The dev image builds on Python 3.11 and is about 1.39 GB. It needs no
system GDAL or C++ toolchain: geopandas reads geopackages through
`pyogrio`, whose manylinux aarch64 wheels bundle GDAL. The headline
numbers in this folder were originally measured on Rocky 9's default
Python 3.9 and reproduce the Tier A output bit-for-bit on 3.11.

### Memory requirements

CONUS (Tier C) peaks at **~19 GB resident** in the main process
and **~29 GB resident across the whole process tree** (main + 8
joblib workers, measured with `MALLOC_ARENA_MAX=2`). **Configure
your container runtime with at least ~32 GB of RAM available to
the VM.** Without this the output-write step will be OOM-killed
(exit code -9) and you'll see the routing log stop mid-way at
"Handling output ..."

How you bump the VM's RAM ceiling depends on your runtime:

- **Docker Desktop:** Settings -> Resources -> Memory slider.
- **Podman:** `podman machine set --memory 32768` then
  `podman machine restart`.
- **OrbStack:** `orb config set memory_mib 32768 && orb stop && orb start`.
- **colima:** `colima start --memory 32`.

Tiers A and B run comfortably in the default 8-16 GB; only Tier C
needs the bump.

### Memory measurement (why `MALLOC_ARENA_MAX=2`)

Without an arena cap, glibc's ptmalloc2 allocator reserves several
MB of virtual memory per thread arena (capped at `8 * ncores`),
which inflates baseline RSS by tens of GB before any application
allocation happens. That overhead is identical in baseline and
after, so it doesn't affect *speedup ratios*, but it swamps the
absolute RSS numbers enough to hide the optimization-driven
changes.

Setting `MALLOC_ARENA_MAX=2` caps the arena count at 2 per
process, trading negligible allocator contention (the main
process is effectively single-threaded; joblib workers are
separate processes) for honest peak RSS measurements. This is a
common production setting in Python data services (Airflow, Dask,
etc.). If you omit the env var the wall numbers shift by a few
percent (baseline runs ~10% faster, narrowing the CONUS ratio
from 2.26x to 2.30x) but the memory wins documented in
`RESULTS.md` will not be visible.

### Source data

Both tiers derive from the **NextGen Hydrofabric v1.1.4 CONUS
geopackage** (`nhf_1.1.4.gpkg`, ~6 GB). The numbers in `RESULTS.md`
were produced against this exact dataset; reviewers with access to
it can reproduce them directly. Remember the local path; the
remaining steps both consume that single file.

### Set up the input data (one-time)

Run the prep scripts inside a container. The host path holding
`nhf_1.1.4.gpkg` is bind-mounted read-only at `/hydrofabric`, and
the benchmark directory is bind-mounted so the synthesized inputs
land in your working tree (the image itself does not bake
`benchmark/` in):

```bash
docker run --rm \
  -e MALLOC_ARENA_MAX=2 \
  -v /path/to/hydrofabric:/hydrofabric:ro \
  -v "$(pwd)/benchmark:/t-route/benchmark" \
  troute-dev:bench \
  bash -c "cd /t-route && \
    python benchmark/prep_ohio_data.py  --src /hydrofabric/nhf_1.1.4.gpkg && \
    python benchmark/prep_conus.py --src /hydrofabric/nhf_1.1.4.gpkg && \
    python benchmark/bench_e2e.py --save-golden && \
    python benchmark/harvest_kernel_inputs.py"
```

What each step does:

- `prep_ohio_data.py` carves the upstream subgraph of fp_id 1725641 (Ohio
  River basin, 11,327 flowpaths) and synthesizes 144 hourly forcing
  CSVs. Output: `benchmark/data/domain/nhf_subset_ohio.gpkg`.
- `prep_conus.py` processes the full CONUS hydrofabric and
  synthesizes 2 hourly forcing CSVs. Output:
  `benchmark/data/conus/nhf_conus.gpkg`.
- `bench_e2e.py --save-golden` runs one Tier A invocation and saves
  its output netCDFs under `benchmark/golden/` as the correctness
  reference. Every later run compares against this.
- `harvest_kernel_inputs.py` intercepts the
  `compute_network_structured` calls from a Tier A run and pickles
  their arguments into `benchmark/data/kernel_calls.pkl`, so Tier B
  can replay the MC kernel without the Python pipeline.

Both prep scripts **synthesize forcing** (constant per-segment
`q_lateral`), so no external CHRTOUT or forcing dataset is needed.
The MC kernel is forcing-shape-sensitive, not forcing-value-
sensitive, for performance purposes: identical shapes produce
representative timing. The saved golden becomes your local
correctness reference; subsequent runs PASS if they reproduce that
output bit-for-bit, FAIL on new NaN/Inf.

### Run the benchmarks

Once the prep step has completed (golden and kernel inputs are now
on disk under `benchmark/`), run the three tiers inside containers
with the same bind mount. `MALLOC_ARENA_MAX=2` is set so glibc
arena overhead doesn't dominate the RSS measurements (see "Memory
measurement" below):

```bash
docker run --rm \
  -e MALLOC_ARENA_MAX=2 \
  -v "$(pwd)/benchmark:/t-route/benchmark" \
  troute-dev:bench \
  bash -c "cd /t-route && \
    python benchmark/bench_e2e.py    --runs 5  --warmup 2 --label tierA --json && \
    python benchmark/bench_kernel.py --runs 15 --warmup 3 --label tierB --json && \
    python benchmark/bench_conus.py  --profile none      --label tierC --json"
```

Per-tier notes:

- **Tier A** (`bench_e2e.py`) reports wall/CPU/RSS over the 5 timed
  runs, compares output to `golden/`, prints `PASS` / `FAIL`. Save a
  fresh golden any time via `--save-golden`.
- **Tier B** (`bench_kernel.py`) replays the harvested
  `compute_network_structured()` calls only: no I/O, no joblib, no
  config parsing. Use this to measure pure MC-kernel changes.
- **Tier C** (`bench_conus.py`) does a single CONUS run. Swap
  `--profile none` for `--profile cprofile` to write
  `results/<label>.conus.pstats`, or `--profile pyspy` for an SVG
  flamegraph (the latter requires the container to have `--privileged`
  or `--cap-add SYS_PTRACE`).

### Regenerating the bar charts in `RESULTS.md`

```bash
docker run --rm \
  -v "$(pwd)/benchmark:/t-route/benchmark" \
  troute-dev:bench \
  python /t-route/benchmark/generate_figures.py
```

The script reads its numbers from constants at the top of the file
(see `BASELINE_*` / `AFTER_*` blocks); update those, rerun, and
commit the new PNGs in `figures/`.

### VS Code Dev Containers workflow

The repo's `.devcontainer/devcontainer.json` resolves the
`/hydrofabric` bind-mount source from the host environment
variable `TROUTE_HYDROFABRIC_DIR`, falling back to
`/tmp/troute-no-hydrofabric` (which the devcontainer's
`initializeCommand` creates on the host before container start)
so the container starts cleanly for non-benchmark users.

To run benchmarks from inside VS Code:

1. Export the env var in your shell **before** launching VS Code so
   it gets inherited:
   ```bash
   export TROUTE_HYDROFABRIC_DIR=/absolute/path/to/your/hydrofabric/dir
   code /path/to/t-route
   ```
2. "Reopen in Container" picks up the mount automatically; the
   geopackage is then visible at `/hydrofabric` inside the
   container.
3. Set `MALLOC_ARENA_MAX=2` in your shell or in the `containerEnv`
   block of `devcontainer.json` so every terminal in the container
   inherits it.
4. From an integrated terminal the bench commands are the same as
   above except you drop the outer `docker run` shell, since you
   are already inside the container.

If `TROUTE_HYDROFABRIC_DIR` is unset, the container still starts
with `/hydrofabric` mapped to the empty `/tmp/troute-no-hydrofabric`
directory; only the benchmark prep scripts will complain.

## Notes

- All measurements were taken inside the devcontainer on linux/arm64.
  Relative speedups generalize; absolute seconds will shift with the
  host CPU and the container runtime's I/O overhead.
- `cpu_pool=1` in `nhf_subset_ohio.yaml` and `cpu_pool=8` in
  `conus.yaml` is deliberate. Tier A measures kernel-dominated
  single-thread cost; Tier C measures whether the workers are
  well-fed.
- cProfile inflates Python-loop self-time by 2-5x. Use clean runs
  (`--profile none`) for headline numbers; reserve cProfile for
  attribution.
