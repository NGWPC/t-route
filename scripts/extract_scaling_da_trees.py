"""Extract simple-scaling DA gage trees from an NHF >= 1.2 geopackage.

Standalone validation of the simple-scaling tree builder on real NextGen
HydroFabric topology. Reads the *flowpath* network straight from the gpkg via
sqlite3 (a gpkg is a SQLite db -- no geopandas/fiona needed, CONUS-fast):

  - flowpaths: fp_id, fp_to_id (downstream), total_da_sqkm (drainage area)
  - gages:     site_no -> fp_id (and status)
  - lakes:     fp_id (waterbody stop set)

and drives :func:`troute.scaling_da.build_gage_trees_from_mappings`, then reports
tree-count, size distribution, coverage, and disjointness.

This exercises the same core builder the production ``build_gage_trees(network)``
adapter uses; it runs on the flowpath id space (the adapter runs on t-route's
discretized virtual-flowpath routing network once drainage area is plumbed).

Usage:
    pixi run python scripts/extract_scaling_da_trees.py \
        --gpkg /path/to/nhf_1.2.0.gpkg
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from troute.scaling_da import build_gage_trees_from_mappings


def _is_terminal(fp_to_id: object, fp_id: int) -> bool:
    if fp_to_id is None:
        return True
    try:
        v = int(fp_to_id)
    except (TypeError, ValueError):
        return True
    return v <= 0 or v == fp_id


def load_mappings(gpkg: Path, *, active_only: bool) -> tuple[dict, dict, dict, frozenset]:
    """Read flowpath connectivity, drainage area, gages, and lakes from the gpkg."""
    con = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
    cur = con.cursor()

    area_sqkm: dict[int, float] = {}
    fp_down: dict[int, int] = {}
    fp_ids: set[int] = set()
    for fp_id, fp_to_id, da in cur.execute(
        "SELECT fp_id, fp_to_id, total_da_sqkm FROM flowpaths"
    ):
        if fp_id is None:
            continue
        fp = int(fp_id)
        fp_ids.add(fp)
        area_sqkm[fp] = float(da) if da is not None else np.nan
        if not _is_terminal(fp_to_id, fp):
            fp_down[fp] = int(fp_to_id)

    # Reverse connectivity: downstream -> [upstream...], keeping only edges whose
    # downstream endpoint is a known flowpath.
    rconn: dict[int, list[int]] = defaultdict(list)
    resolved = 0
    for fp, dn in fp_down.items():
        if dn in fp_ids:
            rconn[dn].append(fp)
            resolved += 1
    rconn = dict(rconn)

    # Gage status values vary by hydrofabric build; report the breakdown and,
    # with --active-only, keep just the 'active'-flavored ones.
    status_counts: dict[str, int] = defaultdict(int)
    gage_to_fp: dict[str, int] = {}
    for site_no, fp_id, status in cur.execute("SELECT site_no, fp_id, status FROM gages"):
        if site_no is None or fp_id is None:
            continue
        s = "" if status is None else str(status).strip().lower()
        status_counts[s or "<empty>"] += 1
        if active_only and s not in ("active", "1", "true", "a", ""):
            continue
        gage_to_fp[str(site_no)] = int(fp_id)

    waterbody_segs: set[int] = set()
    for (fp_id,) in cur.execute("SELECT fp_id FROM lakes"):
        if fp_id is not None:
            waterbody_segs.add(int(fp_id))

    con.close()
    print(
        f"  flowpaths={len(fp_ids):,}  edges_resolved={resolved:,}  "
        f"gages={len(gage_to_fp):,}  lake_fps={len(waterbody_segs):,}"
    )
    print(f"  gage status breakdown: {dict(status_counts)}")
    return rconn, gage_to_fp, area_sqkm, frozenset(waterbody_segs)


def report(trees: dict, gage_to_fp: dict) -> int:
    if not trees:
        print("No trees built.", file=sys.stderr)
        return 1
    sizes = np.array([t.n_segments for t in trees.values()])
    order = np.argsort(sizes)
    pct = {p: int(np.percentile(sizes, p)) for p in (50, 90, 99)}
    print("\n=== tree summary ===")
    print(f"  trees built:        {len(trees):,} / {len(gage_to_fp):,} gages")
    print(f"  segments covered:   {int(sizes.sum()):,}")
    print(f"  tree size  min/median/p90/p99/max: "
          f"{int(sizes.min())}/{pct[50]}/{pct[90]}/{pct[99]}/{int(sizes.max())}")

    # Disjointness: the stop rule guarantees it; verify on real data.
    seg_owner: dict[int, str] = {}
    overlaps = 0
    for site, tree in trees.items():
        for seg in tree.seg_order.tolist():
            if seg in seg_owner:
                overlaps += 1
            else:
                seg_owner[seg] = site
    print(f"  overlapping segments (should be 0): {overlaps:,}")

    biggest_site = list(trees)[int(order[-1])]
    bt = trees[biggest_site]
    print(f"  largest tree: site {biggest_site} at fp {bt.gage_fp} "
          f"({bt.n_segments:,} segments, A_o={bt.gage_area_sqkm:,.0f} km^2)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpkg", type=Path, required=True,
                    help="path to an NHF >= 1.2 geopackage")
    ap.add_argument("--theta", type=float, default=0.77)
    ap.add_argument("--active-only", action="store_true",
                    help="keep only active-status gages (default: all gages with a valid fp_id)")
    args = ap.parse_args(argv)

    if not args.gpkg.exists():
        print(f"gpkg not found: {args.gpkg}", file=sys.stderr)
        return 2

    print(f"=== reading {args.gpkg} ===")
    t0 = time.time()
    rconn, gage_to_fp, area_sqkm, waterbody_segs = load_mappings(
        args.gpkg, active_only=args.active_only
    )
    print(f"  read in {time.time() - t0:.1f}s")

    t1 = time.time()
    trees = build_gage_trees_from_mappings(
        rconn, gage_to_fp, area_sqkm, waterbody_segs=waterbody_segs, theta_default=args.theta
    )
    print(f"  built {len(trees):,} trees in {time.time() - t1:.1f}s")

    return report(trees, gage_to_fp)


if __name__ == "__main__":
    raise SystemExit(main())
