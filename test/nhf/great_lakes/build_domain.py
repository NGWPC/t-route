"""Build the Great Lakes test domain for ``run_test.py``.

The Great Lakes drainage is far too large to subset by outlet (the documented
``subset_nhf.py --outlet-fp-id`` approach), so this carves a small slice instead:
the fp_id-bearing Great Lakes (4800002/4800004/4800006 -- 4800007/Lake Ontario
has no fp_id and is forced via the Ontario outflow file) plus a few downstream
hops, which is all ``run_test.py`` needs to watch the DA-forced outflow
propagate.

A pre-built ``domain/nhf.gpkg`` is committed so the test runs out of the box;
re-run this to regenerate it from a newer NHF release:

    python build_domain.py --source-gpkg /path/to/nhf.gpkg
"""
import argparse
import sqlite3
import sys
from pathlib import Path

# subset_nhf lives one directory up (test/nhf/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subset_nhf as S  # noqa: E402

# Great Lakes lake ids that carry an fp_id and can be anchored to a flowpath.
GREAT_LAKES_FP_BEARING_IDS = (4800002, 4800004, 4800006, 4800007)
FP_LINKAGE = {
    "4800002": 1278348162056612,
    "4800004": 1276364270499315,
    "4800006": 1286192735893685,
    "4800007": 1287248237297035
}

def great_lakes_fp_ids(source_gpkg: str) -> list[int]:
    conn = sqlite3.connect(source_gpkg)
    placeholders = ",".join("?" * len(GREAT_LAKES_FP_BEARING_IDS))
    rows = conn.execute(
        f"SELECT fp_id FROM lakes WHERE lake_id IN ({placeholders}) "
        "AND fp_id IS NOT NULL",
        GREAT_LAKES_FP_BEARING_IDS,
    ).fetchall()
    conn.close()
    return [int(r[0]) for r in rows]

def patch_gpkg_lakes_fp_ids(gpkg_path: str) -> None:
    """Temporary fix: UPDATE the lakes table in-place so each Great Lake's fp_id
    matches FP_LINKAGE.

    NHF releases occasionally assign a Great Lake's fp_id to a flowpath that
    doesn't survive the domain subset, leaving the lake with a stale fp_id that
    later causes an IndexError in _build_div_weighting_matrix.  Patching the
    geopackage directly is the simplest fix until NHF corrects the source data.
    """
    conn = sqlite3.connect(gpkg_path)
    try:
        # GeoPackage triggers use ST_IsEmpty (a spatialite function) which
        # plain sqlite3 doesn't have.  Drop them temporarily, patch, recreate.
        triggers = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger' AND tbl_name='lakes'"
        ).fetchall()
        for name, _ in triggers:
            conn.execute(f"DROP TRIGGER IF EXISTS \"{name}\"")

        for lake_id, correct_fp_id in FP_LINKAGE.items():
            row = conn.execute(
                "SELECT fp_id FROM lakes WHERE lake_id = ?", (lake_id,)
            ).fetchone()
            if row is None or row[0] is None or int(row[0]) == correct_fp_id:
                continue
            print(f"  [patch_gpkg] lake_id {lake_id}: fp_id {int(row[0])} -> {correct_fp_id}")
            conn.execute(
                "UPDATE lakes SET fp_id = ? WHERE lake_id = ?",
                (correct_fp_id, lake_id),
            )

        for _, sql in triggers:
            conn.execute(sql)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-gpkg", required=True,
                    help="full NHF geopackage to carve the domain from")
    ap.add_argument("--out-gpkg", default="domain/nhf.gpkg",
                    help="output domain path (default: domain/nhf.gpkg)")
    ap.add_argument("--downstream-hops", type=int, default=30,
                    help="how many hops below each Great Lake to include "
                         "(default 30)")
    args = ap.parse_args()

    ### TEMPORARY PATCH UNTIL NHF 1.2.2 ###
    patch_gpkg_lakes_fp_ids(args.source_gpkg)

    gl_fps = great_lakes_fp_ids(args.source_gpkg)
    if not gl_fps:
        sys.exit("ERROR: no fp_id-bearing Great Lakes found in the source "
                 "geopackage (expected lake_ids 4800002/4800004/4800006)")
    print(f"Great Lakes flowpaths: {gl_fps}")

    fp_ids = S.get_downstream_fp_ids(args.source_gpkg, gl_fps, args.downstream_hops)
    print(f"downstream domain (<= {args.downstream_hops} hops): {len(fp_ids)} flowpaths")

    layers = S.extract_layers(args.source_gpkg, fp_ids)
    S.write_gpkg(layers, Path(args.out_gpkg))


if __name__ == "__main__":
    main()
