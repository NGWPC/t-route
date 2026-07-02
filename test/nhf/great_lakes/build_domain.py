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
import geopandas as gpd

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
VFP_LINKAGE = {
    "4800002": 1278346877373953,
    "4800004": 1276364270423160,
    "4800006": 1286154743979494,
    "4800007": 1287248166320950
}


def patch_gpkg_lakes(gpkg_path: str) -> None:
    """Patch Great Lakes fp_id and virtual_fp_id values in-place."""
    linkages = {"fp_id": FP_LINKAGE, "virtual_fp_id": VFP_LINKAGE}
    lake_id_list = ", ".join(FP_LINKAGE.keys())  # same keys for both

    # Quick check: skip if all columns already match.
    gl = gpd.read_file(gpkg_path, layer="lakes", where=f"lake_id IN ({lake_id_list})")
    if all(
        row[col] == {int(k): v for k, v in mapping.items()}[int(row["lake_id"])]
        for col, mapping in linkages.items()
        for _, row in gl.iterrows()
    ):
        return

    gdf = gpd.read_file(gpkg_path, layer="lakes")
    dirty = False
    for col, mapping in linkages.items():
        old = gdf[col].copy()
        gdf[col] = gdf["lake_id"].map(mapping).fillna(gdf[col]).astype(gdf[col].dtype)
        changed = ~((gdf[col] == old) | (gdf[col].isna() & old.isna()))
        for lake_id, old_val, new_val in (
            gdf.loc[changed, ["lake_id", col]]
            .assign(old_col=old[changed])[["lake_id", "old_col", col]]
            .itertuples(index=False)
        ):
            print(f"  [patch_gpkg] lake_id {lake_id}: {col} {old_val} -> {new_val}")
        dirty |= changed.any()

    if dirty:
        gdf.to_file(gpkg_path, layer="lakes", driver="GPKG")


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
    patch_gpkg_lakes(args.source_gpkg)

    gl_fps = FP_LINKAGE.values()
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
