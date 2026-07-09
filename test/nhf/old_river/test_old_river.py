from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from ..utils.integration_helpers import (
    assert_peak_bounds,
    delete_outputs,
    has_files,
    run_troute,
    skip_if_not_built,
)
from ..utils.generate_reference_data import generate_reference_data
from ..utils.make_configs import Config
from ..utils.make_forcing import build_forcing_dataset, create_hot_start_file
from ..utils.subset_nhf import get_offnetwork_upstreams, extract_layers, write_gpkg

FP_IDS = [
    1269985531956909,
    1269989421970703,
    1270477057754811,
    1270477071348474,
    1270477162655082,
    1270478408627136,
    1270478480769474,
    1270478443545299,
    1270478544606477,
    1270479710852096,
    1270478559871774,
    1270481040988413,
    1270482252262935,
    1270482207072236,
    1269974759984431,
    1269977357089802,
    1269978640043814,
    1269978588129958,
    1269978508717011,
    1269978605187084,
    1269979803636892,
    1269979801723785,
    1269980994441113,
    1269981094844870,
    1269982274955620,
    1269982288653121,
    1269983535480997,
    1269983486734837,
    1269983407894074,
    1269983415902697,
    1269983275710118,
    1269985856917587,
    1269987150880464,
    1269987179564045,
    1269988420397162,
    1269988372956327,
    1269988326884596,
    1269989538245521,
    1269989622554125,
    1270477248837734,
    1270477334887319,
    1270478607299993,
    1270478625399418,
    1270479816524705,
    1270481161746904,
    1270482410241539,
    1270483792118807,
    1270485077827397,
    1270486400302327,
    1270487708931571,
    1270490204531832,
    1270490290599818,
    1270491588752225,
    1270491644483850,
    1270492969620749,
    1270495491254275,
    1270496783022913,
    1270498170206546,
    1270499546086799,
    1270499583677310,
    1270500815903050,
    1270502177889062,
    1270502329921098,
    1270989956036842,
    1270989989038756,
    1270991250585901,
    1270991287045452,
    1270992549474126,
    1270992623454142,
    1271018265769831,
    1271018298184438,
    1271020831654835,
    1271022201616581,
    1270999001416760,
]
START_TIME = "2011-04-14 00:00"
END_TIME = "2011-06-30 00:00"
FORCING_MODE = "retro"


RUNOUT_PERIOD = int(
    (pd.Timestamp(END_TIME) - pd.Timestamp(START_TIME)).total_seconds() / 3600 / 2
)
END_TIME_WITH_RUNOUT = (
    pd.Timestamp(END_TIME) + pd.Timedelta(hours=RUNOUT_PERIOD)
).strftime("%Y-%m-%d %H:%M")

DATA_DIR = Path(__file__).parent / "data"
CFG = Config(DATA_DIR, START_TIME, END_TIME_WITH_RUNOUT, restart_dir_name="restart")

GAGES_PATCH = {
    "07381482": {"fp_id": 1270478544606477, "virtual_fp_id": 1270478544606478},
    "07381490": {"fp_id": 1269985531956909, "virtual_fp_id": 1269985531956910},
}


def patch_gages(domain_path: Path) -> None:
    """Manually patch fp_id and virtual_fp_id in the gages table."""
    gages = gpd.read_file(domain_path, layer="gages")
    for site_no, fields in GAGES_PATCH.items():
        for field, value in fields.items():
            gages.loc[gages["site_no"] == site_no, field] = value
    gages.to_file(domain_path, layer="gages", driver="GPKG")


def setup(source_gpkg: str | Path, refresh: bool = True):
    """Subset the NHF domain and generate forcing for a standard test case."""
    offnetwork_upstreams = None
    if refresh or not CFG.config_path.exists():
        CFG.write_yaml()

    if refresh or not CFG.domain_path.exists():
        offnetwork_upstreams = get_offnetwork_upstreams(source_gpkg, FP_IDS)
        layers = extract_layers(source_gpkg, FP_IDS + offnetwork_upstreams)
        write_gpkg(layers, CFG.domain_path)
        patch_gages(CFG.domain_path)

    if refresh or not has_files(CFG.channel_forcing_dir, CFG.qlat_file_pattern):
        if offnetwork_upstreams is None:
            offnetwork_upstreams = get_offnetwork_upstreams(source_gpkg, FP_IDS)
        build_forcing_dataset(
            FORCING_MODE,
            START_TIME,
            END_TIME,
            CFG.channel_forcing_dir,
            CFG.domain_path,
            RUNOUT_PERIOD,
            offnetwork_upstreams=offnetwork_upstreams
        )

    restart_file = CFG.root_dir / CFG.restart_dir_name / "restart.pkl"
    if refresh or not restart_file.exists():
        if offnetwork_upstreams is None:
            offnetwork_upstreams = get_offnetwork_upstreams(source_gpkg, FP_IDS)
        create_hot_start_file(
            t_start=START_TIME,
            restart_dir=str(CFG.root_dir / CFG.restart_dir_name),
            hydrofabric_path=str(CFG.domain_path),
            offnetwork_upstreams=offnetwork_upstreams
        )

    if refresh or not CFG.reference_data_path.exists():
        generate_reference_data(
            hydrofabric_path=CFG.domain_path,
            t_start=pd.Timestamp(START_TIME),
            t_end=pd.Timestamp(END_TIME),
            output_dir=CFG.reference_data_path.parent,
        )


@pytest.mark.integration
def test_patuxent():
    skip_if_not_built(CFG)
    delete_outputs(CFG.output_dir)
    run_troute(CFG.config_path)
    assert_peak_bounds(CFG.output_dir, PEAK_BOUNDS)


# python -m test.nhf.utils.generate_diagnostics -f test/nhf/old_river/data/config.yaml