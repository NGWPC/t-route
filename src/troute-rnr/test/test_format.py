import shutil

import pandas as pd
import yaml
from troute.config import Config
from troute_rnr import format, nwps
from troute_rnr.settings import Settings


def test_format_config(
    mock_rfc_inputs: nwps.ProcessedData, mock_settings: Settings, mock_flows: pd.DataFrame
) -> None:
    """Tests to make sure all t-route inputs are formatted correctly

    Parameter
    ---------
    mock_rfc_inputs: nwps.ProcessedData
        The preprocessed RFC inputs
    mock_settings: Settings
        The settings for the t-route formatting
    mock_flows: pd.DataFrame
        The correct flow values
    """
    yaml_file_path, tmp_flow_files_path = format.format_config(mock_rfc_inputs, mock_settings)

    with open(yaml_file_path) as custom_file:
        data = yaml.load(custom_file, Loader=yaml.SafeLoader)

    # Testing to make sure config will validate in strict mode
    _ = Config.with_strict_mode(**data)

    first_flows_df = pd.read_csv(tmp_flow_files_path / "202505052100.CHRTOUT_DOMAIN1.csv")
    pd.testing.assert_frame_equal(
        first_flows_df,
        mock_flows,
        check_dtype=True,
        check_index_type=False,
        check_column_type=False,
        rtol=1e-5,
    )

    yaml_file_path.unlink()
    mock_settings.tmp_geopackage.unlink()
    shutil.rmtree(tmp_flow_files_path)
