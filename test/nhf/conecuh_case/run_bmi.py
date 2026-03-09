"""
Run the Conecuh NHF test case via the BMI path.

Usage:
    cd test/nhf/conecuh_case
    uv run python run_bmi.py
"""
import os
import sys
import glob

import numpy as np
import pandas as pd

from troute_nwm_bmi.troute_bmi import BmiTroute

CONFIG_FILE = "test_case_bmi.yaml"
FORCING_DIR = "channel_forcing"
FORCING_PATTERN = "*.CHRTOUT_DOMAIN1.csv"


def main():
    # Initialize BMI model
    model = BmiTroute()
    model.initialize(CONFIG_FILE)

    dt = model.get_time_step()  # 300s

    # Collect and sort forcing files by timestamp
    forcing_files = sorted(glob.glob(os.path.join(FORCING_DIR, FORCING_PATTERN)))
    if not forcing_files:
        print(f"No forcing files found in {FORCING_DIR}/{FORCING_PATTERN}")
        sys.exit(1)

    print(f"Found {len(forcing_files)} forcing files, dt={dt}s")

    # Read first file to get IDs (all files share the same feature_id index)
    first_df = pd.read_csv(forcing_files[0]).set_index("feature_id")
    feature_ids = np.array(first_df.index, dtype=np.intc)
    n_features = len(feature_ids)

    # Resize BMI arrays to match number of features
    model.set_value("land_surface_water_source__id__count", np.array([n_features], dtype=np.int64))
    model.set_value("land_surface_water_source__volume_flow_rate__count", np.array([n_features], dtype=np.int64))

    # Set the IDs (constant across all timesteps)
    model.set_value("land_surface_water_source__id", feature_ids)

    # Feed forcing data: one CSV per hour, update_until advances 12 dt-steps per hour
    for i, fpath in enumerate(forcing_files):
        df = pd.read_csv(fpath).set_index("feature_id")
        flow_values = np.array(df.iloc[:, 0], dtype=float)

        model.set_value("land_surface_water_source__volume_flow_rate", flow_values)

        # Advance model by one hour (3600s / dt = 12 update() calls)
        target_time = (i + 1) * 3600.0
        model.update_until(target_time)

        if (i + 1) % 24 == 0:
            print(f"  Processed {i + 1}/{len(forcing_files)} hours")

    # Finalize triggers routing computation and output writing
    print("Running routing computation...")
    model.finalize()
    print("Done. Check output/ directory for results.")


if __name__ == "__main__":
    main()
