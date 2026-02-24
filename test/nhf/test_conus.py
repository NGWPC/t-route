"""End to end tests for running t-route for NHF on the CONUS dataset."""
import subprocess
import sys

import pytest


class TestNHFConus:
    """Test class for running t-route for NHF on the CONUS dataset."""

    @pytest.mark.integration
    def test_nhf_conus(self):
        """Test running t-route for NHF on the CONUS dataset."""
        result = subprocess.run(
            [sys.executable, '-m', 'troute.NHF', '--config', 'test/nhf/test_conus_config.yaml'],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"NHF script failed with error: {result.stderr}"

    @pytest.mark.skip(reason="Need working t-route code first")
    def test_nhf_conus_courant(self):
        """Check that Courant condition is met for a random sample of reaches."""
        pass

    @pytest.mark.skip(reason="Need working t-route code first")
    def test_mass_conservation(self):
        """Check that mass is conserved for a random sample of reaches."""
        pass

    @pytest.mark.skip(reason="Need working t-route code first")
    def test_gage_accuracy(self):
        """Check that simulated streamflow matches observed streamflow at some USGS gages."""
        pass
