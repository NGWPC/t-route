# CONUS Test Case

This test runs t-route for a portion of the 2022 US summer floods over all of CONUS.  Channel forcing data is retrieved from National Water Model 3.0 retrospective data. To run the test, do the following commands from the home directory:

```sh
uv venv --python 3.10
source .venv/bin/activate
./compiler.sh --uv
cd test/nhf/conus
mkdir domain
ln -s /path/to/your/nhf.gpkg domain/nhf.gpkg
uv pip install -r requirements.txt
uv run python make_forcing.py
uv run python -m nwm_routing -V5 -f test_case.yaml
```
