# Conecuh Test Case

The following file is meant to test the NHF on a test case in Alabama on USGS Gage 02374250. To run the test, do the following commands from the home directory:

```sh
uv venv --python 3.10
source .venv/bin/activate
./compiler.sh --uv
cd test/nhf/conecuh_case
uv pip install -r requirements.txt
```
