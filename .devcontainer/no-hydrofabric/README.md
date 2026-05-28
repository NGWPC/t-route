# No-hydrofabric placeholder

This directory is the **default bind-mount source** for `/hydrofabric`
inside the devcontainer. It is intentionally empty.

The benchmark workflow needs a real NextGen Hydrofabric geopackage
mounted into the container. The Dev Containers config at
`.devcontainer/devcontainer.json` resolves the mount source from the
host environment variable `TROUTE_HYDROFABRIC_DIR`; if you have not
set it, the source falls back to this directory and the container
still starts (with an empty `/hydrofabric`).

## To enable benchmarks in VS Code Dev Containers

Set the env var in your shell (or your shell's rc file) **before**
launching VS Code:

```bash
export TROUTE_HYDROFABRIC_DIR=/absolute/path/to/your/hydrofabric/dir
# then launch VS Code so it inherits the env var
code /path/to/t-route
```

Then "Reopen in Container" picks up the mount automatically. Inside
the container the data is visible at `/hydrofabric`.

If you prefer a one-shot benchmark run without changing your
devcontainer setup, run the benchmark from the host with
`docker run -v ...` (see `benchmark/README.md`).
