# numpy 1.26.4 is not viable on Python 3.14

This branch (`test/numpy-from-source`) is **unmodified `development`** plus one
Dockerfile (`docker/Dockerfile.numpy-from-source-test`) and one check script
(`check_numpy_py314.py`). It demonstrates that t-route runs correctly on
**Python 3.11 + numpy 1.26.4** but fails on **Python 3.14 + numpy 1.26.4**, and
that the cause is numpy 1.26.4 itself, not t-route and not the build config.

## Summary

| Python | numpy | install method | result |
|--------|-------|----------------|--------|
| **3.11** | 1.26.4 | manylinux **wheel** | PASS (CONUS clean; check PASS) |
| **3.11** | 1.26.4 | **compiled from source** (same gcc) | PASS (check PASS) |
| **3.14** | 1.26.4 | **compiled from source** (no wheel) | FAIL |
| **3.14** | 1.26.4 | **source, no BLAS at all** (`-Dallow-noblas`, lapack_lite) | FAIL (BLAS ruled out) |
| **3.14** | **2.5.0** | **compiled from source** (same gcc) | PASS (check PASS) |
| **3.14** | 2.5.0 | wheel | PASS (CONUS clean; check PASS) |

The key control is the pair of source builds on **the same py3.14 with the same
gcc**: numpy **1.26.4 source fails** but numpy **2.5 source passes**. The fix is
in **numpy 2's source** (the CPython 3.13/3.14 C-API patches), not in how a wheel
is built. numpy 1.26.4 does not contain those patches and never will.

numpy 1.26.4's last wheels are for **CPython 3.12 and earlier**. On 3.14 it must
be built from source against a Python it does not support; that build imports
fine and passes small cases, but **miscomputes at scale**, crashing t-route's NHF
CONUS network build:

```
File "troute/nhf_discretize.py", line 167, in discretize_flowpaths
    dup_up_node = virtual_flowpaths.groupby(FIELD_UP_VIRTUAL_NEX_ID).cumcount()
  ...
ValueError: operands could not be broadcast together with shape (575447,) (1643409,)
```

**Recommendation: do NOT run Python 3.14 with numpy 1.26.4. Either:**

1. upgrade to **numpy 2.3.4 or newer** ([first](https://numpy.org/doc/stable/release/2.3.4-notes.html)
   release to officially support Python 3.14.0), or
2. stay on **Python 3.11** with numpy 1.26.4.

## Reproduce (no t-route data needed for the quick check)

```bash
# Build the same source two ways (only the Python/numpy-install differs):
docker build -f docker/Dockerfile.numpy-from-source-test --build-arg PYVER=3.11 -t troute-np-py311 .
docker build -f docker/Dockerfile.numpy-from-source-test --build-arg PYVER=3.14 -t troute-np-py314 .
```

The build runs `check_numpy_py314.py`, which does `groupby().cumcount()` on a
float+NaN key (the same op as `nhf_discretize.py:167`, with synthetic data) at
increasing sizes (10, 1k, 100k, 2M). Small sizes compute correctly everywhere;
only the 2M case exposes the bug, which is why it slips past small tests and only
bites the full CONUS run:

* `troute-np-py311` build **succeeds**: every size OK, check prints `RESULT: PASS`.
* `troute-np-py314` build **fails at that step**: 10/1k/100k OK, 2M FAILs,
  check prints `RESULT: FAIL`.

Run the check standalone on an already-built image:

```bash
docker run --rm --entrypoint /opt/venv/bin/python troute-np-py311 check_numpy_py314.py   # PASS
docker run --rm --entrypoint /opt/venv/bin/python troute-np-py314 check_numpy_py314.py   # FAIL
```

Control: same py3.14 image, swap numpy 1.26.4 for **numpy 2.5 compiled from
source**, and the check passes (proving the fix is in numpy 2's source, not the
build environment):

```bash
docker run --rm --entrypoint bash troute-np-py314 -c \
  'pip install -q --force-reinstall --no-binary numpy "numpy==2.5.0" && python check_numpy_py314.py'
# prints: RESULT: PASS
```

### Full CONUS confirmation (NHF v1.2.0, the production dataset)

We can reproduce the same issue by running t-route for **NHF v1.2.0**
which has ~1.1 M flowpaths. The default py3.14 build fails at the
correctness check, so build that image with the check gated off first:

```bash
docker build -f docker/Dockerfile.numpy-from-source-test \
    --build-arg PYVER=3.14 --build-arg RUN_NUMPY_CHECK=0 -t troute-np-py314 .

# Mount the prepped CONUS benchmark dataset and run the CONUS case in each image:
docker run --rm -v "$PWD/benchmark:/t-route/benchmark" --entrypoint /opt/venv/bin/python \
    troute-np-py311 benchmark/bench_conus.py --profile none   # completes
docker run --rm -v "$PWD/benchmark:/t-route/benchmark" --entrypoint /opt/venv/bin/python \
    troute-np-py314 benchmark/bench_conus.py --profile none   # crashes at nhf_discretize.py:167
```

Verified on NHF v1.2.0 (1,102,154 flowpaths): py3.11 + numpy 1.26.4 wheel
completes (total 133 s; network build 65 s, routing 52 s); py3.14 + numpy 1.26.4
source crashes with the broadcast ValueError shown above. numpy 2 on py3.14 runs
CONUS clean.

## Why, and why it is not a fixable build issue

The failure was narrowed by elimination, all on this branch's images:

* **Not t-route**: the failing line is unmodified `development`, and it works on 3.11.
* **Not the build config, compiler, or system deps**: numpy 1.26.4 **compiled from
  source on 3.11** with the *same* Debian gcc and meson defaults **passes**. Only
  the Python version differs between pass and fail.
* **Not a numpy/pandas ABI mismatch**: no ABI warning on import; the numpy C-API
  version is identical (16777225) on 3.11 and 3.14; `factorize` and `bincount` are
  correct. (pandas is an official cp314 wheel and is not the culprit.)
* **Not dispatched SIMD**: `NPY_DISABLE_CPU_FEATURES=...` does not change the result.
* **Not BLAS/LAPACK**: numpy 1.26.4 built from source with **no BLAS at all**
  (`-Csetup-args=-Dallow-noblas=true`, lapack_lite, `ldd` confirms no libopenblas)
  fails identically. The broken op is core numpy (`np.repeat` in the grouper), not BLAS.
* **It is Python 3.14 + numpy 1.26.4 specifically**: numpy 1.26.4's C extensions,
  compiled against py3.14 headers, silently miscompute in hot loops at scale.
  Definitive control: building **numpy 2.5 from source** in the *same* py3.14 image
  with the *same* gcc makes the check pass. So the CPython 3.13/3.14 C-API changes
  this requires are fixed in numpy 2's **source**, and 1.26.4 lacks them. No build
  flag backports them to 1.26.4.

Corroboration: forcing the cumcount fix past line 167 only exposes a *second*,
non-deterministic corruption deeper in the NHF discretization
(`_discretize_links`), not reproducible in isolation, consistent with
memory/ABI-level corruption from compiling numpy 1.26.4 against an unsupported
CPython. numpy 2.5 on 3.14 has neither problem.
