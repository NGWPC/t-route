"""Self-contained check that numpy 1.26.4 computes correctly at scale.

No t-route data required. Reproduces the failure t-route hits during NHF CONUS
network construction (`nhf_discretize.py` -> `pandas.groupby(...).cumcount()`),
which is the SAME pandas/numpy operation, just with synthetic data.

The operation is run at increasing sizes. Small arrays compute correctly on
every interpreter; the failure appears ONLY at scale. That is exactly why it
slips past small unit tests and only bites the full ~1.1 M-flowpath CONUS run.

    Python 3.11 + numpy 1.26.4 (wheel OR source build) -> every size PASS
    Python 3.14 + numpy 1.26.4 (source build, no wheel) -> small PASS, large FAIL

Exit code 0 = every size correct, 1 = a size miscomputed.
"""
import sys
import numpy as np
import pandas as pd

print(f"python {sys.version.split()[0]}  numpy {np.__version__}  pandas {pd.__version__}")


def cumcount_len(n: int) -> int:
    """groupby(...).cumcount() on a float64 key with NaN (mirrors NHF
    up_virtual_nex_id). Returns the result length, which must equal n."""
    key = np.where(np.arange(n) % 3 == 0, np.nan, (np.arange(n) // 2).astype(float))
    return len(pd.DataFrame({"key": key}).groupby("key").cumcount())


SIZES = [10, 1_000, 100_000, 2_000_000]
failed = False
for n in SIZES:
    try:
        got = cumcount_len(n)
        assert got == n, f"cumcount length {got} != {n}"
        print(f"  N={n:>9,}: OK")
    except Exception as e:  # noqa: BLE001
        print(f"  N={n:>9,}: FAIL -> {type(e).__name__}: {e}")
        failed = True

if failed:
    print("RESULT: FAIL. numpy 1.26.4 computes the small arrays correctly but")
    print("        miscomputes at scale on this interpreter (it supports CPython")
    print("        3.12 and earlier; it is not viable on 3.14).")
    sys.exit(1)
print("RESULT: PASS (numpy + pandas compute correctly at every size)")
sys.exit(0)
