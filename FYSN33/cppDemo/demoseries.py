import math, time, ctypes, os, sys

# Load the shared library cross-platform
libname = "libsumseries.dylib" if sys.platform == "darwin" else "libsumseries.so"

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), libname))

# Declare signature: double sum_sin_series(uint64_t)
lib.sum_sin_series.argtype = (ctypes.c_uint64)
lib.sum_sin_series.restype  = ctypes.c_double

# The pure Python version of the method.
def py_sum_sin_series(n: int) -> float:
    s = 0.0
    for i in range(n):
        x = (i % 1000) * 0.001
        s += math.sin(x)
    return s

N = 5_000_000

# Time the execution
t0 = time.perf_counter()
s_py = py_sum_sin_series(N)
t1 = time.perf_counter()
s_c = lib.sum_sin_series(N)
t2 = time.perf_counter()

# Print the result
print(f"Python result: {s_py:.6f} in {t1 - t0:.3f}s")
print(f"C/C++   result: {s_c:.6f} in {t2 - t1:.3f}s")
print(f"Speedup: {(t1 - t0)/(t2 - t1):.1f}×")

