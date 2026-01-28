import math, time, random, os, sys, ctypes

# Load the shared library
libname = "libgenerateparticles.dylib" if sys.platform == "darwin" else \
          ("libgenerateparticles.so" if os.name != "nt" else "generateparticles.dll")
libpath = os.path.join(os.path.dirname(__file__), libname)
lib = ctypes.CDLL(libpath)

# C function signature: double event_total_energy(int n, int seed)
lib.event_total_energy.argtypes = (ctypes.c_int, ctypes.c_int)
lib.event_total_energy.restype  = ctypes.c_double

# Pure Python baseline.
# The equivalent to the C++ Particle class.
class PyParticle:
    def __init__(self, m, px, py, pz):
        self.m, self.px, self.py, self.pz = m, px, py, pz
    def energy(self):
        return math.sqrt(self.m*self.m + self.px*self.px + self.py*self.py + self.pz*self.pz)

def py_total_energy(n: int, seed: int) -> float:
    rng = random.Random(seed)
    m = 0.13957
    s = 0.0
    for _ in range(n):
        p = PyParticle(m, rng.uniform(-2,2), rng.uniform(-2,2), rng.uniform(-2,2))
        s += p.energy()
    return s

# Demo
N = 1_000_000
SEED = 42

t0 = time.perf_counter()
E_py = py_total_energy(N, SEED)
t1 = time.perf_counter()

E_cpp = lib.event_total_energy(N, SEED)
t2 = time.perf_counter()

print(f"Python total energy: {E_py:.6f} in {t1 - t0:.3f}s")
print(f"C++    total energy: {E_cpp:.6f} in {t2 - t1:.3f}s")
print(f"Speedup: {(t1 - t0)/(t2 - t1):.1f}× (aggregation + generation)")
ok = math.isclose(E_py, E_cpp, rel_tol=5e-5, abs_tol=1e-6)
print(f"Results match: ", ok)

