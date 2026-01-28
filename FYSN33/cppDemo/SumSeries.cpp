/*
 * SumSeries.cpp
 *
 * Demonstration of exposing a simple numerical loop to Python via ctypes.
 *
 * Provides a single function:
 *     extern "C" double sum_sin_series(std::uint64_t n);
 *
 * The function computes the sum of sin(x) over n iterations, where
 * x cycles through values in [0, 1) with step 0.001. It is written
 * in C++ for performance and compiled as a shared library (.so/.dylib/.dll).
 *
 * Intended as a teaching example: shows how moving tight loops to
 * native code can yield large speedups while keeping Python as the
 * high-level driver.
 */

#include <cmath>
#include <cstdint>

extern "C" double sum_sin_series(std::uint64_t n) {
    double s = 0.0;
    for (std::uint64_t i = 0; i < n; ++i) {
        double x = (i % 1000) * 0.001; // cycles through 0.000..0.999
        s += std::sin(x);
    }
    return s;
}
