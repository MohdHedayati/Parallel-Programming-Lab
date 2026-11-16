// Benchmark for 2f (Pi calculation)
//   Task 1: same data (steps), increasing threads
//   Task 2: same threads, increasing data (steps)
//   Task 3: sequential vs parallel
// Compile: g++ -O2 -fopenmp bench_2f_pi.cpp -o bench_2f
// Run:     ./bench_2f
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

template <class F>
double timeIt(F f) {  // best of 3 runs, in seconds
    double best = 1e30;
    for (int r = 0; r < 3; r++) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

double piSeq(long steps) {
    double step = 1.0 / steps, sum = 0.0;
    for (long i = 0; i < steps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return step * sum;
}

double piPar(long steps, int threads) {
    double step = 1.0 / steps, sum = 0.0;
    #pragma omp parallel for num_threads(threads) reduction(+ : sum) schedule(static)
    for (long i = 0; i < steps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return step * sum;
}

std::vector<int> threadCounts() {
    int mx = omp_get_max_threads();
    std::vector<int> t;
    for (int k = 1; k < mx; k *= 2) t.push_back(k);
    t.push_back(mx);
    return t;
}

int main() {
    const int maxThreads = omp_get_max_threads();
    volatile double sink;

    // ---------- Task 1: fixed steps, increasing threads ----------
    const long S1 = 400000000;
    std::cout << "=== 2f Task 1: fixed steps = " << S1 << ", increasing threads ===\n";
    double pi = 0;
    double tSeq = timeIt([&] { pi = piSeq(S1); sink = pi; });
    std::cout << std::left << std::setw(10) << "Threads" << std::setw(14) << "Time(s)"
              << std::setw(10) << "Speedup" << std::setw(12) << "Efficiency" << "Pi\n";
    std::cout << std::setw(10) << "seq" << std::fixed << std::setprecision(4) << std::setw(14)
              << tSeq << std::setw(10) << "1.00" << std::setw(12) << "-" << std::setprecision(10)
              << pi << "\n";
    for (int T : threadCounts()) {
        double p = 0;
        double t = timeIt([&] { p = piPar(S1, T); sink = p; });
        std::cout << std::setw(10) << T << std::setprecision(4) << std::setw(14) << t
                  << std::setprecision(2) << std::setw(10) << tSeq / t << std::setw(12)
                  << tSeq / t / T << std::setprecision(10) << p << "\n";
    }

    // ---------- Task 2 + 3: fixed threads, increasing steps, sequential vs parallel ----------
    std::cout << "\n=== 2f Task 2/3: fixed " << maxThreads
              << " threads, increasing steps (sequential vs parallel) ===\n";
    std::cout << std::left << std::setw(14) << "Steps" << std::setw(14) << "Seq(s)" << std::setw(14)
              << "Par(s)" << std::setw(10) << "Speedup" << "Pi (parallel)\n";
    for (long s : {10000L, 100000L, 1000000L, 10000000L, 100000000L, 400000000L}) {
        double a = 0, b = 0;
        double ts = timeIt([&] { a = piSeq(s); sink = a; });
        double tp = timeIt([&] { b = piPar(s, maxThreads); sink = b; });
        std::cout << std::left << std::setw(14) << s << std::fixed << std::setprecision(6)
                  << std::setw(14) << ts << std::setw(14) << tp << std::setprecision(2)
                  << std::setw(10) << ts / tp << std::setprecision(10) << b << "\n";
    }
    return 0;
}
