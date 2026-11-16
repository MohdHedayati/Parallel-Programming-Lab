// Benchmark for 2e (Largest element in an array)
//   Task 1: same data (n), increasing threads
//   Task 2: same threads, increasing data (n)
//   Task 3: sequential vs parallel
// Compile: g++ -O2 -fopenmp bench_2e_max.cpp -o bench_2e
// Run:     ./bench_2e
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <climits>
#include <algorithm>
#include <chrono>
#include <omp.h>

// Best of 5 runs (this task is very short, so use more repeats)
template <class F>
double timeIt(F f) {
    double best = 1e30;
    for (int r = 0; r < 5; r++) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

int maxSeq(const std::vector<int>& a, long n) {
    int m = INT_MIN;
    for (long i = 0; i < n; i++)
        if (a[i] > m) m = a[i];
    return m;
}

int maxPar(const std::vector<int>& a, long n, int threads) {
    int m = INT_MIN;
    #pragma omp parallel for num_threads(threads) reduction(max : m)
    for (long i = 0; i < n; i++)
        if (a[i] > m) m = a[i];
    return m;
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
    const long BIG = 100000000;  // 100 million ints (~400 MB)

    std::vector<int> a(BIG);
    std::mt19937 gen(1);
    std::uniform_int_distribution<int> dist(-1000000000, 1000000000);
    for (auto& x : a) x = dist(gen);

    volatile int sink;  // stops the compiler from deleting the work

    // ---------- Task 1: fixed n, increasing threads ----------
    std::cout << "=== 2e Task 1: fixed n = " << BIG << ", increasing threads ===\n";
    int expected = 0;
    double tSeq = timeIt([&] { expected = maxSeq(a, BIG); sink = expected; });
    std::cout << std::left << std::setw(10) << "Threads" << std::setw(14) << "Time(s)"
              << std::setw(10) << "Speedup" << "Efficiency\n";
    std::cout << std::setw(10) << "seq" << std::fixed << std::setprecision(5) << std::setw(14)
              << tSeq << std::setw(10) << "1.00" << "-\n";
    for (int T : threadCounts()) {
        int r = 0;
        double t = timeIt([&] { r = maxPar(a, BIG, T); sink = r; });
        std::cout << std::setw(10) << T << std::setprecision(5) << std::setw(14) << t
                  << std::setprecision(2) << std::setw(10) << tSeq / t << tSeq / t / T
                  << (r == expected ? "" : "  <-- MISMATCH") << "\n";
    }

    // ---------- Task 2 + 3: fixed threads, increasing n, sequential vs parallel ----------
    std::cout << "\n=== 2e Task 2/3: fixed " << maxThreads
              << " threads, increasing n (sequential vs parallel) ===\n";
    std::cout << std::left << std::setw(14) << "n" << std::setw(14) << "Seq(s)" << std::setw(14)
              << "Par(s)" << "Speedup\n";
    for (long n : {1000L, 10000L, 100000L, 1000000L, 10000000L, 100000000L}) {
        int x = 0, y = 0;
        double ts = timeIt([&] { x = maxSeq(a, n); sink = x; });
        double tp = timeIt([&] { y = maxPar(a, n, maxThreads); sink = y; });
        std::cout << std::left << std::setw(14) << n << std::fixed << std::setprecision(6)
                  << std::setw(14) << ts << std::setw(14) << tp << std::setprecision(2)
                  << ts / tp << (x == y ? "" : "  <-- MISMATCH") << "\n";
    }
    return 0;
}
