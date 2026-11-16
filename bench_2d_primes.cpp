// Benchmark for 2d (Prime numbers between 2 and N, stored in an array)
//   Task 1: same data (N), increasing threads
//   Task 2: same threads, increasing data (N)
//   Task 3: sequential vs parallel
// Compile: g++ -O2 -fopenmp bench_2d_primes.cpp -o bench_2d
// Run:     ./bench_2d
#include <iostream>
#include <iomanip>
#include <vector>
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

bool isPrime(int n) {
    if (n < 2) return false;
    for (int i = 2; (long long)i * i <= n; i++)
        if (n % i == 0) return false;
    return true;
}

// Sequential: test every number, append primes to the array
size_t primesSeq(int N, std::vector<int>& out) {
    out.clear();
    for (int v = 2; v <= N; v++)
        if (isPrime(v)) out.push_back(v);
    return out.size();
}

// Parallel: every thread marks flag[v] (different cells, so no race),
// then the primes are collected into the array in order.
// schedule(dynamic) because large numbers take longer to test than small ones.
size_t primesPar(int N, int threads, std::vector<char>& flag, std::vector<int>& out) {
    flag.assign(N + 1, 0);
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 512)
    for (int v = 2; v <= N; v++)
        flag[v] = isPrime(v);
    out.clear();
    for (int v = 2; v <= N; v++)
        if (flag[v]) out.push_back(v);
    return out.size();
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
    std::vector<int> out;
    std::vector<char> flag;

    // ---------- Task 1: fixed N, increasing threads ----------
    const int N1 = 2000000;
    std::cout << "=== 2d Task 1: fixed N = " << N1 << ", increasing threads ===\n";
    double tSeq = timeIt([&] { primesSeq(N1, out); });
    size_t expected = out.size();
    std::cout << std::left << std::setw(10) << "Threads" << std::setw(14) << "Time(s)"
              << std::setw(10) << "Speedup" << std::setw(12) << "Efficiency" << "Primes found\n";
    std::cout << std::setw(10) << "seq" << std::fixed << std::setprecision(4) << std::setw(14)
              << tSeq << std::setw(10) << "1.00" << std::setw(12) << "-" << expected << "\n";
    for (int T : threadCounts()) {
        size_t cnt = 0;
        double t = timeIt([&] { cnt = primesPar(N1, T, flag, out); });
        std::cout << std::setw(10) << T << std::setprecision(4) << std::setw(14) << t
                  << std::setprecision(2) << std::setw(10) << tSeq / t << std::setw(12)
                  << tSeq / t / T << cnt << (cnt == expected ? "" : "  <-- MISMATCH") << "\n";
    }

    // ---------- Task 2 + 3: fixed threads, increasing N, sequential vs parallel ----------
    std::cout << "\n=== 2d Task 2/3: fixed " << maxThreads
              << " threads, increasing N (sequential vs parallel) ===\n";
    std::cout << std::left << std::setw(12) << "N" << std::setw(14) << "Seq(s)" << std::setw(14)
              << "Par(s)" << std::setw(10) << "Speedup" << "Primes found\n";
    for (int N : {100000, 250000, 500000, 1000000, 2000000, 4000000}) {
        size_t a = 0, b = 0;
        double ts = timeIt([&] { a = primesSeq(N, out); });
        double tp = timeIt([&] { b = primesPar(N, maxThreads, flag, out); });
        std::cout << std::left << std::setw(12) << N << std::fixed << std::setprecision(4)
                  << std::setw(14) << ts << std::setw(14) << tp << std::setprecision(2)
                  << std::setw(10) << ts / tp << a << (a == b ? "" : "  <-- MISMATCH") << "\n";
    }
    return 0;
}
