// Benchmark for 2c (Matrix Multiplication)
//   Task 3: sequential vs parallel execution time
// Compile: g++ -O2 -fopenmp bench_2c_matrix.cpp -o bench_2c
// Run:     ./bench_2c
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <omp.h>

using Mat = std::vector<int>;  // n x n matrix stored in one flat array

// Best (smallest) time of 3 runs, in seconds
template <class F>
double timeIt(F f) {
    double best = 1e30;
    for (int r = 0; r < 3; r++) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

void multiplySeq(const Mat& A, const Mat& B, Mat& C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int s = 0;
            for (int k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

void multiplyPar(const Mat& A, const Mat& B, Mat& C, int n, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int s = 0;
            for (int k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

int main() {
    const int threads = omp_get_max_threads();
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 9);

    std::cout << "=== 2c Matrix Multiplication: sequential vs parallel (" << threads
              << " threads) ===\n";
    std::cout << std::left << std::setw(10) << "Size" << std::setw(16) << "Sequential(s)"
              << std::setw(16) << "Parallel(s)" << std::setw(10) << "Speedup" << "Correct\n";

    for (int n : {100, 200, 400, 600, 800, 1000}) {
        Mat A(n * n), B(n * n), C1(n * n), C2(n * n);
        for (auto& x : A) x = dist(gen);
        for (auto& x : B) x = dist(gen);

        double ts = timeIt([&] { multiplySeq(A, B, C1, n); });
        double tp = timeIt([&] { multiplyPar(A, B, C2, n, threads); });

        std::cout << std::left << std::setw(10) << (std::to_string(n) + "x" + std::to_string(n))
                  << std::fixed << std::setprecision(5) << std::setw(16) << ts << std::setw(16) << tp
                  << std::setprecision(2) << std::setw(10) << ts / tp
                  << (C1 == C2 ? "yes" : "NO") << "\n";
    }
    return 0;
}
