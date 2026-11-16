// Practical 2d: OpenMP - find primes between 2 and N and store them in an array
// Compile: g++ -fopenmp 2d_prime_numbers.cpp -o 2d
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

bool isPrime(int number) {
    if (number < 2) return false;
    for (int i = 2; i * i <= number; i++)
        if (number % i == 0) return false;
    return true;
}

int main() {
    int numThreads, N;
    std::cout << "Enter the number of threads: ";
    std::cin >> numThreads;
    std::cout << "Enter the value of N: ";
    std::cin >> N;

    omp_set_num_threads(numThreads);

    std::vector<int> primes;  // shared result array

    #pragma omp parallel for
    for (int value = 2; value <= N; value++) {
        if (isPrime(value)) {
            // Many threads writing to one vector at once would be a race
            // condition, so only one thread at a time may add to it.
            #pragma omp critical
            primes.push_back(value);
        }
    }

    // Threads finish in arbitrary order, so sort to get a clean output.
    std::sort(primes.begin(), primes.end());

    std::cout << "Number of prime numbers between 2 and " << N << ": " << primes.size() << "\n";
    std::cout << "Primes: ";
    for (int p : primes) std::cout << p << " ";
    std::cout << "\n";
    return 0;
}
