// Practical 2a: OpenMP Fork-Join model
// Compile: g++ -fopenmp 2a_fork_join.cpp -o 2a
#include <iostream>
#include <omp.h>

int main() {
    // Only the master thread exists here.
    std::cout << "Before parallel region: 1 thread (master)\n";

    // FORK: a team of threads is created. Each thread gets its own copy of tid.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // critical: only one thread prints at a time, so output is not garbled
        #pragma omp critical
        std::cout << "Hello World from thread = " << tid << "\n";

        // Only the master thread (id 0) does this
        if (tid == 0) {
            #pragma omp critical
            std::cout << "Number of threads = " << omp_get_num_threads() << "\n";
        }
    }   // JOIN: all threads merge back into the master thread

    std::cout << "After parallel region: back to 1 thread (master)\n";
    return 0;
}
