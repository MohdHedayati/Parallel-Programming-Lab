// Practical 2b: OpenMP Producer-Consumer
// Thread 0 (master) is the producer. All other threads are consumers that wait
// until the producer has filled the buffer, then consume from it.
// Compile: g++ -fopenmp 2b_producer_consumer.cpp -o 2b
#include <iostream>
#include <atomic>
#include <omp.h>

int main() {
    int buffer = 0;                  // shared "data available" counter
    std::atomic<bool> ready(false);  // flag set by producer when data is ready

    omp_set_num_threads(4);

    #pragma omp parallel shared(buffer, ready)
    {
        int tid = omp_get_thread_num();

        if (tid == 0) {
            // ---- Producer ----
            #pragma omp critical
            {
                std::cout << "Producer (thread " << tid << ") is adding data to the buffer\n";
                buffer += 10;
            }
            ready = true;  // notify consumers
        } else {
            // ---- Consumer ----
            while (!ready) {
                // busy-wait until producer signals (no printing, to avoid flooding output)
            }

            #pragma omp critical
            {
                if (buffer > 0) {
                    buffer -= 5;
                    std::cout << "Consumer (thread " << tid << ") consumed data. Buffer left = "
                              << buffer << "\n";
                } else {
                    std::cout << "Consumer (thread " << tid << ") found no data\n";
                }
            }
        }
    }

    std::cout << "Final buffer value = " << buffer << "\n";
    return 0;
}
