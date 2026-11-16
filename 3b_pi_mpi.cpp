// Practical 3b: MPI - calculate Pi
// Each process sums a share of the intervals; MPI_Reduce adds the partial results.
// Compile: mpic++ 3b_pi_mpi.cpp -o 3b
// Run:     mpirun -np 4 ./3b
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>

int main(int argc, char* argv[]) {
    const double PI25DT = 3.141592653589793238462643;
    int n, myid, numprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    while (true) {
        if (myid == 0) {
            std::cout << "Enter the number of intervals: (0 quits) " << std::flush;
            std::cin >> n;
        }

        // Rank 0 sends n to every other process
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (n == 0) break;

        double h = 1.0 / n;
        double sum = 0.0;

        // Process k handles intervals k+1, k+1+numprocs, k+1+2*numprocs, ...
        for (int i = myid + 1; i <= n; i += numprocs) {
            double x = h * (i - 0.5);
            sum += 4.0 / (1.0 + x * x);
        }
        double mypi = h * sum;

        double pi;
        MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (myid == 0) {
            std::cout << std::fixed << std::setprecision(16)
                      << "pi is approximately " << pi
                      << ", Error is " << std::fabs(pi - PI25DT) << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
