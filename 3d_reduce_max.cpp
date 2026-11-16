// Practical 3d: MPI_Reduce - maximum of 6 integers using 6 processes
// Each process owns one element; the maximum is collected at the root (rank 0).
// Compile: mpic++ 3d_reduce_max.cpp -o 3d
// Run:     mpirun -np 6 ./3d
#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, numtasks;
    int array[6] = {100, 600, 300, 800, 250, 720};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks != 6) {
        if (rank == 0)
            std::cout << "This program needs exactly 6 processes." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int inputNumber = array[rank];
    std::cout << "Local Input for process " << rank << " is " << inputNumber << std::endl;

    int maxNumber;
    MPI_Reduce(&inputNumber, &maxNumber, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        std::cout << "Maximum of all is: " << maxNumber << std::endl;

    MPI_Finalize();
    return 0;
}
