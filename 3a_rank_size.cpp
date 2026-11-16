// Practical 3a: MPI - rank and number of processes
// Compile: mpic++ 3a_rank_size.cpp -o 3a
// Run:     mpirun -np 4 ./3a      (on Windows MS-MPI: mpiexec -n 4 3a.exe)
#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);                  // start MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    // id of this process
    MPI_Comm_size(MPI_COMM_WORLD, &size);    // total number of processes

    std::cout << "Hello world from process " << rank << " of " << size << std::endl;

    MPI_Finalize();                          // shut down MPI
    return 0;
}
