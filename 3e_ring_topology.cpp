// Practical 3e: MPI - Ring topology
// Rank 0 sends its rank to rank 1. Each process adds its own rank to the value it
// receives and passes it on. The last process sends the final sum back to rank 0,
// closing the ring. Final sum = 0 + 1 + ... + (p-1).
// Compile: mpic++ 3e_ring_topology.cpp -o 3e
// Run:     mpirun -np 4 ./3e
#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, numprocs;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (numprocs == 1) {
        std::cout << "MyRank 0 Final SUM 0 (only one process, no ring)" << std::endl;
        MPI_Finalize();
        return 0;
    }

    const int tag = 0;
    int next = (rank + 1) % numprocs;             // neighbour to send to
    int prev = (rank - 1 + numprocs) % numprocs;  // neighbour to receive from
    int value, sum;

    if (rank == 0) {
        // Start the ring with our own rank, then wait for it to come back around
        MPI_Send(&rank, 1, MPI_INT, next, tag, MPI_COMM_WORLD);
        MPI_Recv(&sum, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, &status);
        std::cout << "MyRank " << rank << " Final SUM " << sum << std::endl;
    } else {
        MPI_Recv(&value, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, &status);
        sum = value + rank;
        // For the last process, next == 0, which completes the ring
        MPI_Send(&sum, 1, MPI_INT, next, tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
