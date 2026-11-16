// Practical 3c: MPI_Scatter - root (rank 0) sends "VJTI" to all 4 processes
// Compile: mpic++ 3c_scatter.cpp -o 3c
// Run:     mpirun -np 4 ./3c
#include <iostream>
#include <mpi.h>

const int SIZE = 4;

int main(int argc, char* argv[]) {
    int numtasks, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks == SIZE) {
        // Only the root's send buffer matters. Row i goes to process i.
        char sendbuf[SIZE][SIZE] = {
            {'V', 'J', 'T', 'I'},
            {'V', 'J', 'T', 'I'},
            {'V', 'J', 'T', 'I'},
            {'V', 'J', 'T', 'I'}
        };
        char recvbuf[SIZE];

        MPI_Scatter(sendbuf, SIZE, MPI_CHAR,   // what root sends (SIZE chars per process)
                    recvbuf, SIZE, MPI_CHAR,   // what each process receives
                    0, MPI_COMM_WORLD);        // root = rank 0

        std::cout << "rank= " << rank << "  Results: "
                  << recvbuf[0] << " " << recvbuf[1] << " "
                  << recvbuf[2] << " " << recvbuf[3] << std::endl;
    } else if (rank == 0) {
        std::cout << "Must specify " << SIZE << " processors. Terminating." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
