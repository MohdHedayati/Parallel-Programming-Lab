// Practical 4: Parallel Hyperquicksort using MPI + OpenMP
//
//   1. Rank 0 generates 10,000,000 random integers.
//   2. The data is divided equally among the MPI processes (4 processes -> 2.5M each).
//   3. Hyperquicksort:
//        a) every process sorts its own chunk (OpenMP task-parallel quicksort)
//        b) for each hypercube dimension, from the highest bit to the lowest:
//             - the root of the current subcube picks a pivot (median of its sorted data)
//               and broadcasts it to the subcube
//             - every process splits its sorted data into "<= pivot" and "> pivot"
//             - partner processes (rank XOR 2^i) swap halves: the lower one keeps the
//               small half and sends the large half, the upper one does the opposite
//             - each process merges what it kept with what it received
//      After log2(p) rounds, process 0 holds the smallest values, process p-1 the
//      largest, and every process's data is sorted.
//   4. Rank 0 gathers the result and checks it against a sequential std::sort.
//
// MPI handles communication between processes; OpenMP uses several threads inside
// each process (for the local sort and for the verification).
//
// Compile: mpic++ -O2 -fopenmp 4_hyperquicksort.cpp -o 4_hyperquicksort
// Run:     mpiexec -n 4 .\4_hyperquicksort.exe [N] [threads_per_process]
//          (number of processes must be a power of 2: 1, 2, 4, 8, ...)
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>

// ---------------------------------------------------------------------------
// OpenMP task-parallel quicksort on a[lo, hi)
// ---------------------------------------------------------------------------
static void quickTask(int* a, long lo, long hi) {
    const long CUTOFF = 50000;  // below this, plain std::sort is faster than making tasks
    if (hi - lo <= CUTOFF) {
        std::sort(a + lo, a + hi);
        return;
    }
    // Three-way partition around the middle element: [< pivot][== pivot][> pivot]
    int pivot = a[lo + (hi - lo) / 2];
    int* m1 = std::partition(a + lo, a + hi, [pivot](int x) { return x < pivot; });
    int* m2 = std::partition(m1, a + hi, [pivot](int x) { return x == pivot; });
    long i1 = m1 - a, i2 = m2 - a;

    #pragma omp task
    quickTask(a, lo, i1);
    #pragma omp task
    quickTask(a, i2, hi);
}

static void parallelSort(std::vector<int>& v) {
    if (v.size() < 2) return;
    #pragma omp parallel
    {
        #pragma omp single nowait
        quickTask(v.data(), 0, (long)v.size());
    }  // implicit barrier: waits for all tasks
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // FUNNELED: only the main thread of each process makes MPI calls
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long N = (argc > 1) ? std::atol(argv[1]) : 10000000L;
    int threads = (argc > 2) ? std::atoi(argv[2])
                             : std::max(1, omp_get_num_procs() / size);
    omp_set_num_threads(threads);

    if ((size & (size - 1)) != 0 || N % size != 0) {
        if (rank == 0)
            std::cout << "Need a power-of-2 number of processes dividing N evenly.\n";
        MPI_Finalize();
        return 1;
    }
    int dims = 0;
    while ((1 << dims) < size) dims++;  // hypercube dimension = log2(size)
    const int localN = (int)(N / size);

    if (rank == 0)
        std::cout << "Hyperquicksort: N = " << N << ", processes = " << size
                  << ", OpenMP threads per process = " << threads
                  << ", hypercube dimension = " << dims << "\n";

    // ---- 1. Generate data on rank 0 ---------------------------------------
    std::vector<int> all;  // only filled on rank 0
    if (rank == 0) {
        all.resize(N);
        std::mt19937 gen(12345);
        std::uniform_int_distribution<int> dist(0, 1000000000);
        for (long i = 0; i < N; i++) all[i] = dist(gen);
        std::cout << "Generated " << N << " random integers.\n";
    }

    // Sequential baseline (rank 0 only) for correctness check and speedup
    std::vector<int> expected;
    double tSeq = 0;
    if (rank == 0) {
        expected = all;
        double t0 = MPI_Wtime();
        std::sort(expected.begin(), expected.end());
        tSeq = MPI_Wtime() - t0;
    }

    // ---- 2. Divide data among the processes -------------------------------
    std::vector<int> local(localN);
    MPI_Barrier(MPI_COMM_WORLD);
    double tScatter0 = MPI_Wtime();
    MPI_Scatter(all.data(), localN, MPI_INT, local.data(), localN, MPI_INT, 0, MPI_COMM_WORLD);
    double tScatter = MPI_Wtime() - tScatter0;

    // ---- 3. Hyperquicksort ------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double tStart = MPI_Wtime();

    // Step (a): sort the local chunk using all OpenMP threads
    parallelSort(local);

    // Step (b): one round per hypercube dimension, highest bit first
    for (int i = dims - 1; i >= 0; i--) {
        // Processes with the same (rank >> (i+1)) form the current subcube
        MPI_Comm sub;
        MPI_Comm_split(MPI_COMM_WORLD, rank >> (i + 1), rank, &sub);
        int subRank;
        MPI_Comm_rank(sub, &subRank);

        // Subcube root picks the pivot = median of its sorted data
        int pivot = 0;
        if (subRank == 0 && !local.empty()) pivot = local[local.size() / 2];
        MPI_Bcast(&pivot, 1, MPI_INT, 0, sub);
        MPI_Comm_free(&sub);

        // Split the sorted data: [0, split) <= pivot, [split, end) > pivot
        long split = std::upper_bound(local.begin(), local.end(), pivot) - local.begin();

        int partner = rank ^ (1 << i);
        bool lower = ((rank >> i) & 1) == 0;

        // Lower process keeps the small part and sends the large part; upper does the reverse
        const int* keepPtr = lower ? local.data() : local.data() + split;
        int keepCnt = (int)(lower ? split : (long)local.size() - split);
        const int* sendPtr = lower ? local.data() + split : local.data();
        int sendCnt = (int)(lower ? (long)local.size() - split : split);

        // Exchange sizes, then the data itself
        int recvCnt;
        MPI_Sendrecv(&sendCnt, 1, MPI_INT, partner, 0, &recvCnt, 1, MPI_INT, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<int> recv(recvCnt);
        MPI_Sendrecv(sendPtr, sendCnt, MPI_INT, partner, 1, recv.data(), recvCnt, MPI_INT, partner, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Merge the two sorted pieces into the new local array
        std::vector<int> merged(keepCnt + recvCnt);
        std::merge(keepPtr, keepPtr + keepCnt, recv.begin(), recv.end(), merged.begin());
        local.swap(merged);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tLocal = MPI_Wtime() - tStart;
    double tPar = 0;
    MPI_Reduce(&tLocal, &tPar, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- 4. Verify --------------------------------------------------------
    // Each process checks its own piece is sorted, using OpenMP threads
    int sortedLocal = 1;
    long n = (long)local.size();
    #pragma omp parallel for reduction(&& : sortedLocal)
    for (long k = 1; k < n; k++)
        sortedLocal = sortedLocal && (local[k - 1] <= local[k]);

    // Collect the piece sizes (hyperquicksort does not guarantee equal sizes) and the data
    int myCount = (int)local.size();
    std::vector<int> counts(size), displs(size, 0);
    MPI_Gather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (int r = 1; r < size && rank == 0; r++) displs[r] = displs[r - 1] + counts[r - 1];

    std::vector<int> result;
    if (rank == 0) result.resize(N);
    MPI_Gatherv(local.data(), myCount, MPI_INT, result.data(), counts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    int allSorted = 0;
    MPI_Reduce(&sortedLocal, &allSorted, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        bool matches = (result == expected);  // identical to sequential sort => correct
        std::cout << "\nElements per process after sorting:";
        for (int r = 0; r < size; r++) std::cout << " P" << r << "=" << counts[r];
        std::cout << "\nFirst 10 sorted values:";
        for (int k = 0; k < 10 && k < N; k++) std::cout << " " << result[k];
        std::cout << "\nLast 10 sorted values: ";
        for (long k = std::max(0L, N - 10); k < N; k++) std::cout << " " << result[k];

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n\nEach piece sorted (checked in parallel): " << (allSorted ? "yes" : "NO")
                  << "\nMatches sequential std::sort:            " << (matches ? "yes" : "NO")
                  << "\n\nScatter time:                 " << tScatter << " s"
                  << "\nParallel hyperquicksort time: " << tPar << " s"
                  << "\nSequential std::sort time:    " << tSeq << " s"
                  << "\nSpeedup (sequential/parallel): " << std::setprecision(2) << tSeq / tPar << "x\n";
    }

    MPI_Finalize();
    return 0;
}
