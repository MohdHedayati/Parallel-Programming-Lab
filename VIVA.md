# Parallel Computing Lab – Viva Questions & Answers

Covers every practical in the manual (1, 2a–2f, 3a–3e, 4a–4b, 5a–5f), plus the
benchmarks (speedup / efficiency) and the hyperquicksort practical (MPI + OpenMP).

Contents: [General](#general-theory) · [OpenMP](#openmp-2a2f) · [MPI](#mpi-3a3e) ·
[Performance](#performance-and-benchmarks) · [Numerical](#numerical-computing-4a4b) ·
[Hyperquicksort](#hyperquicksort-mpi--openmp) · [CUDA](#cuda-5a5f)

---

## General theory

**Q1. What is parallel computing?**
Solving a problem by splitting it into parts that run at the same time on multiple
processing units, then combining the results.

**Q2. Why has parallel computing become dominant?**
Clock frequency stopped scaling because of power and heat limits (the "power wall").
Chip makers now add more cores instead of making one core faster, so software must be
parallel to use them.

**Q3. Name the forms of parallelism.**
Bit-level, instruction-level, data parallelism (same operation on different data) and
task parallelism (different operations at the same time).

**Q4. Data parallelism vs task parallelism?**
Data: the same operation is applied to different pieces of data (matrix multiply, vector add).
Task: different threads do different jobs (producer and consumer).

**Q5. Shared memory vs distributed memory?**
Shared: all threads see the same memory (OpenMP, threads inside one machine). Communication
is by reading and writing shared variables, and synchronisation is needed to avoid races.
Distributed: each process has its own private memory and communicates by sending
messages (MPI). It scales across machines.

**Q6. Define Flynn's taxonomy.**
SISD (normal sequential CPU), SIMD (vector units, GPU warps), MISD (rare), MIMD (multicore
CPUs, clusters).

**Q7. What are the objectives and outcomes of this lab?**
Objectives: learn to design parallel programs and evaluate them, understand the benefits
and limits of parallel systems, and write code in OpenMP, MPI and CUDA.
Outcomes: reason about how to parallelise a problem, evaluate a platform for it, and be
familiar with OpenMP, MPI and CUDA.

**Q8. What is a race condition?**
When several threads access the same variable and at least one writes, and the result
depends on the order of execution. Fix it with `critical`, `atomic`, `reduction`, or by
making the data private.

**Q9. What is a deadlock?**
Two or more processes each wait for something the other holds, so none can proceed.
Example: two MPI processes both do a blocking `MPI_Send` to each other before receiving.

**Q10. What is Amdahl's law?**
Speedup = 1 / (s + (1 − s)/p), where s is the serial fraction and p the processors.
The serial part limits the maximum speedup to 1/s, however many processors you add.

**Q11. What is Gustafson's law?**
Scaled speedup = p − s(p − 1). If the problem size grows with the number of processors,
speedup can keep growing. It is the more optimistic view.

**Q12. What is load balancing and why does it matter?**
Giving every processor about the same amount of work. The program finishes only when the
slowest processor finishes, so imbalance wastes the others.

**Q13. What is granularity?**
The amount of work done between communications or synchronisations. Fine-grained means
lots of communication (high overhead). Coarse-grained means more work per communication.

---

## OpenMP (2a–2f)

### General OpenMP

**Q14. What is OpenMP?**
An API of compiler directives (`#pragma omp ...`), library functions and environment
variables for shared-memory parallel programming in C, C++ and Fortran.

**Q15. Explain the fork-join model.**
The program starts with one master thread. At a `#pragma omp parallel` it forks a team of
threads, which run the block concurrently. At the end of the block they join: the threads
finish and only the master continues.

**Q16. How do you compile an OpenMP program?**
`g++ -fopenmp file.cpp -o out`. Without `-fopenmp` the pragmas are ignored and the program
runs sequentially.

**Q17. Which header and functions are important?**
`<omp.h>`: `omp_get_thread_num()` (my id; master = 0), `omp_get_num_threads()` (team size),
`omp_set_num_threads(n)`, `omp_get_max_threads()`, `omp_get_wtime()` (timer).

**Q18. How can you set the number of threads?**
Three ways: `omp_set_num_threads(n)`, the clause `num_threads(n)` on the pragma, or the
environment variable `OMP_NUM_THREADS`.

**Q19. Difference between `parallel` and `parallel for`?**
`parallel` creates a team, and every thread runs the whole block. `parallel for` also
splits the loop iterations among the threads (a work-sharing construct).

**Q20. `shared` vs `private` vs `firstprivate`?**
`shared`: one copy seen by all threads. `private`: each thread gets its own uninitialised
copy. `firstprivate`: private copy initialised with the value from before the region.
Variables declared inside the parallel region are automatically private.

**Q21. What is `reduction`?**
Each thread keeps a private partial result (initialised to the identity: 0 for +, −∞ for max),
and OpenMP combines the partial results at the end with the operator. Example:
`reduction(+:sum)`. It avoids races without the slowness of `critical`.

**Q22. `critical` vs `atomic`?**
`critical` protects any block, and only one thread at a time may enter it. `atomic` protects
a single simple memory update (like `x += 1`), using hardware support, so it is cheaper.

**Q23. What are the `schedule` options?**
`static`: iterations split into equal chunks up front (lowest overhead, best when each
iteration costs the same). `dynamic`: threads grab chunks as they finish (good when work is
uneven). `guided`: chunks shrink over time. `auto`/`runtime`: left to the compiler or the
environment.

**Q24. What is an implicit barrier?**
At the end of a `parallel` region (and of `for` / `single` unless `nowait`) all threads wait
for each other. `#pragma omp barrier` adds one explicitly.

**Q25. What does `nowait` do?**
Removes the implicit barrier at the end of a work-sharing construct, so threads move on
without waiting.

**Q26. What are OpenMP tasks?**
`#pragma omp task` packages a piece of work that any thread in the team can execute later.
They suit irregular or recursive work (like quicksort). `taskwait` waits for the child tasks.

**Q27. Why might output from threads look garbled?**
All threads share `stdout`, and printing is not atomic, so characters from different threads
interleave. Wrap printing in `critical`, or print after the parallel region.

### 2a – Fork-join

**Q28. What does your 2a program show?**
That a team of threads is forked at `#pragma omp parallel`. Each thread prints its id from
`omp_get_thread_num()`. Thread 0 (the master) also prints the team size, and the threads join
afterwards. The order of the hello lines changes between runs.

**Q29. Why is the output order different every time?**
Threads are scheduled by the OS and run concurrently, so there is no guaranteed order.

**Q30. Why did you print the thread count only from thread 0?**
It is the same for all threads, so printing it once avoids duplicates.

### 2b – Producer-consumer

**Q31. Explain the producer-consumer problem.**
A producer puts items into a shared buffer and consumers remove them. Consumers must not
consume before data exists, and updates to the buffer must not conflict.

**Q32. How did you implement it?**
Thread 0 is the producer: it adds data (`buffer += 10`) inside a `critical` section and then
sets an atomic flag `ready`. The other threads spin until `ready` is true, then inside a
`critical` section each consumes 5 units if any remain, otherwise reports no data.

**Q33. Why is the flag `std::atomic`?**
A plain variable read in a busy-wait loop may be cached in a register by the compiler, so
the consumer might never see the producer's update. An atomic guarantees visibility.

**Q34. What was wrong in the manual's version?**
`#pragma critical` (should be `#pragma omp critical`, otherwise it is ignored) and it printed
inside the spin loop, flooding the output, plus the non-atomic flag.

**Q35. What is the drawback of busy-waiting?**
Waiting threads burn CPU. Better options are condition variables, locks, or OpenMP tasks and
dependencies.

### 2c – Matrix multiplication

**Q36. How did you parallelise matrix multiplication?**
`#pragma omp parallel for` on the outer `i` loop, so each thread computes complete rows of C.
Rows are independent, so no synchronisation is needed.

**Q37. What is the complexity?**
O(n³) sequential for n×n matrices; ideally O(n³/p) with p threads.

**Q38. Why are `i`, `j`, `k` private?**
Each thread needs its own loop counters. If they were shared, threads would overwrite each
other's values. Declaring them inside the loop makes them private automatically.

**Q39. Which loop should be parallel, and why the outer?**
The outer loop gives the biggest chunks of work per thread (coarse granularity), so thread
management overhead is small. Parallelising inner loops would create and sync teams
repeatedly.

**Q40. How can matrix multiplication be made faster?**
Loop tiling or blocking for cache, swapping loop order (`i-k-j`) to read B row-wise,
`collapse(2)`, or an optimised library such as BLAS.

### 2d – Prime numbers

**Q41. What does the program do?**
Finds all primes between 2 and N in parallel and stores them in an array.

**Q42. What was the problem in the manual's version?**
`arrayVal[indexCount++]` from many threads at once is a race. Also `reduction(+:indexCount)`
gives each thread its own private counter, so the indexes collide and the array is garbled.
Its `IsPrime` also loops up to n instead of √n.

**Q43. How did you fix it?**
Primes are added to the shared vector inside a `critical` section, and the result is sorted
at the end. The faster benchmark version has each thread write its own cell of a flag array
(no race), then collects the primes in order.

**Q44. Why `schedule(dynamic)` in the benchmark?**
Testing a large number takes longer than a small one, so iterations have uneven cost.
Dynamic scheduling balances it. Static would give the last thread the most work.

**Q45. Why check only up to √n?**
If n has a divisor larger than √n, it also has a matching divisor smaller than √n.

### 2e – Largest element

**Q46. How is the maximum found in parallel?**
Each thread finds the max of its share, and the thread maxima are combined, which is
`reduction(max:currentMax)`.

**Q47. What was wrong with the manual's version?**
Putting `critical` around every comparison serialises the loop (only one thread works at a
time) so it is no faster than sequential. It also started from −1, which fails for all-negative
data (we use `INT_MIN`).

**Q48. Why is the speedup for finding the max small?**
Almost no computation per element, so the loop is limited by memory bandwidth, not CPU. Extra
threads cannot read memory faster than the bus allows. This is a memory-bound problem.

### 2f – Pi calculation

**Q49. How is Pi computed?**
Pi = ∫₀¹ 4/(1+x²) dx. The interval is split into N rectangles of width `step = 1/N`, the
height is evaluated at the midpoint `x = (i + 0.5)·step`, and Pi ≈ step × Σ heights.

**Q50. How is it parallelised?**
The loop over rectangles is split across threads with `parallel for`, and `reduction(+:sum)`
adds the partial sums.

**Q51. Why does a bigger number of steps give a better answer?**
It is a numerical approximation. Smaller rectangles reduce the error (the midpoint rule has
error ≈ 1/N²), at the cost of more computation.

**Q52. Why is Pi a good example for parallel computing?**
It is "embarrassingly parallel": iterations are independent, with only one combine step at
the end.

---

## MPI (3a–3e)

### General MPI

**Q53. What is MPI?**
Message Passing Interface: a standard library for distributed-memory parallel programming.
Independent processes with private memory cooperate by sending and receiving messages.

**Q54. What is a rank? What is a communicator?**
Rank: the id (0 to size−1) of a process in a communicator. Communicator: a group of
processes that can communicate. `MPI_COMM_WORLD` contains all processes.

**Q55. What do `MPI_Init`, `MPI_Comm_rank`, `MPI_Comm_size`, `MPI_Finalize` do?**
`MPI_Init`: starts MPI. `MPI_Comm_rank`: my id. `MPI_Comm_size`: total processes.
`MPI_Finalize`: shuts MPI down (must be the last MPI call).

**Q56. How do you compile and run an MPI program?**
`mpic++ prog.cpp -o prog` then `mpiexec -n 4 prog` (or `mpirun -np 4`).

**Q57. What is SPMD?**
Single Program, Multiple Data: every process runs the same program but behaves differently
based on its rank (`if (rank == 0) {...} else {...}`).

**Q58. Blocking vs non-blocking communication?**
Blocking (`MPI_Send`, `MPI_Recv`) returns when the buffer is safe to reuse (or the data
arrived). Non-blocking (`MPI_Isend`, `MPI_Irecv`) returns immediately, so computation can
overlap communication, and you later call `MPI_Wait`.

**Q59. Point-to-point vs collective communication?**
Point-to-point: between two processes (`Send` and `Recv`). Collective: involves every
process in a communicator (`Bcast`, `Scatter`, `Gather`, `Reduce`, `Allreduce`, `Barrier`).
All processes must call the collective.

**Q60. What are the arguments of `MPI_Send`?**
`(buffer, count, datatype, destination rank, tag, communicator)`. `MPI_Recv` adds a `status`
and uses the source rank instead.

**Q61. What is a tag?**
An integer label to distinguish different messages between the same pair of processes.

**Q62. Explain `MPI_Bcast`, `MPI_Scatter`, `MPI_Gather`, `MPI_Reduce`.**
Bcast: the root sends the same data to all. Scatter: the root splits an array and gives each
process one piece. Gather: the opposite of scatter, collecting pieces at the root. Reduce:
combines values from all processes using an operation (SUM, MAX, MIN) and gives the result to
the root.

**Q63. `MPI_Reduce` vs `MPI_Allreduce`?**
Reduce delivers the result to one root. Allreduce delivers it to every process.

**Q64. Which is faster: many `Send`s from root or one collective?**
Collectives are usually faster, since MPI implements them with tree algorithms
(O(log p) steps instead of p sequential sends).

### 3a – Rank and size

**Q65. What does 3a print?**
"Hello world from process r of p" from every process. Run with `-n 4` it prints 4 lines in
random order.

**Q66. Why is the order random?**
Processes run independently, with no ordering between their `printf`s.

### 3b – Pi (MPI)

**Q67. Explain the algorithm.**
Rank 0 reads the number of intervals n and broadcasts it. Process `r` handles intervals
`r+1, r+1+p, r+1+2p, …` (cyclic distribution), computing a partial sum. `MPI_Reduce` with
`MPI_SUM` adds the partial results into pi on rank 0.

**Q68. Why a cyclic distribution?**
It spreads work evenly even when n is not divisible by p.

**Q69. Why `MPI_Bcast` of `n`?**
Only rank 0 reads the keyboard, so the others need n from rank 0.

**Q70. OpenMP Pi (2f) vs MPI Pi (3b)?**
2f uses threads sharing memory and a `reduction` clause. 3b uses separate processes with
messages (`Bcast` and `Reduce`), so it scales across machines.

### 3c – Scatter

**Q71. What does 3c do?**
Rank 0 scatters a 4×4 char array so each of the 4 processes gets one row ("VJTI").

**Q72. What do `sendcount` and `recvcount` mean in `MPI_Scatter`?**
`sendcount` is the number of elements sent to **each** process (not the total), and
`recvcount` the number each process receives.

**Q73. Why does the program require exactly 4 processes?**
The array has 4 rows, one per process. With another count the data would not divide evenly.

**Q74. Do non-root processes need a valid send buffer?**
No. The send arguments are only used by the root.

### 3d – Reduce max

**Q75. What does 3d do?**
6 processes, each holding one element `array[rank]`. `MPI_Reduce` with `MPI_MAX` finds the
maximum (800) at rank 0.

**Q76. What happens if you run with fewer or more than 6 processes?**
`array[rank]` would read out of bounds with more than 6. Our version checks the count and
exits with a message.

**Q77. Other predefined reduction operations?**
`MPI_SUM`, `MPI_PROD`, `MPI_MIN`, `MPI_MAX`, `MPI_LAND`, `MPI_LOR`, `MPI_MAXLOC`, `MPI_MINLOC`.

### 3e – Ring topology

**Q78. Explain the ring program.**
Rank 0 sends its rank to rank 1. Each process receives a value, adds its own rank and sends
it to the next. The last process sends to rank 0, closing the ring. The result is
0+1+…+(p−1), which is 6 for 4 processes.

**Q79. Why must rank 0 send first, before receiving?**
If every process called a blocking `Recv` first, they would all wait forever: a deadlock.
Rank 0 starts the chain.

**Q80. How do you compute neighbours in a ring?**
`next = (rank + 1) % p`, `prev = (rank − 1 + p) % p`.

---

## Performance and benchmarks

**Q81. Define speedup.**
S = T_sequential / T_parallel. Ideal speedup with p processors is p.

**Q82. Define efficiency.**
E = S / p. 1.0 (100%) is perfect. It falls as overheads grow.

**Q83. What is strong scaling? What is weak scaling?**
Strong: fixed problem size, increase processors (our "same data, more threads" test). Weak:
problem size grows with processors, keeping work per processor constant.

**Q84. What did your "same data, more threads" experiment show?**
Time fell as threads increased but efficiency also fell. For example, Pi with 400M steps:
1.9× with 2 threads, 3.5× with 4, 5.45× with 8 (efficiency 68%). The serial parts, thread
overhead and shared hardware limit it.

**Q85. What did your "same threads, more data" experiment show?**
For small inputs, parallel can be slower than sequential (Pi at 10K steps: 0.17×), because
thread startup costs more than the work. As data grows the speedup rises and then levels off
(≈5×).

**Q86. Why can parallel be slower than sequential?**
Overheads: creating threads, synchronisation, communication, and false sharing. If the work
is tiny, the overhead dominates.

**Q87. Why is the speedup lower than the number of threads?**
Amdahl's serial fraction, overheads, memory bandwidth limits, load imbalance, and the machine
having fewer physical cores than hardware threads (hyper-threading shares execution units).

**Q88. How did you measure time?**
`std::chrono::steady_clock` (high-resolution), best of several runs. (`omp_get_wtime()` on this
build only had ~1 ms resolution, which was too coarse for small runs.) `MPI_Wtime()` is the
MPI equivalent.

**Q89. Why use the best of several runs?**
It removes noise from the OS and cold caches, so we compare the algorithm rather than
background activity.

**Q90. Why compile benchmarks with `-O2`?**
Timing unoptimised code is misleading. `-O2` enables compiler optimisations so we measure
realistic performance.

**Q91. What is false sharing?**
Different threads write to different variables that sit in the same cache line, forcing the
line to bounce between cores and slowing everything. Fix by padding or using private
variables.

**Q92. What is a memory-bound vs compute-bound program?**
Compute-bound: limited by arithmetic (Pi, primes), so it scales well. Memory-bound: limited by
memory bandwidth (finding the max), so it scales poorly.

---

## Numerical computing (4a–4b)

### 4a – Trapezoidal rule

**Q93. What is the trapezoidal rule?**
Approximates ∫ₐᵇ f(x)dx by dividing [a,b] into n trapezoids of width h=(b−a)/n:
Area ≈ h·[f(a)/2 + f(x₁)+…+f(x_{n−1}) + f(b)/2].

**Q94. How is it parallelised with MPI?**
Each process integrates its own sub-interval: `local_n = n/p`, `local_a = a + rank·local_n·h`.
Partial areas are sent to rank 0 (or combined with `MPI_Reduce`), which adds them.

**Q95. In the manual, how does process 0 get a, b and n?**
`Get_data`: rank 0 reads them and sends to each process with `MPI_Send`. (`MPI_Bcast` would
be simpler.)

**Q96. What is a limitation of `local_n = n/p`?**
If n isn't divisible by p, the leftover trapezoids are dropped, giving a wrong answer. It
needs `n % p == 0` or distributing the remainder.

**Q97. What is the exact answer for f(x)=x²+1 from 0 to 1, and why does n matter?**
4/3 ≈ 1.3333. More trapezoids means a smaller error (error ∝ h²).

### 4b – Gaussian filter

**Q98. What is a Gaussian filter?**
A filter whose impulse response is a Gaussian function. In 2D:
G(x,y) = (1/2πσ²)·e^{−(x²+y²)/(2σ²)}, where σ is the standard deviation.

**Q99. Where is it used?**
Blurring and noise reduction in image processing, and signal processing (oscilloscopes,
telecom), because it has no overshoot to a step input and a minimal rise/fall time.

**Q100. How would you parallelise it?**
Image filtering is data parallel: split the image into row blocks, one per process (with
`MPI_Scatter`). Each process convolves its block. Neighbouring rows ("halo" or "ghost" rows)
must be exchanged with `MPI_Sendrecv`, then `MPI_Gather` the result.

**Q101. What is a halo / ghost row?**
Border rows belonging to a neighbour, which a process needs to compute its edge pixels
correctly.

**Q102. What does σ control?**
Blur strength. A larger σ gives a wider kernel and more smoothing.

---

## Hyperquicksort (MPI + OpenMP)

**Q103. What is hyperquicksort?**
A parallel version of quicksort for hypercube-structured processes. Processes first sort their
local data, then in log₂p rounds they exchange data around a pivot so that lower-ranked
processes end up with smaller values.

**Q104. Why does the number of processes need to be a power of 2?**
The processes form a hypercube of dimension d = log₂p, and the partner in round i is
`rank XOR 2^i`, which exists only when p = 2^d.

**Q105. Explain one round.**
The subcube root broadcasts a pivot. Each process splits its sorted list into "≤ pivot" and
"> pivot". The lower partner keeps the small part and sends the big part; the upper partner
does the opposite. Each then merges what it kept with what it received (the two lists are both
sorted).

**Q106. Why is the data kept sorted and merged?**
Local sorting is done once at the start. After that, splitting is a binary search and combining
is a linear merge, which is much cheaper than re-sorting.

**Q107. How is the pivot chosen, and does it matter?**
The median of the subcube root's sorted data. A good pivot splits data evenly. A poor pivot
leaves some processes with more data (load imbalance). In our run the final sizes differed by
less than 0.1%.

**Q108. Why are the final piece sizes unequal?**
Pivots come from one process's median, not the global median, so splits are only
approximately even.

**Q109. Where did MPI and where did OpenMP come in?**
MPI: scatter the data, broadcast pivots, exchange halves between partners, gather the result.
OpenMP: threads inside each process do the local sort (task-parallel quicksort) and the
parallel sortedness check. This is the "hybrid" model.

**Q110. What does `MPI_THREAD_FUNNELED` mean?**
Threads exist, but only the main thread makes MPI calls. It is enough here, since MPI calls
happen outside the OpenMP parallel regions.

**Q111. How does the OpenMP quicksort work?**
It partitions around a pivot into three groups (`<`, `==`, `>`), then spawns an OpenMP task
for each of the outer groups. Below a cutoff size it calls `std::sort`, so tiny tasks don't
cost more than they save.

**Q112. Why is a 3-way partition used?**
It handles many duplicate values (all the equal items stay in place), and avoids quadratic
behaviour.

**Q113. How did you verify correctness?**
Rank 0 gathers all pieces with `MPI_Gatherv` and compares the result to a sequential
`std::sort`. Each process also checks in parallel that its piece is sorted.

**Q114. Why `MPI_Gatherv` rather than `MPI_Gather`?**
The pieces have different sizes, and `Gatherv` accepts per-process counts and displacements.

**Q115. Time complexity?**
Local sort O((n/p) log(n/p)), then log p rounds each with O(n/p) merging and communication,
so about O((n/p)(log n + log p)) overall.

**Q116. Quicksort vs hyperquicksort vs sample sort?**
Hyperquicksort uses log p pairwise exchange rounds with pivots from the median. Sample sort
picks p−1 splitters from samples and redistributes in one all-to-all step, which usually
balances load better.

**Q117. What speedup did you get?**
About 3.8× with 4 processes × 2 threads (0.18 s vs 0.68 s for sequential `std::sort` on 10M
integers). The time was similar for 1, 2, 4 and 8 processes, so the gain comes mainly from
the threads and the fast local sort, not from adding processes.

**Q118. Why doesn't it speed up much with more processes on one PC?**
All processes share the same CPU cores, memory bandwidth and cache, so extra processes mostly
compete for them. Communication through MPI adds overhead, and the problem is also fairly
memory-bound.

---

## CUDA (5a–5f)

### General CUDA

**Q119. What is CUDA?**
Compute Unified Device Architecture: NVIDIA's platform and programming model for running
general-purpose code on its GPUs (GPGPU).

**Q120. CPU vs GPU?**
CPU: a few powerful cores, optimised for low latency on single threads. GPU: thousands of
simple cores, optimised for throughput on many threads running at once.

**Q121. Explain host and device.**
Host: the CPU and its memory. Device: the GPU and its memory. They have separate memories, so
data must be copied explicitly.

**Q122. What is a kernel?**
A function marked `__global__` that runs on the GPU, called from the host with
`kernel<<<blocks, threads>>>(args)`.

**Q123. `__global__`, `__device__`, `__host__`?**
`__global__`: runs on the device, called from the host. `__device__`: runs on and is called
from the device. `__host__`: runs on the host (the default).

**Q124. Describe the thread hierarchy.**
Threads are grouped into blocks, and blocks into a grid. Built-ins: `threadIdx`, `blockIdx`,
`blockDim`, `gridDim` (each with x, y, z).

**Q125. How do you compute a global thread index?**
1D: `i = blockIdx.x * blockDim.x + threadIdx.x`. 2D adds the same for y.

**Q126. What are the usual steps of a CUDA program?**
1. `cudaMalloc` on the device. 2. `cudaMemcpy` host→device. 3. Launch the kernel. 4.
`cudaMemcpy` device→host. 5. `cudaFree`.

**Q127. What is a warp?**
A group of 32 threads that execute the same instruction together (SIMT). Branch divergence
inside a warp makes it run both paths one after the other.

**Q128. What is `__syncthreads()`?**
A barrier for all threads in the same block. It does not synchronise across blocks.

**Q129. What is shared memory?**
Fast on-chip memory shared by the threads in a block. It's used to cut slow global memory
accesses.

**Q130. Why is there an `if (i < N)` check in kernels?**
The grid usually has more threads than elements (rounded up to whole blocks). Extra threads
must not write past the end of the array.

**Q131. How do you time GPU code correctly?**
Kernel launches are asynchronous, so use `cudaEvent` timers (as in 5e) or call
`cudaDeviceSynchronize()` before stopping a CPU timer. Timing without synchronising only
measures the launch.

**Q132. Maximum threads per block?**
1024 on modern GPUs (older cards: 512).

### 5a – Hello World

**Q133. What does the manual's Hello World show?**
The structure of a CUDA program: an empty `__global__` kernel launched with `<<<...>>>`, then
a normal `printf` on the host. (Note the manual's `<<input parameters>>` is a placeholder,
not valid syntax. It should be like `kernel<<<1,1>>>()`.)

**Q134. Can a kernel call `printf`?**
Yes, on devices of compute capability 2.0 and up, with output flushed at synchronisation.

### 5b – Matrix addition

**Q135. How is the work mapped?**
A 2D grid of 2D blocks: thread (i, j) adds one element `c[i+j*N] = a[i+j*N] + b[i+j*N]`. With
N=4 and 2×2 blocks, the grid is 2×2.

**Q136. How is the 2D index flattened?**
`index = i + j*N`, which is the matrix stored as a 1D array (row-major).

**Q137. How is the result checked?**
The same addition is done on the CPU and compared with `assert`.

### 5c – Prefix sum

**Q138. What is a prefix sum (scan)?**
Output[i] = x[0]+x[1]+…+x[i]. Example: [1,2,3,4,5] → [1,3,6,10,15].

**Q139. What is the manual's approach, and its weakness?**
Each thread computes its own output by looping from 0 to its index: O(n²) total work, with
one block only. Efficient GPU scans (Hillis-Steele or Blelloch) use O(n log n) or O(n) work
with shared memory.

**Q140. Inclusive vs exclusive scan?**
Inclusive includes the current element in its sum. Exclusive sums only the elements before it.

### 5d – Matrix transpose

**Q141. How does the naive transpose work?**
Thread (x, y) copies `in[x + N*y]` to `out[y + N*x]`.

**Q142. Why is the naive version slow for large matrices?**
Reads are coalesced but writes are strided (uncoalesced). Improve it by loading a tile into
shared memory and writing it out transposed, avoiding bank conflicts with padding.

**Q143. What is memory coalescing?**
When threads of a warp access consecutive addresses, the hardware combines them into a few
memory transactions. Scattered access needs many.

### 5e – Vector addition

**Q144. How is the launch configured?**
`<<<B, T>>>` blocks and threads per block, with `T × B ≥ N`. Each thread adds one element
`c[tid] = a[tid] + b[tid]`.

**Q145. How do you choose blocks for arbitrary N?**
`blocks = (N + T − 1) / T` (ceiling division).

**Q146. Why are `cudaEvent`s used for timing?**
They are recorded on the GPU's timeline, so they time the work including the asynchronous
kernel accurately.

### 5f – Vector multiplication

**Q147. What does 5f compute?**
Element-wise product `z[i] = x[i] * y[i]` (not a dot product), with `block_size = 4` and
`num_blocks = (n + 3)/4` for n = 20.

**Q148. How would you compute the dot product?**
Multiply element-wise, then do a parallel reduction (using shared memory within blocks and
atomics or a second kernel across blocks).

**Q149. Why can the GPU be slower than the CPU for small arrays?**
The cost of copying data over PCIe and launching the kernel dominates when each element needs
very little work.

**Q150. OpenMP vs MPI vs CUDA: when to use which?**
OpenMP: multicore shared memory on one machine, easy incremental parallelism. MPI: clusters
and multi-node, distributed memory. CUDA: massively data-parallel work on NVIDIA GPUs. They
can be combined (hybrid MPI+OpenMP, or MPI+CUDA).

---

### Quick one-liners to remember

- Fork-join: one master → team of threads → back to one.
- `reduction` beats `critical` for sums and maxima.
- Races happen when threads write to shared data without protection.
- MPI = messages between separate processes. OpenMP = threads sharing memory.
- Speedup = T_seq / T_par. Efficiency = speedup / p.
- Small problems may not be worth parallelising: overhead.
- Hyperquicksort = local sort + log₂p rounds of (pivot, swap halves, merge).
