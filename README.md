# Parallel Computing Lab – OpenMP & MPI (C++)

C++ versions of Practicals 2a–2f (OpenMP) and 3a–3e (MPI) from the VJTI Parallel
Computing Lab Manual, plus benchmark programs for 2c–2f.

## Files

| File | Practical | Needs input? |
|---|---|---|
| `2a_fork_join.cpp` | OpenMP fork-join model | no |
| `2b_producer_consumer.cpp` | OpenMP producer-consumer | no |
| `2c_matrix_multiplication.cpp` | OpenMP matrix multiplication | yes (sizes + matrices) |
| `2d_prime_numbers.cpp` | OpenMP primes between 2 and N | yes (threads, N) |
| `2e_largest_element.cpp` | OpenMP largest array element | yes (count + elements) |
| `2f_pi_calculation.cpp` | OpenMP Pi | no |
| `3a_rank_size.cpp` | MPI rank and process count | no |
| `3b_pi_mpi.cpp` | MPI Pi (`Bcast` + `Reduce`) | yes (intervals, `0` quits) |
| `3c_scatter.cpp` | MPI `Scatter` of "VJTI" | no |
| `3d_reduce_max.cpp` | MPI `Reduce` max of 6 numbers | no |
| `3e_ring_topology.cpp` | MPI ring topology | no |
| `4_hyperquicksort.cpp` | Parallel hyperquicksort, MPI + OpenMP (10M ints, 4 processes) | no (optional args) |
| `bench_2c_matrix.cpp` | Sequential vs parallel (2c) | no |
| `bench_2d_primes.cpp` | More threads / more data / seq vs par (2d) | no |
| `bench_2e_max.cpp` | More threads / more data / seq vs par (2e) | no |
| `bench_2f_pi.cpp` | More threads / more data / seq vs par (2f) | no |

## Requirements

- **g++ with OpenMP** (MSYS2 UCRT64: `C:\msys64\ucrt64\bin`)
- **MS-MPI** runtime and SDK (`C:\Program Files\Microsoft MPI\Bin`), with `mpic++` available

## Setup (once per terminal)

Open a terminal in this folder (VS Code: Terminal → New Terminal), then in PowerShell:

```powershell
$env:Path = "C:\msys64\ucrt64\bin;C:\Program Files\Microsoft MPI\Bin;" + $env:Path
```

Check it worked: `g++ --version` and `mpiexec` should both be found.

## OpenMP programs (2a–2f)

Compile with `-fopenmp`, then run:

```powershell
g++ -fopenmp 2a_fork_join.cpp             -o 2a ; .\2a.exe
g++ -fopenmp 2b_producer_consumer.cpp     -o 2b ; .\2b.exe
g++ -fopenmp 2c_matrix_multiplication.cpp -o 2c ; .\2c.exe
g++ -fopenmp 2d_prime_numbers.cpp         -o 2d ; .\2d.exe
g++ -fopenmp 2e_largest_element.cpp       -o 2e ; .\2e.exe
g++ -fopenmp 2f_pi_calculation.cpp        -o 2f ; .\2f.exe
```

Sample inputs:

- **2c:** rows of A = `2`, cols of A = `2`, cols of B = `2`; then matrix A `1 2 3 4`, matrix B `5 6 7 8`
  → result `19 22 / 43 50`
- **2d:** threads = `4`, N = `30` → 10 primes
- **2e:** count = `5`, elements `3 -7 22 9 1` → max `22`

Set the thread count without editing code (works for 2a, 2c, 2e, 2f; 2b is fixed at 4
threads in code, and 2d asks you for the thread count):

```powershell
$env:OMP_NUM_THREADS = 4
.\2a.exe
```

## MPI programs (3a–3e)

Compile with `mpic++`, run with `mpiexec -n <processes>`:

```powershell
mpic++ 3a_rank_size.cpp     -o 3a ; mpiexec -n 4 .\3a.exe
mpic++ 3b_pi_mpi.cpp        -o 3b ; mpiexec -n 4 .\3b.exe
mpic++ 3c_scatter.cpp       -o 3c ; mpiexec -n 4 .\3c.exe
mpic++ 3d_reduce_max.cpp    -o 3d ; mpiexec -n 6 .\3d.exe
mpic++ 3e_ring_topology.cpp -o 3e ; mpiexec -n 4 .\3e.exe
```

| Program | Processes | Notes |
|---|---|---|
| 3a | any | prints rank and size from each process |
| 3b | any | type e.g. `1000` for the interval count, then `0` to quit |
| 3c | **exactly 4** | otherwise it prints an error and stops |
| 3d | **exactly 6** | one array element per process; max is `800` |
| 3e | 2 or more | final sum = 0+1+…+(p−1), e.g. `6` for 4 processes |

Output order from different processes is not deterministic; lines may appear shuffled.

## Practical 4: Hyperquicksort (MPI + OpenMP)

Sorts 10,000,000 random integers: rank 0 generates the data and scatters it over the
processes (2.5M each with 4), each process sorts its chunk with an OpenMP task-parallel
quicksort, then the hypercube rounds (pivot broadcast, swap halves with partner
`rank ^ 2^i`, merge) leave the data globally sorted across ranks. Rank 0 gathers the
result, checks it against a sequential `std::sort`, and prints the speedup.

```powershell
mpic++ -O2 -fopenmp 4_hyperquicksort.cpp -o 4_hyperquicksort
mpiexec -n 4 ._hyperquicksort.exe
```

Optional arguments: `._hyperquicksort.exe [N] [threads_per_process]`, e.g.
`mpiexec -n 4 ._hyperquicksort.exe 10000000 2`. By default the threads are
(hardware threads / processes). The process count must be a power of 2 (1, 2, 4, 8, ...).
Per-process piece sizes differ slightly after sorting; that is normal for hyperquicksort.

## Benchmarks (bench_2c … bench_2f)

No input needed. Use `-O2` so timings are meaningful:

```powershell
g++ -O2 -fopenmp bench_2c_matrix.cpp -o bench_2c ; .\bench_2c.exe
g++ -O2 -fopenmp bench_2d_primes.cpp -o bench_2d ; .\bench_2d.exe
g++ -O2 -fopenmp bench_2e_max.cpp    -o bench_2e ; .\bench_2e.exe
g++ -O2 -fopenmp bench_2f_pi.cpp     -o bench_2f ; .\bench_2f.exe
```

What each one measures (times are best of 3 runs, 5 for 2e):

1. **Same data, more threads** (2d–2f): speedup and efficiency for 1, 2, 4, … up to
   the maximum thread count.
2. **Same threads, more data** (2d–2f): thread count fixed at the maximum, data size grows.
3. **Sequential vs parallel** (2c–2f): both times side by side, with speedup.

Speedup = sequential time / parallel time. Efficiency = speedup / threads.

Notes:
- `bench_2e` allocates about 400 MB of RAM. The full set takes a minute or two.
- For small inputs the parallel version can be slower, because thread start-up costs more than the work saved.
- 2e (max) is limited by memory bandwidth, so its timings are noisy and speedups are modest.
- Results depend on your machine.

## Run everything in one go

```powershell
$env:Path = "C:\msys64\ucrt64\bin;C:\Program Files\Microsoft MPI\Bin;" + $env:Path
New-Item -ItemType Directory -Force build | Out-Null

foreach ($f in '2a_fork_join','2b_producer_consumer','2c_matrix_multiplication',
               '2d_prime_numbers','2e_largest_element','2f_pi_calculation') {
    g++ -fopenmp "$f.cpp" -o "build\$f.exe"
}
foreach ($f in '3a_rank_size','3b_pi_mpi','3c_scatter','3d_reduce_max','3e_ring_topology') {
    mpic++ "$f.cpp" -o "build\$f.exe"
}
foreach ($f in 'bench_2c_matrix','bench_2d_primes','bench_2e_max','bench_2f_pi') {
    g++ -O2 -fopenmp "$f.cpp" -o "build\$f.exe"
}
```

Executables land in `build\`. Run them from the project folder, e.g. `.\build\2a_fork_join.exe`
or `mpiexec -n 4 .\build\3a_rank_size.exe`.

## VS Code snippets

`snippets/cpp.json` has every program as a snippet (type the prefix in a `.cpp` file and press
Tab). To install: copy it to `%APPDATA%\Code\User\snippets\cpp.json`, or in VS Code use
Ctrl+Shift+P -> "Snippets: Configure Snippets" -> `cpp` and paste the contents.

| Prefix | Program |
|---|---|
| `omp2a` ... `omp2f` | OpenMP 2a-2f |
| `mpi3a` ... `mpi3e` | MPI 3a-3e |
| `hqsort` | MPI + OpenMP hyperquicksort |
| `bench2c` ... `bench2f` | Benchmarks (2c-2f) |

## Espanso (works in any app, not just VS Code)

`espanso/parallel_lab.yml` has the same programs as text-expander triggers. Copy it to
`%APPDATA%\espanso\match\parallel_lab.yml` (espanso reloads automatically), then type a
trigger anywhere: `:omp2a` ... `:omp2f`, `:mpi3a` ... `:mpi3e`, `:hqsort`, `:bench2c` ... `:bench2f`.
The text is pasted via the clipboard, so editors don't auto-indent or auto-close brackets.

## Troubleshooting

| Problem | Fix |
|---|---|
| `'g++' is not recognized` | Run the PATH line from *Setup* |
| PowerShell won't run `2a.exe` | Use `.\2a.exe` (in `cmd`, plain `2a.exe` works) |
| `mpic++` not found | Install MS-MPI (runtime + SDK) and add its `Bin` folder to PATH |
| MPI hangs on first run | Allow `mpiexec` / `smpd` through the Windows firewall prompt |
| Piped input (`echo 4 \| prog`) does nothing in some tools | Use a file: `.\2d.exe < input.txt` |
| 3c / 3d print "must specify N processors" | Use `-n 4` for 3c and `-n 6` for 3d |
