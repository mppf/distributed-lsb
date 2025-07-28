# MPI Implementation of LSD Radix Sort

`mpi_lsbsort.cpp` is a C++ and MPI implementation of LSD Radix Sort.

`mpi_lsbsort_onesided.cpp` is an attempt to use MPI one-sided calls to
improve the radix sort performance (although we did not observe it to
perform better than the original).

Additionally, `comparison-kadis-ams.cpp` is a similar program that uses
the [KaDiS AMS distributed
sort](https://github.com/MichaelAxtmann/KaDiS/). It is not a radix sort,
so it is not directly comparable.

`comparison-kadis-ams.cpp

# Building

First, it is necessary to get a copy of the PCG random number generator,
if it is not already available through a system-wide installation:

```
git clone https://github.com/imneme/pcg-cpp.git
```

Next, this program can be compiled with `mpic++` (or just CC on a Cray
system):

```
mpic++ -O3 mpi_lsbsort.cpp -o mpi_lsbsort -I pcg-cpp/include/
```


# Running

mpirun can launch it to run multiple ranks on the local system:

```
mpirun -n 4 ./mpi_lsbsort --n 100
```

# Details of Measured Version


Performance was measured on an HPE Cray Supercomputing EX using 128 nodes
(each using 128 cores) and a problem size of 137438953472 elements total
(so the elements require 16 GiB of space per node).

Compile command:

```
CC -DNDEBUG -O3 mpi_lsbsort.cpp -o mpi_lsbsort -I pcg-cpp/include/
```

Run command:

```
srun --exclusive --nodes 128 --ntasks-per-node=16 --cpus-per-task=8 --cpu-bind=ldoms ./mpi_lsbsort --n 137438953472
```

Example Output:

```
Total number of MPI ranks: 2048
Problem size: 137438953472
Generating random values
Generated random values in 0.208902 s
Sorting
Sorted 137438953472 values in 17.0758
That's 8048.74 M elements sorted / s
```

Other details:
 * loaded modules `PrgEnv-gnu` and `gcc-native/14.2`
 * used `cray-mpich/8.1.32`
 * used the system default processor selection module `craype-x86-rome`
 * used these additional environment variables
   ```
   export SLURM_UNBUFFEREDIO=1
   ```
