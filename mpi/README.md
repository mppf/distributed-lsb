# MPI Implementation of LSD Radix Sort

`mpi_lsbsort.cpp` is a C++ and MPI implementation of LSD Radix Sort.

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


Performance was measured on an HPE Cray Supercomputing EX using 64 nodes
(each using 128 cores) and a problem size of 68719476736 elements total
(so the elements require 16 GiB of space per node).

Compile command:

```
CC -O3 mpi_lsbsort.cpp -o mpi_lsbsort -I pcg-cpp/include/
```

Run command:

```
srun --nodes 64 --ntasks-per-node=128 ./mpi_lsbsort --n 68719476736
```

Output:

```
Total number of MPI ranks: 8192
Problem size: 68719476736
Generating random values
Generated random values in 0.222903 s
Sorting
Sorted 68719476736 values in 82.076
That's 837.267 M elements sorted / s
```

Other details:
 * loaded modules `PrgEnv-gnu` and `gcc-native/14.2`
 * used `cray-mpich/8.1.32`
 * used the system default processor selection module `craype-x86-rome`
 * used these additional environment variables
   ```
   export SLURM_UNBUFFEREDIO=1
   ```
