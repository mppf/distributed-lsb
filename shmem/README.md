# OpenSHMEM Implementation of LSD Radix Sort

`shmem_lsbsort.cpp` is a C++ and OpenSHMEM implementation of LSD Radix
Sort.  An open-source implementation of OpenSHMEM is available at
[osss-ucx](https://github.com/openshmem-org/osss-ucx).

# Building

First, it is necessary to get a copy of the PCG random number generator,
if it is not already available through a system-wide installation:

```
git clone https://github.com/imneme/pcg-cpp.git
```

Next, this program can be compiled with `oshc++` (or just CC on a Cray
system):

```
oshc++ -O3 shmem_lsbsort.cpp -o shmem_lsbsort -I pcg-cpp/include/
```


# Running

oshrun can launch it to run multiple PEs (aka processes, or ranks) on the
local system. For example, the following command runs with 3 ranks and
sorts 100 elements:

```
oshrun -np 3 ./shmem_lsbsort --n 100
```

# Details of Measured Version


Performance was measured on an HPE Cray Supercomputing EX using 64 nodes
(each using 128 cores) and a problem size of 68719476736 elements total
(so the elements require 16 GiB of space per node).

Compile command:

```
CC -O3 -DNDEBUG shmem_lsbsort.cpp -o shmem_lsbsort -I pcg-cpp/include/
```

Run command:

```
srun --nodes 64 --ntasks-per-node=128 --hint=nomultithread ./shmem_lsbsort --n 68719476736
```

Output:

```
Total number of shmem PEs: 8192
Problem size: 68719476736
Generating random values
Generated random values in 0.220942 s
Sorting
Sorted 68719476736 values in 36.476
That's 1883.97 M elements sorted / s
```

Other details:
 * used `PrgEnv-gnu` and `gcc-native/14.2`
 * used `cray-openshmemx/11.7.4`
 * used these additional environment variables
   ```
   export SLURM_UNBUFFEREDIO=1
   ```
