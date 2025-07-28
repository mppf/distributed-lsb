# OpenSHMEM Implementation of LSD Radix Sort

`shmem_lsbsort.cpp` is a C++ and OpenSHMEM implementation of LSD Radix
Sort. `shmem_lsbsort_convey.cpp` builds upon that version by using
[Conveyors](https://github.com/jdevinney/bale/blob/master/docs/uconvey.pdf)
instead of `put_shmem` to move elements.

Open-source implementations of OpenSHMEM are available:

 * [osss-ucx](https://github.com/openshmem-org/osss-ucx)
 * [Sandia OpenSHMEM](https://github.com/Sandia-OpenSHMEM/SOS)
 * [OpenMPI](https://www.open-mpi.org/) (when configured to include
   OpenSHMEM)

Additionally, [Cray OpenSHMEMX](https://cray-openshmemx.readthedocs.io)
is available on Cray EX systems.

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

Building the Conveyors version involves downloading and buildig
Conveyors. There are some commands to do so in `build.sh`.

# Running

oshrun can launch it to run multiple PEs (aka processes, or ranks) on the
local system. For example, the following command runs with 3 ranks and
sorts 100 elements:

```
oshrun -np 3 ./shmem_lsbsort --n 100
```

# Details of Measured Version

Performance was measured on an HPE Cray Supercomputing EX using 128 nodes
(each using 128 cores) and a problem size of 137438953472 elements total
(so the elements require 16 GiB of space per node).

Compile commands:

```
CC -O3 -DNDEBUG shmem_lsbsort.cpp -o shmem_lsbsort -I pcg-cpp/include/

CC -O3 -DNDEBUG shmem_lsbsort_convey.cpp -o shmem_lsbsort_convey -I pcg-cpp/include/ $(pkg-config --cflags --libs convey)
```

Run commands:

```
srun --hint=nomultithread --nodes 128 --ntasks-per-node=128 ./shmem_lsbsort --n 137438953472

srun --hint=nomultithread --nodes 128 --ntasks-per-node=128 ./shmem_lsbsort_convey --n 137438953472
```

Example Output:

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
