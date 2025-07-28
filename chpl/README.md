# Chapel Implementation of LSD Radix Sort

This directory includes several [Chapel](https://chapel-lang.org/)
implementations of LSD Radix Sort:

  * `arkouda-radix-sort.chpl` is a standalone version of the LSD Radix
    sort developed as part of the
    [Arkouda](https://github.com/Bears-R-Us/arkouda) project.

  * `arkouda-radix-sort-strided-counts-no-agg.chpl` is the same version
    but with simplified counts transfers (using array slice assignment)
    and no aggregators. A Chapel user first implementing this algorithm
    would likely start with something like this simplified version.

  * `arkouda-radix-sort-strided-counts.chpl` uses simplified counts
    transfers and aggregators for transfering elements.

Additionally, `chpl-std-lib-sort.chpl` is a similar program that uses the
distributed sort functionality added in Chapel 2.5. It is not an LSD
Radix Sort, so it is not directly comparable.

# Building

```
chpl --fast arkouda-radix-sort.chpl
```

# Running

To run on 4 nodes (including possibly simulating 4 nodes locally with
https://chapel-lang.org/docs/platforms/udp.html#using-udp):

```
./arkouda-radix-sort -nl 4
```

Use `--n` to specify the problem size, e.g.:
```
./arkouda-radix-sort -nl 4 --n=100
```

# Details of Measured Versions

Performance was measured on an HPE Cray Supercomputing EX using 128 nodes
(each using 128 cores) and a problem size of 137438953472 elements total
(so the elements require 16 GiB of space per node).

`chpl` version was 2.5.

Compile commands:

```
chpl --fast arkouda-radix-sort-strided-counts-no-agg.chpl
chpl --fast arkouda-radix-sort-strided-counts.chpl
chpl --fast arkouda-radix-sort.chpl
```

Run commands:

```
CHPL_RT_COMM_OFI_DEDICATED_AMH_CORES=true ./arkouda-radix-sort -nl 128x2 --n=137438953472

CHPL_RT_COMM_OFI_DEDICATED_AMH_CORES=true ./arkouda-radix-sort-strided-counts -nl 128x2 --n=137438953472

./arkouda-radix-sort-strided-counts-no-agg -nl 128 --n=137438953472
```

Example Output:

```
256 locales with 63 tasks per locale
Generating 137438953472 2*uint(64) elements
Sorting
Sorted 137438953472 elements in 8.21081 s
That's 16738.8 M elements sorted / s
```

Further details:
 * used the bundled LLVM backend with LLVM 19 (`export CHPL_LLVM=bundled`)
 * used `CHPL_COMM=ofi` and `CHPL_LIBFABRIC=system`
   (the defaults on this system)
 * loaded modules `cray-pmi`, `gcc-native`, and  `PrgEnv-gnu` in addition
   to the system defaults
 * used the system default processor selection module `craype-x86-rome`
 * used these additional environment variables
   ```
   export CHPL_LAUNCHER_MEM=unset
   export CHPL_RT_MAX_HEAP_SIZE="50%"
   export SLURM_UNBUFFEREDIO=1
   ```
