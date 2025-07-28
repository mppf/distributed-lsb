# Comparing Frameworks with Distributed Radix Sorting

This repository contains code for doing a distributed parallel
least-significant-digit-first (LSD) radix sort with several
distributed-memory parallel programming frameworks. These implementations
support comparisons of these different frameworks and their productivity
and performance.

As of July 2025, here are the performance results. These performance
results are in units of millions of 16-byte elements sorted per second on
when sorting 2³⁷ elements on 128 nodes of a HPE Cray Supercomputing EX
using 128 cores per node. The program size reported here is a count of
source lines of code produced by ``count-lines.py``, which processes
special comments to leave out verification and debugging code.

| Variant              | EX Performance           | Source Lines of Code |
| ---                  | ---                   | ---                  |
|                      | in million elements sorted / s |             |
| Chapel, fine-grained | 10,455                | 110                  |
| Chapel, aggregated   | 16,782                | 113                  |
| MPI                  | 7,823                 | 410                  |
| OpenSHMEM, fine-grained  | 3,786             | 291                  |
| Conveyors (over OpenSHMEM)  | 14,687         | 330                  |

PRs contributing improved versions or implementations in other
distributed-memory parallel programming frameworks are welcome!

## What is LSD Radix Sort?

[Least Significant Digit Radix Sort](https://en.wikipedia.org/wiki/Radix_sort)
(or LSD Radix Sort) is a linear-time sort algorithm. It is theoretically
efficient and relatively simple to implement. Because of its simplicity,
it is practical to explore many implementations of it.

LSD Radix Sort operates in passes. Each pass stably sorts by one digit of
the data to be sorted. The first pass sorts the data by the least
significant digit (in the number 1234, 4 is the least significant digit).
The next pass sorts by the next digit (in our example 1234, 3 is the next
digit).  It proceeds in this way until all the digits have been sorted.
Each of these passes has to be stable; that is, it does not change the
order of equal elements (in this case, that is elements with the same
current digit).

A parallel LSD radix sort has the following outline:

* Create arrays `A` and `B` of total size `n`
  * Arrays `A` and `B` store elements evenly among tasks
* for each digit, starting with the least significant, shuffle the data
  by that digit:
  * Each task counts the number of elements with each digit among its
    elements in `A`
  * Collect the counts of digits into a global array, in digit order
  * Perform a parallel prefix sum on that global array to compute the
    output range from each task
  *  Each task copies the elements in its portion of `A` into the
     appropriate output portion of `B`
  * swap arrays `A` and `B` if it is not the last digit


## What is a Distributed-Memory Parallel Programming Framework?

A distributed-memory parallel programming framework is a programming
system that supports programs that run on a
[supercomputer](https://en.wikipedia.org/wiki/Supercomputer)
or [cluster computer](https://en.wikipedia.org/wiki/Computer_cluster).
It is called "distributed memory" because the supercomputer or cluster is
made up of individual nodes and each node has its own memory.
Distributed-memory computation is needed when one node can't solve the
problem fast enough or when solving the problem involves more data than
fits on one node.

Generally speaking, efforts to write programs for these large systems
come with challenges that a programmer needs to address to
achieve good performance. In particular, programs that run in parallel on
a PC or server won't necessarily perform well on a supercomputer or
cluster because the distributed-memory environment has additional
challenges. The main challenges are communication overhead and massive
parallelism.

Communication overhead is an issue because accessing data on a remote
node or coordinating with a remote node is orders of magnitude slower
than accessing local memory.
 * Local memory latency is usually measured in nanoseconds, e.g. 20ns
 * Network latency is usually measured in microseconds, e.g. 2 μs

Massive parallelism used effectively can significantly reduce the time
it takes to do large computations.
Massive parallelism is an issue because, to use the massive amount of
parallelism available in a cluster or supercomputer, the application must
expose that much parallelism and it needs to be able to run it without
too much load imbalance. If you want to run in parallel on a PC, you
might only need to handle keeping 8 cores busy. If you want to run on a
big part of a [Top500 system](https://en.wikipedia.org/wiki/TOP500),
you'll be using thousands or millions of cores.

## Which Frameworks?

This repository provides implementations for distributed,
parallel LSD-radix sort in Chapel, MPI, and OpenSHMEM.
There is a directory in this repository for each framework. See the
README in each directory for details on how to compile and run these and
on the software versions measured.

### Chapel

[Chapel](https://chapel-lang.org/) is a programming language designed for
parallel computing, including distributed-memory parallel computing.

The Chapel LSD Radix sort is implemented using Block-distributed arrays
and `coforall` + `on` statements to create work on different nodes, and
`coforall` statements to use multiple cores on each node. It uses
[aggregators](https://chapel-lang.org/docs/modules/packages/CopyAggregation.html)
to avoid small messages.

### MPI

MPI is a library supporting distributed-memory parallel computing. It can
be used from many languages, but it is most commonly used from C, C++, and
Fortran.

MPI is standardized by the [MPI Forum](https://www.mpi-forum.org/) and
the most common implementations are [MPICH](https://www.mpich.org/) and
[OpenMPI](https://www.open-mpi.org/) or derivatives of these.

The MPI LSD Radix sort here uses C++'s `std::vector` to store the portion
of the array per-rank. It uses a `struct DistributedArray` abstraction to
make it clearer what operations apply to distributed (rather than local)
arrays and to make it more readable to go between global indices and
local indices. In each shuffle step, it stably sorts the local data by
the digit. It uses `MPI_Alltoallv` to communicate counts between the
global counts arrays and the local counts arrays and to shuffle the data.
One challenge with that is that, while the algorithm computes what to
send to each other node, it doesn't directly compute how much to receive
from each other node, so it does a preparatory `MPI_Alltoallv` to share
that information.


### OpenSHMEM

[OpenSHMEM](http://openshmem.org/) is a library for one-sided
communication.

Similarly to the MPI version, the OpenSHMEM LSD Radix sort here uses
C++'s `std::vector` to store the portion of the array per-rank and a
`struct DistributedArray` abstraction to make it clearer what operations
apply to distributed (rather than local) arrays and to make it more
readable to go between global indices and local indices. In each shuffle
step, it stably sorts the local data by the digit. It uses the strided
`iput` and `iget` functions to copy counts data between global counts
arrays and local counts arrays. It uses `shmem_putmem` copy the array
elements.

### Conveyors

[Conveyors](https://github.com/jdevinney/bale/blob/master/docs/uconvey.pdf)
is a library that supports aggregation for many-to-many communication of
small data items. Users of Conveyors need to write loops that enqueue
messages for other nodes and handle messages sent to the current node.
Conveyors can be configured to use OpenSHMEM or MPI.  The Conveyors
implementation of LSD Sort is an optimiza- tion on the OpenSHMEM version,
retaining its structure while integrating conveyors into the
communication-intensive shuffle loop. Thus, it uses Conveyors instead of
`shem_putmem`.


# Get Involved

In addition to comparing the productivity and performance of different
parallel computing frameworks, we'd like to find the the fastest and most
scalable distributed parallel sort. If you know of other parallel sorts
out there that could be competitive, please let us know by opening an
issue in this repository or reaching out to @mppf on social media.

We also would be happy to see pull requests that provide other
implementations of distributed parallel sorts or improve
on the ones here.
