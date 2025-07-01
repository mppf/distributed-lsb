/* This is a version of Arkouda's RadixSortLSD
   that has been modified to be a self-contained test.
   It is based upon Arkouda commit ca6b25672913ef689c3c734d9e4d83be5bff821a. */
module ArkoudaRadixSortStandalone
{
    /* number of tuples to sort */
    config const n = 128*1024*1024;

    /*
    Bit width of digits for the LSD radix sort and related ops
     */
    config param bitsPerDigit = 16;
    private param numBuckets = 1 << bitsPerDigit;
    private param maskDigit = numBuckets-1;

    // these need to be const for comms/performance reasons

    // tasks per locale based on locale0
    config const numTasks = here.maxTaskPar;
    const Tasks = {0..#numTasks};


    /* BEGIN_IGNORE_FOR_LINE_COUNT (verification code) */
    config const verify = n < 100_000;
    /* END_IGNORE_FOR_LINE_COUNT */

    use BlockDist;
    use Random;
    use RangeChunk;
    use Time;
    use Sort;

    record KeysRanksComparator: keyComparator {
      inline proc key(kr) { const (k, _) = kr; return k; }
    }

    inline proc getDigit(key: uint, rshift: int, last: bool, negs: bool): int {
      return ((key >> rshift) & (maskDigit:uint)):int;
    }

    // calculate sub-domain for task
    inline proc calcBlock(task: int, low: int, high: int) {
        var totalsize = high - low + 1;
        var div = totalsize / numTasks;
        var rem = totalsize % numTasks;
        var rlow: int;
        var rhigh: int;
        if (task < rem) {
            rlow = task * (div+1) + low;
            rhigh = rlow + div;
        }
        else {
            rlow = task * div + rem + low;
            rhigh = rlow + div - 1;
        }
        return {rlow .. rhigh};
    }

    // calc global transposed index
    // (bucket,loc,task) = (bucket * numLocales * numTasks) + (loc * numTasks) + task;
    inline proc calcGlobalIndex(bucket: int, loc: int, task: int): int {
        return ((bucket * numLocales * numTasks) + (loc * numTasks) + task);
    }

    /* Radix Sort Least Significant Digit
       In-place radix sort a block distributed array
       comparator is used to extract the key from array elements
     */
    private proc radixSortLSDCore(ref a:[?aD] ?t, nBits, negs, comparator) {
        var temp = blockDist.createArray(aD, a.eltType);
        temp = a;

        // create a global count array to scan
        // here, counts are in the order of digit, then locale, then task
        var globalCounts = blockDist.createArray(0..<(numLocales*numTasks*numBuckets), int);

        // loop over digits
        for rshift in {0..#nBits by bitsPerDigit} {
            const last = (rshift + bitsPerDigit) >= nBits;
            // count digits
            coforall loc in Locales with (ref globalCounts) {
                on loc {
                    // allocate counts
                    coforall task in Tasks {
                        var taskBucketCounts: [0..#numBuckets] int;
                        // get local domain's indices
                        var lD = temp.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
                        // count digits in this task's part of the array
                        for i in tD {
                            const key = comparator.key(temp.localAccess[i]);
                            var bucket = getDigit(key, rshift, last, negs); // calc bucket from key
                            taskBucketCounts[bucket] += 1;
                        }
                        // copy to the distributed counts array in
                        // transposed order
                        const start = calcGlobalIndex(0, loc.id, task);
                        const stride = calcGlobalIndex(1, loc.id, task) - start;
                        globalCounts[start.. by stride #numBuckets] =
                          taskBucketCounts[0..#numBuckets];
                    }//coforall task
                }//on loc
            }//coforall loc

            // scan globalCounts to get bucket ends on each locale/task
            var globalStarts = + scan globalCounts;
            globalStarts -= globalCounts;

            // calc new positions and permute
            coforall loc in Locales with (ref a) {
                on loc {
                    coforall task in Tasks with (ref a) {
                        var taskBucketPos: [0..#numBuckets] int;
                        // read start pos in to globalStarts back
                        // from transposed order
                        const start = calcGlobalIndex(0, loc.id, task);
                        const stride = calcGlobalIndex(1, loc.id, task) - start;
                        taskBucketPos[0..#numBuckets] =
                          globalStarts[start.. by stride #numBuckets];

                        // get local domain's indices
                        var lD = temp.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
                        // calc new position and put data there in temp
                        {
                            for i in tD {
                                const ref tempi = temp.localAccess[i];
                                const key = comparator.key(tempi);
                                var bucket = getDigit(key, rshift, last, negs); // calc bucket from key
                                var pos = taskBucketPos[bucket];
                                taskBucketPos[bucket] += 1;
                                a[pos] = tempi;
                            }
                        }
                    }//coforall task
                }//on loc
            }//coforall loc

            // copy back to temp for next iteration
            // Only do this if there are more digits left
            if !last {
              temp <=> a;
            }
        } // for rshift
    }//proc radixSortLSDCore

    proc main() {
      writeln(numLocales, " locales with ", numTasks, " tasks per locale");

      var A = blockDist.createArray(0..<n, (uint(64), uint(64)));

      writeln("Generating ", n, " ", A.eltType:string, " elements");

      // fill in random values
      var rs = new randomStream(uint, seed=1);
      forall (elt, i, rnd) in zip(A, A.domain, rs.next(A.domain)) {
        elt[0] = rnd % 4; // TODO put it back
        elt[1] = i;
      }

      writeln("Sorting");

      var t: Time.stopwatch;
      t.start();

      radixSortLSDCore(A, nBits=64, negs=false, new KeysRanksComparator());

      t.stop();

      writeln("Sorted ", n, " elements in ", t.elapsed(), " s");
      writeln("That's ", n/t.elapsed()/1000.0/1000.0, " M elements sorted / s");

      /* BEGIN_IGNORE_FOR_LINE_COUNT (verification code) */
      if verify {
        writeln("Verifying with 1-locale sort");
        var B:[0..<n] (uint(64), uint(64));
        var rs2 = new randomStream(uint, seed=1);
        forall (elt, i, rnd) in zip(B, B.domain, rs2.next(B.domain)) {
          elt[0] = rnd % 4; // TODO
          elt[1] = i;
        }
        Sort.sort(B);
        forall (a, b) in zip (A, B) {
          assert(a == b);
        }
        writeln("Verification OK");
      }
      /* END_IGNORE_FOR_LINE_COUNT */
    }
}
