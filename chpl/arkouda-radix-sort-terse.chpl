// this is a version of arkouda-radix-sort.chpl with comments removed
// and lines limited to 80 characters for source code size comparisons.
module ArkoudaRadixSortStandalone
{
    config const n = 128*1024*1024;

    config param bitsPerDigit = 16;
    private param numBuckets = 1 << bitsPerDigit;
    private param maskDigit = numBuckets-1;

    config const numTasks = here.maxTaskPar;
    const Tasks = {0..#numTasks};

    use BlockDist;
    use CopyAggregation;
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

    inline proc calcGlobalIndex(bucket: int, loc: int, task: int): int {
        return ((bucket * numLocales * numTasks) + (loc * numTasks) + task);
    }

    private proc radixSortLSDCore(ref a:[?aD] ?t, nBits, negs, comparator) {
        var temp = blockDist.createArray(aD, a.eltType);
        temp = a;

        var globalCounts =
          blockDist.createArray(0..<(numLocales*numTasks*numBuckets), int);

        for rshift in {0..#nBits by bitsPerDigit} {
            const last = (rshift + bitsPerDigit) >= nBits;
            coforall loc in Locales with (ref globalCounts) {
                on loc {
                    var tasksBucketCounts: [Tasks] [0..#numBuckets] int;
                    coforall task in Tasks with (ref tasksBucketCounts) {
                        ref taskBucketCounts = tasksBucketCounts[task];
                        var lD = temp.localSubdomain();
                        var tD = calcBlock(task, lD.low, lD.high);
                        for i in tD {
                            const key = comparator.key(temp.localAccess[i]);
                            var bucket = getDigit(key, rshift, last, negs);
                            taskBucketCounts[bucket] += 1;
                        }
                    }
                    coforall tid in Tasks
                      with (ref tasksBucketCounts, ref globalCounts) {
                        var aggregator = new DstAggregator(int);
                        for task in Tasks {
                            ref taskBucketCounts = tasksBucketCounts[task];
                            for bucket in chunk(0..#numBuckets, numTasks, tid) {
                                aggregator.copy(
                                    globalCounts[
                                      calcGlobalIndex(bucket, loc.id, task)],
                                    taskBucketCounts[bucket]);
                            }
                        }
                        aggregator.flush();
                    }
                }
            }

            var globalStarts = + scan globalCounts;
            globalStarts -= globalCounts;

            coforall loc in Locales with (ref a) {
                on loc {
                    var tasksBucketPos: [Tasks] [0..#numBuckets] int;
                    coforall tid in Tasks with (ref tasksBucketPos) {
                        var aggregator = new SrcAggregator(int);
                        for task in Tasks {
                            ref taskBucketPos = tasksBucketPos[task];
                            for bucket in chunk(0..#numBuckets, numTasks, tid) {
                              aggregator.copy(taskBucketPos[bucket],
                                   globalStarts[
                                     calcGlobalIndex(bucket, loc.id, task)]);
                            }
                        }
                        aggregator.flush();
                    }
                    coforall task in Tasks with (ref tasksBucketPos, ref a) {
                        ref taskBucketPos = tasksBucketPos[task];
                        var lD = temp.localSubdomain();
                        var tD = calcBlock(task, lD.low, lD.high);
                        {
                            var aggregator = new DstAggregator(t);
                            for i in tD {
                                const ref tempi = temp.localAccess[i];
                                const key = comparator.key(tempi);
                                var bucket = getDigit(key, rshift, last, negs);
                                var pos = taskBucketPos[bucket];
                                taskBucketPos[bucket] += 1;
                                aggregator.copy(a[pos], tempi);
                            }
                            aggregator.flush();
                        }
                    }
                }
            }

            if !last {
              temp <=> a;
            }
        }
    }

    proc main() {
      var A = blockDist.createArray(0..<n, (uint(64), uint(64)));

      writeln("Generating ", n, " ", A.eltType:string, " elements");

      var rs = new randomStream(uint, seed=1);
      forall (elt, i, rnd) in zip(A, A.domain, rs.next(A.domain)) {
        elt[0] = rnd;
        elt[1] = i;
      }

      writeln("Sorting");

      var t: Time.stopwatch;
      t.start();

      radixSortLSDCore(A, nBits=64, negs=false, new KeysRanksComparator());

      t.stop();

      writeln("Sorted ", n, " elements in ", t.elapsed(), " s");
      writeln("That's ", n/t.elapsed()/1000.0/1000.0, " M elements sorted / s");
    }
}
