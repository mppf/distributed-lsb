// Version of the benchmark using the Chapel standard library sort
// For this benchmark, that will use a most-significant-digit first
// distributed radix sort.
// Uses features added in Chapel 2.5.
module StandardLibrarySort
{
    /* number of tuples to sort */
    config const n = 128*1024*1024;

    /* BEGIN_IGNORE_FOR_LINE_COUNT (verification code) */
    config const verify = n < 100_000;
    /* END_IGNORE_FOR_LINE_COUNT */

    use BlockDist;
    use Random;
    use Time;
    use Sort;

    record KeysRanksComparator: keyComparator {
      inline proc key(kr) { const (k, _) = kr; return k; }
    }

    proc main() {
      const numTasks = here.maxTaskPar;
      writeln(numLocales, " locales with ", numTasks, " tasks per locale");

      var A = blockDist.createArray(0..<n, (uint(64), uint(64)));

      writeln("Generating ", n, " ", A.eltType:string, " elements");

      // fill in random values
      var rs = new randomStream(uint, seed=1);
      forall (elt, i, rnd) in zip(A, A.domain, rs.next(A.domain)) {
        elt[0] = rnd;
        elt[1] = i;
      }

      writeln("Sorting");

      var t: Time.stopwatch;
      t.start();

      sort(A, new KeysRanksComparator());

      t.stop();

      writeln("Sorted ", n, " elements in ", t.elapsed(), " s");
      writeln("That's ", n/t.elapsed()/1000.0/1000.0, " M elements sorted / s");

      /* BEGIN_IGNORE_FOR_LINE_COUNT (verification code) */
      if verify {
        writeln("Verifying with 1-locale sort");
        var B:[0..<n] (uint(64), uint(64));
        var rs2 = new randomStream(uint, seed=1);
        forall (elt, i, rnd) in zip(B, B.domain, rs2.next(B.domain)) {
          elt[0] = rnd;
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
