#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cassert>
#include <cstdint>

#include <unistd.h>

#include <shmem.h>
#include <pcg_random.hpp>

extern "C" {
#include "convey.h"
}

#define RADIX 16
#define N_DIGITS (64/RADIX)
#define N_BUCKETS (1 << RADIX)
#define COUNTS_SIZE ((int64_t) N_BUCKETS)
#define MASK (N_BUCKETS - 1)
using counts_array_t = std::array<int64_t, COUNTS_SIZE>;

// the elements to sort
struct SortElement {
  uint64_t key = 0; // to sort by
  uint64_t val = 0; // carried along
};

struct IdxSortElement {
  int64_t locIdx;
  SortElement value;
};

struct IdxValue {
  int64_t locIdx;
  int64_t value;
};

/* BEGIN_IGNORE_FOR_LINE_COUNT (printing code) */
std::ostream& printhex(std::ostream& os, uint64_t key) {
  std::ios oldState(nullptr);
  oldState.copyfmt(os);
  os << std::setfill('0') << std::setw(16) << std::hex << key;
  os.copyfmt(oldState);
  return os;
}
std::ostream& operator<<(std::ostream& os, const SortElement& x) {
  os << "(";
  printhex(os, x.key);
  os << "," << x.val << ")";
  return os;
}
/* END_IGNORE_FOR_LINE_COUNT */

bool operator==(const SortElement& x, const SortElement& y) {
  return x.key == y.key && x.val == y.val;
}
bool operator<(const SortElement& x, const SortElement& y) {
  return x.key < y.key;
}

// to help with sending sort elements to remote locales
// helper to divide while rounding up
static inline int64_t divCeil(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// Store a different type for distributed arrays just to make the code
// clearer.
// This actually just stores the current rank's portion of a distributed
// array along with some metadata.
// It doesn't support communication directly. Communication is expected
// to happen in the form of shmem calls working with localPart().
template<typename EltType>
struct DistributedArray {
  struct RankAndLocalIndex {
    int rank = 0;
    int64_t locIdx = 0;
  };

  std::string name_;
  EltType* localPart_ = nullptr;
  int64_t numElementsTotal_ = 0;    // number of elements on all ranks
  int64_t numElementsPerRank_ = 0 ; // number per rank
  int64_t numElementsHere_ = 0;     // number this rank
  int myRank_ = 0;
  int numRanks_ = 0;

  static DistributedArray<EltType>
  create(std::string name, int64_t totalNumElements);

  ~DistributedArray() {
    if (localPart_ != nullptr) {
      shmem_free(localPart_);
    }
  }

  // convert a local index to a global index
  inline int64_t localIdxToGlobalIdx(int64_t locIdx) const {
    return myRank_*numElementsPerRank_ + locIdx;
  }
  // convert a global index into a local index
  inline RankAndLocalIndex globalIdxToLocalIdx(int64_t glbIdx) const {
    RankAndLocalIndex ret;
    int64_t rank = glbIdx / numElementsPerRank_;
    int64_t locIdx = glbIdx - rank*numElementsPerRank_;
    ret.rank = rank;
    ret.locIdx = locIdx;
    return ret;
  }

  // accessors
  inline const std::string& name() const { return name_; }
  inline const EltType* localPart() const { return localPart_; }
  inline EltType* localPart() { return localPart_; }
  inline int64_t numElementsTotal() const { return numElementsTotal_; }
  inline int64_t numElementsPerRank() const { return numElementsPerRank_; }
  inline int64_t numElementsHere() const { return numElementsHere_; }
  inline int myRank() const { return myRank_; }
  inline int numRanks() const { return numRanks_; }

  /* BEGIN_IGNORE_FOR_LINE_COUNT (printing and verification code) */
  // helper to print part of the distributed array
  void print(int64_t nToPrintPerRank) const;
  bool checkSorted() const;
  /* END_IGNORE_FOR_LINE_COUNT */
};

template<typename EltType>
DistributedArray<EltType>
DistributedArray<EltType>::create(std::string name, int64_t totalNumElements) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  int64_t eltsPerRank = divCeil(totalNumElements, numRanks);
  int64_t eltsHere = eltsPerRank;
  if (eltsPerRank*myRank + eltsHere > totalNumElements) {
    eltsHere = totalNumElements - eltsPerRank*myRank;
  }
  if (eltsHere < 0) eltsHere = 0;

  DistributedArray<EltType> ret;
  ret.name_ = std::move(name);
  ret.localPart_ = (EltType*) shmem_malloc(eltsPerRank * sizeof(EltType));
  ret.numElementsTotal_ = totalNumElements;
  ret.numElementsPerRank_ = eltsPerRank;
  ret.numElementsHere_ = eltsHere;
  ret.myRank_ = myRank;
  ret.numRanks_ = numRanks;

  return ret;
}

/* BEGIN_IGNORE_FOR_LINE_COUNT (printing code) */
static void flushOutput() {
  // this is a workaround to make it more likely that the output is printed
  // to the terminal in the correct order.
  // *it might not work*
  std::cout << std::flush;
  usleep(100);
}

template<typename EltType>
void DistributedArray<EltType>::print(int64_t nToPrintPerRank) const {
  shmem_barrier_all();

  if (myRank_ == 0) {
    if (nToPrintPerRank*numRanks_ >= numElementsTotal_) {
      std::cout << name_ << ": displaying all "
                << numElementsTotal_ << " elements\n";
    } else {
      std::cout << name_ << ": displaying first " << nToPrintPerRank
                << " elements on each rank"
                << " out of " << numElementsTotal_ << " elements\n";
    }
  }

  for (int rank = 0; rank < numRanks_; rank++) {
    if (myRank_ == rank) {
      int64_t i = 0;
      for (i = 0; i < nToPrintPerRank && i < numElementsHere_; i++) {
        int64_t glbIdx = localIdxToGlobalIdx(i);
        std::cout << name_ << "[" << glbIdx << "] = " << localPart_[i] << " (rank " << myRank_ << ")\n";
      }
      if (i < numElementsHere_) {
        std::cout << "...\n";
      }
      flushOutput();
    }
    shmem_barrier_all();
  }
}
/* END_IGNORE_FOR_LINE_COUNT */

/* BEGIN_IGNORE_FOR_LINE_COUNT (verification code) */
template<typename EltType>
bool DistributedArray<EltType>::checkSorted() const {
    int myRank = 0;
    int numRanks = 0;
    myRank = shmem_my_pe();
    numRanks = shmem_n_pes();

    int8_t* locallySorted = (int8_t*)shmem_malloc(sizeof(int8_t));
    EltType* myBoundaries = (EltType*)shmem_malloc(2*sizeof(EltType));
    EltType* allBoundaries = (EltType*)shmem_malloc(2*numRanks*sizeof(EltType));
    int64_t* numBoundaries = (int64_t*)shmem_malloc(sizeof(int64_t));

    // Check if the local array is sorted
    std::vector<EltType> myVec(localPart_, localPart_+numElementsHere_);
    *locallySorted = std::is_sorted(myVec.begin(), myVec.end());

    // Check if the boundaries between PEs are sorted (make sure that the last
    // element on myRank < first element on myRank+1). To simplify cases where
    // the number of elements per rank isn't uniform just exchange first/last
    // with every rank and check all boundaries locally. When there's fewer
    // elements than ranks, some ranks can have duplicates (but that's fine)
    // and others will have uninitialized memory that won't be looked at.
    *numBoundaries = 0;
    if (numElementsHere_) {
      myBoundaries[0] = myVec.front();
      myBoundaries[1] = myVec.back();
      *numBoundaries = 2;
    }

    shmem_barrier_all();
    shmem_fcollectmem(SHMEM_TEAM_WORLD, allBoundaries, myBoundaries, 2*sizeof(EltType));
    shmem_int8_and_reduce(SHMEM_TEAM_WORLD, locallySorted, locallySorted, 1);
    shmem_int64_sum_reduce(SHMEM_TEAM_WORLD, numBoundaries, numBoundaries, 1);
    shmem_barrier_all();

    std::vector<EltType> boundariesVec(allBoundaries, allBoundaries+*numBoundaries);
    bool boundariesSorted = std::is_sorted(boundariesVec.begin(), boundariesVec.end());

    return *locallySorted && boundariesSorted;
}
/* END_IGNORE_FOR_LINE_COUNT */

// compute the bucket for a value when sort is on digit 'd'
inline int getBucket(SortElement x, int d) {
  return (x.key >> (RADIX*d)) & MASK;
}

void copyCountsToGlobalCounts(counts_array_t& localCounts,
                              DistributedArray<int64_t>& GlobalCounts, convey_t * request) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  // Now, each rank has an array of counts, like this
  //  [r0d0, r0d1, ... r0d255]  | on rank 0
  //  [r1d0, r1d1, ... r1d255]  | on rank 1
  //  ...
  //

  // We need to transpose these so that the counts have the
  // starting digits first
  //  [r0d0, r1d0, r2d0, ...]   | on rank 0
  //  [r0d1, r1d1, r2d1, ...]   |
  //  [r0d2, r1d2, r2d2, ...]   | on rank 1 ...
  //  ...

  int64_t i = 0;
  convey_begin(request, sizeof(IdxValue), alignof(IdxValue));
  while (convey_advance(request, i == COUNTS_SIZE)) {
    int64_t* GCA = &GlobalCounts.localPart()[0]; // it's symmetric
    for (; i < COUNTS_SIZE; i++) {
      int64_t dstGlobalIdx = i*numRanks + myRank;
      auto dst = GlobalCounts.globalIdxToLocalIdx(dstGlobalIdx);
      int dstRank = dst.rank;

      IdxValue payload = { .locIdx = dst.locIdx, .value = localCounts[i] };
      if (! convey_push(request, &payload, dst.rank))
        break;
    }

    IdxValue local;
    while( convey_pull(request, &local, NULL) == convey_OK)
      GCA[local.locIdx] = local.value;

  }
  convey_reset(request);

  shmem_barrier_all();
}

void exclusiveScan(const DistributedArray<int64_t>& Src,
                   DistributedArray<int64_t>& Dst) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  // Now compute the total of for each chunk of the global src array
  int64_t myTotal = 0;
  for (int64_t i = 0; i < Src.numElementsHere(); i++) {
    myTotal += Src.localPart()[i];
  }

  // allocate a remotely accessible array
  // only rank 0's values will be used
  int64_t* PerRankStarts = (int64_t*) shmem_malloc(sizeof(int64_t) * numRanks);

  // Send the total from each rank to rank 0
  shmem_int64_p(PerRankStarts + myRank, myTotal, 0);

  // wait for rank 0 to get all of them
  shmem_barrier_all();

  if (myRank == 0) {
    // change from counts per rank to starts per rank
    int64_t sum = 0;
    for (int i = 0; i < numRanks; i++) {
      int64_t count = PerRankStarts[i];
      PerRankStarts[i] = sum;
      sum += count;
    }
    // send the start for each rank to that rank
    for (int i = 0; i < numRanks; i++) {
      shmem_int64_p(PerRankStarts + i, PerRankStarts[i], i);
    }
  }

  // wait for rank 0 to dissemenate starts
  shmem_barrier_all();

  int64_t myGlobalStart = PerRankStarts[myRank];

  // scan the region in each rank
  {
    int64_t sum = myGlobalStart;
    int64_t nHere = Dst.numElementsHere();
    for (int64_t i = 0; i < nHere; i++) {
      Dst.localPart()[i] = sum;
      sum += Src.localPart()[i];
    }
  }

  shmem_free(PerRankStarts);
}

void copyStartsFromGlobalStarts(DistributedArray<int64_t>& GlobalStarts,
                                counts_array_t& localStarts, convey_t* request,
                                convey_t* reply) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  // starts look like this:
  //  [r0d0, r1d0, r2d0, ...]   | on rank 0
  //  [r0d1, r1d1, r2d1, ...]   |
  //  [r0d2, r1d2, r2d2, ...]   | on rank 1 ...
  //  ...

  // Need to get the values for each rank so it's like this:
  //  [r0d0, r0d1, ... r0d255]  | on rank 0
  //  [r1d0, r1d1, ... r1d255]  | on rank 1
  //  ...
  //

  convey_begin(request, sizeof(IdxValue), alignof(IdxValue));
  convey_begin(reply, sizeof(IdxValue), alignof(IdxValue));

  int64_t* GSA = GlobalStarts.localPart(); // it's symmetric

  int64_t i = 0;
  bool more;
  while (more = convey_advance(request, i == COUNTS_SIZE),
	 more | convey_advance(reply, !more)) {
    for (; i < COUNTS_SIZE; i++) {
      int64_t srcGlobalIdx = i*numRanks + myRank;
      auto src = GlobalStarts.globalIdxToLocalIdx(srcGlobalIdx);
      int srcRank = src.rank;
      int64_t srcIndex = src.locIdx;

      IdxValue packet = { .locIdx = i, .value = srcIndex };
      if (! convey_push(request, &packet, srcRank))
	break;
    }

    IdxValue* p;
    int64_t from;
    while ((p = (IdxValue*)convey_apull(request, &from)) != NULL) {
      IdxValue packet = { .locIdx = p->locIdx, .value = GSA[p->value] };
      if (! convey_push(reply, &packet, from)) {
	convey_unpull(request);
	break;
      }
    }

    while ((p = (IdxValue*)convey_apull(reply, NULL)) != NULL)
      localStarts[p->locIdx] = p->value;
  }

  convey_reset(request);
  convey_reset(reply);

  shmem_barrier_all();
}

// shuffles the data from A into B
void globalShuffle(DistributedArray<SortElement>& A,
                   DistributedArray<SortElement>& B,
                   int digit, convey_t* request, convey_t* reply) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  auto starts = std::make_unique<counts_array_t>();
  auto counts = std::make_unique<counts_array_t>();

  // clear out starts and counts
  starts->fill(0);
  counts->fill(0);

  // compute the count for each digit
  int64_t locN = A.numElementsHere();
  SortElement* localPart = A.localPart();
  for (int64_t i = 0; i < locN; i++) {
    SortElement elt = localPart[i];
    (*counts)[getBucket(elt, digit)] += 1;
  }

  // Now, each rank has an array of counts, like this
  //  [r0d0, r0d1, ... r0d255]  | on rank 0
  //  [r1d0, r1d1, ... r1d255]  | on rank 1
  //  ...
  //

  // We need to transpose these so that the counts have the
  // starting digits first
  //  [r0d0, r1d0, r2d0, ...]   | on rank 0
  //  [r0d1, r1d1, r2d1, ...]   |
  //  [r0d2, r1d2, r2d2, ...]   | on rank 1 ...
  //  ...

  // create a distributed array storing the result of this transposition
  auto GlobalCounts = DistributedArray<int64_t>::create("GlobalCounts",
                                                        COUNTS_SIZE*numRanks);
  // and one storing the start positions for each task
  // (that will be the result of a scan operation)
  auto GlobalStarts = DistributedArray<int64_t>::create("GlobalStarts",
                                                        COUNTS_SIZE*numRanks);

  // copy the per-bucket counts to the global counts array
  copyCountsToGlobalCounts(*counts, GlobalCounts, request);

  // scan to fill in GlobalStarts
  exclusiveScan(GlobalCounts, GlobalStarts);

  // copy the per-bucket starts from the global counts array
  copyStartsFromGlobalStarts(GlobalStarts, *starts, request, reply);

  // Now go through the data in B assigning each element its final
  // position and sending that data to the other ranks
  // Leave the result in B
  convey_begin(request, sizeof(IdxSortElement), alignof(IdxSortElement));

  SortElement* GB = B.localPart(); // it's symmetric
  int64_t i = 0;
  while (convey_advance(request, i == locN)) {
    for (; i < locN; i++) {
      SortElement elt = localPart[i];
      int bucket = getBucket(elt, digit);
      int64_t &next = (*starts)[bucket];
      int64_t dstGlobalIdx = next;

      // store 'elt' into 'dstGlobalIdx'
      auto dst = B.globalIdxToLocalIdx(dstGlobalIdx);

      assert(0 <= dst.rank && dst.rank < numRanks);
      IdxSortElement payload = { .locIdx = dst.locIdx, .value = elt };
      if (! convey_push(request, &payload, dst.rank))
        break;

      next += 1;
    }

    IdxSortElement* local;
    while((local = (IdxSortElement*)convey_apull(request, NULL)) != NULL) {
      GB[local->locIdx] = local->value;
    }

  }
  convey_reset(request);

}

// Sort the data in A, using B as scratch space.
void mySort(DistributedArray<SortElement>& A,
            DistributedArray<SortElement>& B) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  convey_t* request = convey_new(SIZE_MAX, 0, NULL, convey_opt_SCATTER);
  convey_t* reply = convey_new(SIZE_MAX, 0, NULL, 0);
  assert(N_DIGITS % 2 == 0);
  for (int digit = 0; digit < N_DIGITS; digit += 2) {
    globalShuffle(A, B, digit,   request, reply);
    globalShuffle(B, A, digit+1, request, reply);
  }
  convey_free(request);
  convey_free(reply);
}

int main(int argc, char *argv[]) {
  shmem_init();

  // read in the problem size

  /* BEGIN_IGNORE_FOR_LINE_COUNT (printing and verification code) */
  bool printSome = false;
  bool verify = true;
  /* END_IGNORE_FOR_LINE_COUNT */

  int64_t n = 100*1000*1000;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--n") {
      n = std::stoll(argv[++i]);
    }

    /* BEGIN_IGNORE_FOR_LINE_COUNT (printing and verification code) */
    else if (std::string(argv[i]) == "--print") {
      printSome = true;
    } else if (std::string(argv[i]) == "--verify") {
      verify = true;
    } else if (std::string(argv[i]) == "--no-verify") {
      verify = false;
    }
    /* END_IGNORE_FOR_LINE_COUNT */
  }

  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  if (myRank == 0) {
    std::cout << "Total number of shmem PEs: " << numRanks << "\n";
    std::cout << "Problem size: " << n << "\n";
    flushOutput();
  }

  // create distributed arrays A and B
  auto A = DistributedArray<SortElement>::create("A", n);
  auto B = DistributedArray<SortElement>::create("B", n);

  // set the keys to random values and the values to global indices
  {
    auto start = std::chrono::steady_clock::now();
    if (myRank == 0) {
      std::cout << "Generating random values\n";
      /* BEGIN_IGNORE_FOR_LINE_COUNT (printing) */
      flushOutput();
      /* END_IGNORE_FOR_LINE_COUNT */
    }

    auto rng = pcg64(myRank);
    int64_t locN = A.numElementsHere();
    for (int64_t i = 0; i < locN; i++) {
      auto& elt = A.localPart()[i];
      elt.key = rng();
      elt.val = A.localIdxToGlobalIdx(i);
    }

    shmem_barrier_all();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (myRank == 0) {
      std::cout << "Generated random values in " << elapsed.count() << " s\n";
      /* BEGIN_IGNORE_FOR_LINE_COUNT (printing) */
      flushOutput();
      /* END_IGNORE_FOR_LINE_COUNT */
    }
    shmem_barrier_all();
  }

  /* BEGIN_IGNORE_FOR_LINE_COUNT (printing) */

  // Print out the first few elements on each locale
  if (printSome) {
    A.print(10);
  }

  /* END_IGNORE_FOR_LINE_COUNT */


  // Shuffle the data in-place to sort by the current digit
  {
    if (myRank == 0) {
      std::cout << "Sorting\n";
      /* BEGIN_IGNORE_FOR_LINE_COUNT (printing) */
      flushOutput();
      /* END_IGNORE_FOR_LINE_COUNT */
    }

    shmem_barrier_all();
    auto start = std::chrono::steady_clock::now();

    mySort(A, B);

    shmem_barrier_all();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (myRank == 0) {
      std::cout << "Sorted " << n << " values in " << elapsed.count() << "\n";;
      std::cout << "That's " << n/elapsed.count()/1000.0/1000.0
                << " M elements sorted / s\n";
      flushOutput();
    }
    shmem_barrier_all();
  }

  /* BEGIN_IGNORE_FOR_LINE_COUNT (printing and verification code) */

  // Print out the first few elements on each locale
  if (printSome) {
    A.print(10);
  }

  bool sorted = true;
  if (verify) {
    sorted = A.checkSorted();
    if (myRank == 0) {
      if (sorted) {
        std::cout << "Array is sorted\n";
      } else {
        std::cout << "Array is NOT sorted\n";
      }
    }
  }

  /* END_IGNORE_FOR_LINE_COUNT */

  // this seems to cause crashes/hangs with openmpi shmem / osss-ucx
  //shmem_finalize();

  return !sorted;
}
