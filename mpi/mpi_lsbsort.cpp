#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cassert>
#include <climits>
#include <cstdint>

#include <unistd.h>

#include <mpi.h>
#include <pcg_random.hpp>

// Note: this code assumes that errors returned by MPI calls do not need to be
// handled. That's the case if they are set to halt the program or
// raise exeptions.

#define RADIX 16
#define N_DIGITS (64/RADIX)
#define N_BUCKETS (1 << RADIX)
#define COUNTS_SIZE (N_BUCKETS + 1)
#define MASK (N_BUCKETS - 1)
using counts_array_t = std::array<int64_t, COUNTS_SIZE>;

// the elements to sort
struct SortElement {
  uint64_t key = 0; // to sort by
  uint64_t val = 0; // carried along
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
static MPI_Datatype gSortElementMpiType;

// to help in communicating the counts
struct CountBufElt {
  int32_t digit = 0;
  int32_t rank = 0;
  int64_t count = 0;
};
/* BEGIN_IGNORE_FOR_LINE_COUNT (printing code) */
std::ostream& operator<<(std::ostream& os, const CountBufElt& x) {
  os << "(" << x.digit << "," << x.rank << "," << x.count << ")";
  return os;
}
/* END_IGNORE_FOR_LINE_COUNT */
static MPI_Datatype gCountBufEltMpiType;

// to help in communicating the elements
struct ShuffleBufSortElement {
  uint64_t key = 0;
  uint64_t val = 0;
  int64_t  dstGlobalIdx = 0;
};
/* BEGIN_IGNORE_FOR_LINE_COUNT (printing code) */
std::ostream& operator<<(std::ostream& os, const ShuffleBufSortElement& x) {
  os << "(";
  printhex(os, x.key);
  os << "," << x.val << "," << x.dstGlobalIdx << ")";
  return os;
}
/* END_IGNORE_FOR_LINE_COUNT */
static MPI_Datatype gShuffleBufSortElementMpiType;


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
// to happen in the form of MPI calls working with localPart().
template<typename EltType>
struct DistributedArray {
  struct RankAndLocalIndex {
    int rank = 0;
    int64_t locIdx = 0;
  };

  std::string name_;
  std::vector<EltType> localPart_;
  int64_t numElementsTotal_ = 0;    // number of elements on all ranks
  int64_t numElementsPerRank_ = 0 ; // number per rank
  int64_t numElementsHere_ = 0;     // number this rank
  int myRank_ = 0;
  int numRanks_ = 0;

  static DistributedArray<EltType>
  create(std::string name, int64_t totalNumElements);

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
  inline const std::vector<EltType>& localPart() const { return localPart_; }
  inline std::vector<EltType>& localPart() { return localPart_; }
  inline int64_t numElementsTotal() const { return numElementsTotal_; }
  inline int64_t numElementsPerRank() const { return numElementsPerRank_; }
  inline int64_t numElementsHere() const { return numElementsHere_; }
  inline int myRank() const { return myRank_; }
  inline int numRanks() const { return numRanks_; }

  /* BEGIN_IGNORE_FOR_LINE_COUNT (printing) */
  // helper to print part of the distributed array
  void print(int64_t nToPrintPerRank) const;
  /* END_IGNORE_FOR_LINE_COUNT */

};

template<typename EltType>
DistributedArray<EltType>
DistributedArray<EltType>::create(std::string name, int64_t totalNumElements) {
  int myRank = 0;
  int numRanks = 0;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &numRanks);

  int64_t eltsPerRank = divCeil(totalNumElements, numRanks);
  int64_t eltsHere = eltsPerRank;
  if (eltsPerRank*myRank + eltsHere > totalNumElements) {
    eltsHere = totalNumElements - eltsPerRank*myRank;
  }
  if (eltsHere < 0) eltsHere = 0;

  DistributedArray<EltType> ret;
  ret.name_ = std::move(name);
  ret.localPart_.resize(eltsPerRank);
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
  MPI_Barrier(MPI_COMM_WORLD);

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
        std::cout << name_ << "[" << glbIdx << "] = " << localPart_[i] << "\n";
      }
      if (i < numElementsHere_) {
        std::cout << "...\n";
      }
      flushOutput();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
/* END_IGNORE_FOR_LINE_COUNT */

// compute the bucket for a value when sort is on digit 'd'
inline int getBucket(SortElement x, int d) {
  return (x.key >> (RADIX*d)) & MASK;
}

// rearranges data according values at 'digit' & returns count information
//   A: contains the input data
//   B: after it runs, contains the rearranged data
//   starts: should not be used after it runs
//   counts: after it runs, contains the number of elements for each bucket
//   digit: the current digit for shuffling
void localShuffle(std::vector<SortElement>& A,
                  std::vector<SortElement>& B,
                  counts_array_t& starts,
                  counts_array_t& counts,
                  int digit,
                  int64_t n) {
  assert(A.size() == B.size());

  // clear out starts and counts
  starts.fill(0);
  counts.fill(0);

  // compute the count for each digit
  for (int64_t i = 0; i < n; i++) {
    SortElement elt = A[i];
    counts[getBucket(elt, digit)] += 1;
  }

  // compute the starts with an exclusive scan
  {
    int64_t sum = 0;
    for (int i = 0; i < COUNTS_SIZE; i++) {
      starts[i] = sum;
      sum += counts[i];
    }
  }

  // shuffle the data
  for (int64_t i = 0; i < n; i++) {
    SortElement elt = A[i];
    int64_t &next = starts[getBucket(elt, digit)];
    B[next] = elt;
    next += 1;
  }
}

// helper for MPI_Alltoallv that:
//   * assumes elements in 'sendbuf' are sorted according to destination rank
//   * 'sendcounts[dstRank]' indicates how many elements to send to dst rank
//     and the total of sendcounts should be the same as sendbuf in size.
//     Since 'sendbuf' is sorted according to destination rank, these
//     elements must contiguous in sendBuf.
//   * automatically computes send displacements, recv counts, and recv
//     displacements
template<typename T>
void myMpiAllToAllV(const std::vector<T>& sendBuf,
                    const std::vector<int>& sendCounts,
                    std::vector<T>& recvBuf,
                    MPI_Datatype dt) {
  int myRank = 0;
  int numRanks = 0;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &numRanks);

  assert(sendCounts.size() == numRanks);

  std::vector<int> recvCounts;
  recvCounts.resize(numRanks);

  // Communicate the send counts to the destination nodes
  // where they will form the receive counts
  std::vector<int> tmpDspl, tmpCounts;
  tmpDspl.resize(numRanks);
  tmpCounts.resize(numRanks);
  for (int i = 0; i < numRanks; i++) {
    tmpDspl[i] = i;
    tmpCounts[i] = 1;
  }
  MPI_Alltoallv(/*sendbuf*/ &sendCounts[0],
                /*sendcounts*/ &tmpCounts[0],
                /*sdispls*/ &tmpDspl[0],
                MPI_INT,
                /*recvbuf*/ &recvCounts[0],
                /*recvcounts*/ &tmpCounts[0],
                /*rdispls*/ &tmpDspl[0],
                MPI_INT,
                MPI_COMM_WORLD);

  // compute the send displacements
  std::vector<int> sendDisplacements;
  sendDisplacements.resize(numRanks);
  {
    int sum = 0;
    for (int i = 0; i < numRanks; i++) {
      sendDisplacements[i] = sum;
      sum += sendCounts[i];
    }
    assert(sum == sendBuf.size());
  }

  // compute the recv displacements
  std::vector<int> recvDisplacements;
  recvDisplacements.resize(numRanks);
  {
    int sum = 0;
    for (int i = 0; i < numRanks; i++) {
      recvDisplacements[i] = sum;
      sum += recvCounts[i];
    }
    assert(sum == recvBuf.size());
  }

  // now do the MPI_Alltoallv to transfer the data
  MPI_Alltoallv(/*sendbuf*/ &sendBuf[0],
                /*sendcounts*/ &sendCounts[0],
                /*sdispls*/ &sendDisplacements[0],
                dt,
                /*recvbuf*/ &recvBuf[0],
                /*recvcounts*/ &recvCounts[0],
                /*rdispls*/ &recvDisplacements[0],
                dt,
                MPI_COMM_WORLD);
}

void copyCountsToGlobalCounts(counts_array_t& localCounts,
                              DistributedArray<int64_t>& GlobalCounts) {
  int myRank = 0;
  int numRanks = 0;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &numRanks);

  // create some buffers to help compute the transposition

  // how many to send to each destination rank?
  std::vector<int> sendCounts;
  sendCounts.resize(numRanks, 0);

  // what to send
  std::vector<CountBufElt> sendBuf;
  sendBuf.resize(COUNTS_SIZE);

  // space for receiving
  std::vector<CountBufElt> recvBuf;
  recvBuf.resize(COUNTS_SIZE);

  // fill in sendBuf and count the number for each destination
  for (int64_t i = 0; i < COUNTS_SIZE; i++) {
    int64_t dstGlobalIdx = i*numRanks + myRank;
    auto dst = GlobalCounts.globalIdxToLocalIdx(dstGlobalIdx);
    assert(0 <= dst.rank && dst.rank < numRanks);
    sendCounts[dst.rank]++;
    CountBufElt c;
    c.digit = i;
    c.rank = myRank;
    c.count = localCounts[i];
    sendBuf[i] = c;
  }

  // use myMpiAllToAllV to communicate the elements in sendBuf to recvBuf.
  myMpiAllToAllV(sendBuf, sendCounts, recvBuf, gCountBufEltMpiType);

  // now recvBuf contains the count data from other nodes,
  // but it isn't yet sorted correctly.
  // Sort it according to digit, breaking ties by rank.
  std::sort(recvBuf.begin(), recvBuf.end(),
            [](const CountBufElt& a, const CountBufElt& b) {
                if (a.digit != b.digit) {
                  return a.digit < b.digit;
                }
                return a.rank < b.rank;
            });

  // now store from recvBuf into GlobalCounts
  for (int64_t i = 0; i < COUNTS_SIZE; i++) {
    CountBufElt elt = recvBuf[i];
    int64_t globalIdx = elt.digit * numRanks + elt.rank;
    auto p = GlobalCounts.globalIdxToLocalIdx(globalIdx);
    assert(p.rank == myRank);
    GlobalCounts.localPart()[p.locIdx] = elt.count;
  }
}

void exclusiveScan(const DistributedArray<int64_t>& Src,
                   DistributedArray<int64_t>& Dst) {
  int myRank = 0;
  int numRanks = 0;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &numRanks);

  // Now compute the total of for each chunk of the global src array
  int64_t myTotal = 0;
  for (int64_t i = 0; i < Src.numElementsHere(); i++) {
    myTotal += Src.localPart()[i];
  }

  // Now use MPI_Scan to add up the totals from each rank
  int64_t myGlobalStart = 0;
  assert(sizeof(int64_t) == sizeof(long long));
  MPI_Exscan(&myTotal, &myGlobalStart, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  if (myRank == 0) {
    myGlobalStart = 0; // MPI_Exscan leaves result on rank 0 undefined
  }

  {
    int64_t sum = myGlobalStart;
    for (int64_t i = 0; i < Dst.numElementsHere(); i++) {
      Dst.localPart()[i] = sum;
      sum += Src.localPart()[i];
    }
  }
}

void copyStartsFromGlobalStarts(DistributedArray<int64_t>& GlobalStarts,
                                counts_array_t& localStarts) {
  int myRank = 0;
  int numRanks = 0;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &numRanks);

  // create some buffers to help compute the transposition

  // how many to send to each destination rank?
  std::vector<int> sendCounts;
  sendCounts.resize(numRanks, 0);

  // what to send
  std::vector<CountBufElt> sendBuf;
  sendBuf.resize(COUNTS_SIZE);

  // space for receiving
  std::vector<CountBufElt> recvBuf;
  recvBuf.resize(COUNTS_SIZE);

  // fill in sendBuf and count the number for each destination
  for (int64_t i = 0; i < GlobalStarts.numElementsHere(); i++) {
    int64_t globalIdx = GlobalStarts.localIdxToGlobalIdx(i);
    int64_t digit = globalIdx / numRanks;
    int64_t rank = globalIdx - digit*numRanks;
    assert(0 <= digit && digit < COUNTS_SIZE);
    assert(0 <= rank && rank < numRanks);
    sendCounts[rank]++;
    CountBufElt c;
    c.digit = digit;
    c.rank = rank;
    c.count = GlobalStarts.localPart()[i];
    sendBuf[i] = c;
  }

  // sort sendBuf according to destination rank and then by digit
  std::sort(sendBuf.begin(), sendBuf.end(),
            [](const CountBufElt& a, const CountBufElt& b) {
                if (a.rank != b.rank) {
                  return a.rank < b.rank;
                }
                return a.digit < b.digit;
            });

  // use myMpiAllToAllV to communicate the elements in sendBuf to recvBuf.
  myMpiAllToAllV(sendBuf, sendCounts, recvBuf, gCountBufEltMpiType);

  // now recvBuf contains the count data from other nodes,
  // but it isn't yet sorted correctly.
  // Sort it according to digit
  std::sort(recvBuf.begin(), recvBuf.end(),
            [](const CountBufElt& a, const CountBufElt& b) {
                return a.digit < b.digit;
            });

  // now store from recvBuf into localCounts
  for (int64_t i = 0; i < COUNTS_SIZE; i++) {
    CountBufElt elt = recvBuf[i];
    assert(elt.rank == myRank);
    assert(elt.digit == i);
    localStarts[i] = elt.count;
  }
}

void globalShuffle(DistributedArray<SortElement>& A,
                   DistributedArray<SortElement>& B,
                   int digit) {
  int myRank = 0;
  int numRanks = 0;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &numRanks);

  auto starts = std::make_unique<counts_array_t>();
  auto counts = std::make_unique<counts_array_t>();

  // Shuffle the data from A into B
  // the data in B will be sorted by the current digit
  localShuffle(A.localPart(), B.localPart(), *starts, *counts, digit,
               A.numElementsHere());

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
  copyCountsToGlobalCounts(*counts, GlobalCounts);

  // scan to fill in GlobalStarts
  exclusiveScan(GlobalCounts, GlobalStarts);

  // copy the per-bucket starts from the global counts array
  copyStartsFromGlobalStarts(GlobalStarts, *starts);

  // Now go through the data in B assigning each element its final
  // position and sending that data to the other ranks
  // Leave the result in A
  {
    // how many to send to each destination rank?
    std::vector<int> sendCounts;
    sendCounts.resize(numRanks, 0);

    int64_t numHere = B.numElementsHere();

    // what to send
    std::vector<ShuffleBufSortElement> sendBuf;
    sendBuf.resize(numHere);

    // space for receiving
    std::vector<ShuffleBufSortElement> recvBuf;
    recvBuf.resize(numHere);

    // fill in sendBuf and count the number for each destination
    for (int64_t i = 0; i < numHere; i++) {
      SortElement elt = B.localPart()[i];
      int bucket = getBucket(elt, digit);
      // what will be the destination index?
      int64_t &next = (*starts)[bucket];
      int64_t dstGlobalIdx = next;
      next += 1;
      auto dst = B.globalIdxToLocalIdx(dstGlobalIdx);
      sendCounts[dst.rank]++;
      ShuffleBufSortElement e;
      e.key = elt.key;
      e.val = elt.val;
      e.dstGlobalIdx = dstGlobalIdx;
      sendBuf[i] = e;
    }

    // use myMpiAllToAllV to communicate the elements in sendBuf to recvBuf.
    myMpiAllToAllV(sendBuf, sendCounts, recvBuf, gShuffleBufSortElementMpiType);

    // Now recvBuf contains the elements from other nodes,
    // but these aren't yet in the correct locations.
    // Store them into A according to dstGlobalIdx
    for (int64_t i = 0; i < numHere; i++) {
      ShuffleBufSortElement e = recvBuf[i];
      auto dst = B.globalIdxToLocalIdx(e.dstGlobalIdx);
      assert(dst.rank == myRank);
      SortElement& elt = A.localPart()[dst.locIdx];
      elt.key = e.key;
      elt.val = e.val;
    }
  }
}

// Sort the data in A, using B as scratch space.
void mySort(DistributedArray<SortElement>& A,
            DistributedArray<SortElement>& B) {
  for (int digit = 0; digit < N_DIGITS; digit++) {
    globalShuffle(A, B, digit);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // read in the problem size

  /* BEGIN_IGNORE_FOR_LINE_COUNT (printing and verification code) */
  bool printSome = false;
  bool verifyLocally = false;
  bool verifyLocallySet = false;
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
      verifyLocally = true;
      verifyLocallySet = true;
    } else if (std::string(argv[i]) == "--no-verify") {
      verifyLocally = false;
      verifyLocallySet = true;
    /* END_IGNORE_FOR_LINE_COUNT */
    }
  }

  /* BEGIN_IGNORE_FOR_LINE_COUNT (printing and verification code) */
  if (!verifyLocallySet) {
    verifyLocally = (n < 128*1024*1024);
  }
  /* END_IGNORE_FOR_LINE_COUNT */

  int myRank = 0;
  int numRanks = 0;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &numRanks);

  if (myRank == 0) {
    std::cout << "Total number of MPI ranks: " << numRanks << "\n";
    std::cout << "Problem size: " << n << "\n";
    /* BEGIN_IGNORE_FOR_LINE_COUNT (printing) */
    flushOutput();
    /* END_IGNORE_FOR_LINE_COUNT */
  }

  /* BEGIN_IGNORE_FOR_LINE_COUNT (error checking) */
  if ((n+numRanks-1) / numRanks > (int64_t) INT_MAX) {
    if (myRank == 0) {
      std::cout << "Too much data per rank (" << n/numRanks
                << ") in this configuration\n";
      std::cout << "int counts and offsets in MPI_AllToAllv could overflow\n";
    }
    MPI_Finalize();
    return 0;
  }
  /* END_IGNORE_FOR_LINE_COUNT */

  // setup the global types
  assert(sizeof(unsigned long long) == sizeof(uint64_t));
  assert(2*sizeof(unsigned long long) == sizeof(SortElement));
  MPI_Type_contiguous(2, MPI_UNSIGNED_LONG_LONG, &gSortElementMpiType);
  MPI_Type_commit(&gSortElementMpiType);
  assert(2*sizeof(unsigned long long) == sizeof(CountBufElt));
  MPI_Type_contiguous(2, MPI_UNSIGNED_LONG_LONG, &gCountBufEltMpiType);
  MPI_Type_commit(&gCountBufEltMpiType);
  assert(3*sizeof(unsigned long long) == sizeof(ShuffleBufSortElement));
  MPI_Type_contiguous(3, MPI_UNSIGNED_LONG_LONG, &gShuffleBufSortElementMpiType);
  MPI_Type_commit(&gShuffleBufSortElementMpiType);


  // create distributed arrays A and B
  auto A = DistributedArray<SortElement>::create("A", n);
  auto B = DistributedArray<SortElement>::create("B", n);
  std::vector<SortElement> LocalInputCopy;

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
    int64_t i = 0;
    for (auto& elt : A.localPart()) {
      elt.key = rng();
      elt.val = A.localIdxToGlobalIdx(i);
      i++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (myRank == 0) {
      std::cout << "Generated random values in " << elapsed.count() << " s\n";
      /* BEGIN_IGNORE_FOR_LINE_COUNT (printing) */
      flushOutput();
      /* END_IGNORE_FOR_LINE_COUNT */
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  /* BEGIN_IGNORE_FOR_LINE_COUNT (printing and verification code) */

  // Print out the first few elements on each locale
  if (printSome) {
    A.print(10);
  }

  if (verifyLocally) {
    // save the input to the sorting problem
    LocalInputCopy.resize(numRanks*A.numElementsPerRank());
    MPI_Gather(& A.localPart()[0], A.numElementsPerRank(), gSortElementMpiType,
               & LocalInputCopy[0], A.numElementsPerRank(),
               gSortElementMpiType, 0, MPI_COMM_WORLD);
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

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::steady_clock::now();

    mySort(A, B);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (myRank == 0) {
      std::cout << "Sorted " << n << " values in " << elapsed.count() << "\n";;
      std::cout << "That's " << n/elapsed.count()/1000.0/1000.0
                << " M elements sorted / s\n";
      flushOutput();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  /* BEGIN_IGNORE_FOR_LINE_COUNT (printing and verification code) */

  // Print out the first few elements on each locale
  if (printSome) {
    A.print(10);
  }

  if (verifyLocally) {
    if (myRank == 0) {
      std::cout << "Verifying\n";
    }
    // gather the output from sorting
    std::vector<SortElement> LocalOutputCopy;
    LocalOutputCopy.resize(numRanks*A.numElementsPerRank());
    MPI_Gather(& A.localPart()[0], A.numElementsPerRank(), gSortElementMpiType,
               & LocalOutputCopy[0], A.numElementsPerRank(),
               gSortElementMpiType, 0, MPI_COMM_WORLD);

    if (myRank == 0) {
      auto LocalSorted = LocalInputCopy;
      std::stable_sort(LocalSorted.begin(), LocalSorted.begin() + n,
                       [](const SortElement& a, const SortElement& b) {
                           return a.key < b.key;
                       });

      bool failures = false;
      for (int64_t i = 0; i < n; i++) {
        if (! (LocalSorted[i] == LocalOutputCopy[i])) {
          std::cout << "Sorted element " << i << " did not match\n";
          std::cout << "Expected: " << LocalSorted[i] << "\n";
          std::cout << "Got:      " << LocalOutputCopy[i] << "\n";
          failures = true;
        }
      }
      assert(!failures);
    }
  }
  /* END_IGNORE_FOR_LINE_COUNT */

  MPI_Finalize();
  return 0;
}
