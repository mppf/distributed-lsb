// This is a sort benchmark for comparison that uses
// sort functions from KaDiS 
// https://github.com/MichaelAxtmann/KaDiS
// Of particular interest is the AMS sort.
// To use this file, replace the contents of example/sorting.cpp
// in the KaDiS repository with this file, and then compile it with cmake.
// The AMS sort can be benchmarked with --sorter 1, for example:
//   srun --exclusive --nodes 64 --ntasks-per-node=128 ./kadisexample --sorter 1 --no-verify --n 68719476736

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

#include <mpi.h>
#include <pcg_random.hpp>

#include "AmsSort/AmsSort.hpp"
#include "RQuick/RQuick.hpp"
#include "RFis/RFis.hpp"
#include "Bitonic/Bitonic.hpp"
#include "HSS/Hss.hpp"

#undef NDEBUG

// Note: this code assumes that errors returned by MPI calls do not need to be
// handled. That's the case if they are set to halt the program or
// raise exeptions.

// the elements to sort
struct SortElement {
  uint64_t key = 0; // to sort by
  uint64_t val = 0; // carried along
};
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
bool operator==(const SortElement& x, const SortElement& y) {
  return x.key == y.key && x.val == y.val;
}
bool operator<(const SortElement& x, const SortElement& y) {
  return x.key < y.key;
}

static MPI_Datatype gSortElementMpiType;

// to help in communicating the counts
struct CountBufElt {
  int32_t digit = 0;
  int32_t rank = 0;
  int64_t count = 0;
};
std::ostream& operator<<(std::ostream& os, const CountBufElt& x) {
  os << "(" << x.digit << "," << x.rank << "," << x.count << ")";
  return os;
}
static MPI_Datatype gCountBufEltMpiType;

// to help in communicating the elements
struct ShuffleBufSortElement {
  uint64_t key = 0;
  uint64_t val = 0;
  int64_t  dstGlobalIdx = 0;
};
std::ostream& operator<<(std::ostream& os, const ShuffleBufSortElement& x) {
  os << "(";
  printhex(os, x.key);
  os << "," << x.val << "," << x.dstGlobalIdx << ")";
  return os;
}
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

  // helper to print part of the distributed array
  void print(int64_t nToPrintPerRank) const;
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

std::mt19937_64 createGen(int rank) {
  std::mt19937_64 gen;
  int data_seed = 3469931 + rank;
  gen.seed(data_seed);
  return gen;
}

// Sort the data in A, using B as scratch space.
void mySort(DistributedArray<SortElement>& A,
            DistributedArray<SortElement>& B,
	    int sorter) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, size;
  const int tag = 12345;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Create RBC communicator.
  RBC::Comm rcomm;
  // Does use RBC collectives and RBC sub-communicators internally.
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  auto gen = createGen(rank);
  auto myless = std::less<SortElement>(); 
  std::vector<SortElement>& data = A.localPart();
  assert(data.size() == A.numElementsHere());
  assert(data.size() == A.numElementsPerRank());

  if (data.size() != A.numElementsPerRank() || data.size() != A.numElementsHere()) {
    std::cout << "quitting due to uneven array\n"; flushOutput();
    exit(1);
  }

  if (sorter == 0) {
    if (rank == 0) { std::cout << "RQuick\n"; flushOutput(); }
    // Sort data descending with RQuick.
    RQuick::sort(gSortElementMpiType, data, tag, gen, comm, myless);
    // Seeing correctness issues with this one.
  } else if (sorter == 1) {
    if (rank == 0) { std::cout << "AMS\n"; flushOutput(); }
    const int num_levels = 2;
    Ams::sortLevel(gSortElementMpiType, data, num_levels, gen, rcomm, myless);
  } else if (sorter == 2) {
    if (rank == 0) { std::cout << "Bitonic\n"; flushOutput(); }
    const bool is_equal_input_size = false;
    Bitonic::Sort(data, gSortElementMpiType, tag, rcomm, myless, is_equal_input_size);
  } else if (sorter == 3) {
    if (rank == 0) { std::cout << "RFis\n"; flushOutput(); }
    RFis::Sort(gSortElementMpiType, data, rcomm);
    // Seeing correctness issues with this one.
  } else if (sorter == 4) {
    if (rank == 0) { std::cout << "Hss\n"; flushOutput(); }
    const int num_levels = 2;
    Hss::sortLevel(gSortElementMpiType, data, num_levels, gen, rcomm);
    // Seeing correctness issues with this one.
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // read in the problem size
  bool printSome = false;
  bool verifyLocally = false;
  bool verifyLocallySet = false;
  int sorter = 0;
  int64_t n = 100*1000*1000;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--n") {
      n = std::stoll(argv[++i]);
    } else if (std::string(argv[i]) == "--sorter") {
      sorter = std::stoll(argv[++i]);
    } else if (std::string(argv[i]) == "--print") {
      printSome = true;
    } else if (std::string(argv[i]) == "--verify") {
      verifyLocally = true;
      verifyLocallySet = true;
    } else if (std::string(argv[i]) == "--no-verify") {
      verifyLocally = false;
      verifyLocallySet = true;
    }
  }

  if (!verifyLocallySet) {
    verifyLocally = (n < 128*1024*1024);
  }

  int myRank = 0;
  int numRanks = 0;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &numRanks);

  // force n to be a multiple of numRanks
  int64_t orig_n = n;
  {
    int64_t nper = (n + numRanks - 1) / numRanks;
    n = numRanks * nper; 
  }

  if (myRank == 0) {
    std::cout << "Total number of MPI ranks: " << numRanks << "\n";
    std::cout << "Problem size: " << n << " (rounded up from " << orig_n << ")\n";
    flushOutput();
  }

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
      flushOutput();
    }

    auto rng = pcg64(myRank);
    int64_t i = 0;
    for (auto& elt : A.localPart()) {
      if (i < A.numElementsHere()) { 
        elt.key = rng();
        elt.val = A.localIdxToGlobalIdx(i);
      } else {
        elt.key = 0;
        elt.val = -1;
      }
      i++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (myRank == 0) {
      std::cout << "Generated random values in " << elapsed.count() << " s\n";
      flushOutput();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

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

  // Shuffle the data in-place to sort by the current digit
  {
    if (myRank == 0) {
      std::cout << "Sorting\n";
      flushOutput();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::steady_clock::now();

    mySort(A, B, sorter);

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

  // Print out the first few elements on each locale
  if (printSome) {
    A.print(10);
  }

  if (verifyLocally) {
    // gather the output from sorting
    std::vector<SortElement> LocalOutputCopy;
    LocalOutputCopy.resize(numRanks*A.numElementsPerRank());
    MPI_Gather(& A.localPart()[0], A.numElementsPerRank(), gSortElementMpiType,
               & LocalOutputCopy[0], A.numElementsPerRank(),
               gSortElementMpiType, 0, MPI_COMM_WORLD);

    if (myRank == 0) {
      auto LocalSorted = LocalInputCopy;
      std::stable_sort(LocalSorted.begin(), LocalSorted.begin() + n,
                       std::less<SortElement>());

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

  MPI_Finalize();
  return 0;
}
