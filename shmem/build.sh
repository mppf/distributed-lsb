#!/bin/bash


if [ -z "$CC" ]
then
  if command -v cc
  then
    CC=cc
  else
    if command -v oshcc
    then
      CC=oshcc
    fi
  fi
fi

if [ -z "$CC" ]
then
  echo "Could not determine CC. Please run with CC=... ./build.sh"
fi

if [ -z "$CXX" ]
then
  if command -v CC
  then
    CXX=CC
  else
    if command -v oshc++
    then
      CXX=oshc++
    fi
  fi
fi

if [ -z "$CXX" ]
then
  echo "Could not determine CXX. Please run with CXX=... ./build.sh"
fi

echo Using C compiler $CC and C++ compiler $CXX

if [ ! -d pcg-cpp ]
then
  git clone https://github.com/imneme/pcg-cpp.git
fi

if [ ! -d bale ]
then
  git clone https://github.com/jdevinney/bale.git
  pushd bale/src/bale_classic/
  ./bootstrap.sh
  # ac_cv_search_shmemx_team_alltoallv avoids an incompatability with
  # newer cray-shmemx

  # TODO: this needs adjustment when not on EX
  PLATFORM=cray nice python3 make_bale --shmem --config_opts "CC=$CC ac_cv_search_shmemx_team_alltoallv=no " -j
  popd

  pushd bale/src/bale_classic/build_cray/convey/
  chmod +x ../../convey/tune_tensor
  if [ ! -f tune.dat ]; then
    make tune LAUNCH="srun --nodes=64 --ntasks-per-node=128 --hint=nomultithread"
  fi
  make
  popd
fi

export PKG_CONFIG_PATH=$PWD/bale/src/bale_classic/build_cray/lib/pkgconfig:$PKG_CONFIG_PATH

for name in `basename --suffix=.cpp *.cpp`
do
echo $CXX -DNDEBUG -O3 $name.cpp -o $name -I pcg-cpp/include/ $(pkg-config --cflags --libs convey)
$CXX -DNDEBUG -O3 $name.cpp -o $name -I pcg-cpp/include/ $(pkg-config --cflags --libs convey)
done
