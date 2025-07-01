#!/bin/bash


if [ -z "$CXX" ]
then
  if command -v CC
  then
    CXX=CC
  else
    if command -v mpic++
    then
      CXX=mpic++
    fi
  fi
fi

if [ -z "$CXX" ]
then
  echo "Could not determine CXX. Please run with CXX=... ./build.sh"
fi

echo Using compiler $CXX

if [ ! -d pcg-cpp ]
then
  git clone https://github.com/imneme/pcg-cpp.git
fi

for name in `basename --suffix=.cpp *.cpp`
do
echo $CXX -DNDEBUG -O3 $name.cpp -o $name -I pcg-cpp/include/
$CXX -DNDEBUG -O3 $name.cpp -o $name -I pcg-cpp/include/
done
