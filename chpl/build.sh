#!/bin/bash

for name in *.chpl
do
  echo chpl --fast $name
  chpl --fast $name
done
