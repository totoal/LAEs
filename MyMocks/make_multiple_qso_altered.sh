#!/bin/bash

declare -i N=1

for i in {1..2}; do
  echo -e "\nROUND $i\n"
  for j in {1..12}; do
    /home/alberto/cosmos/LAEs/MyMocks/Make_QSO_altered.py $N&
    N=$((N + 1))
  done
  wait
done 2>/dev/null