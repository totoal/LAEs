#!/bin/bash

declare -i N=5
declare -i N_max=23

for i in {1..2}; do
  echo -e "\nROUND $i\n"
  for j in {1..14}; do
    /home/alberto/cosmos/LAEs/train_test_prep.py $N&
    echo "nb = $N"
    if [ "$N" -eq "$N_max" ]; then
      break
    fi
    N=$((N + 1))
  done
  wait
done 2>/dev/null