#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\tbash hw4_train.sh [train label data] [train unlabel data]"
  exit
fi

python3 hw4_train.py $1 $2
