#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\tbash hw4_test.sh [testing data] [prediction file]"
  exit
fi

embedding=140.112.90.197:10297/hw4/embedding.zip
model=140.112.90.197:10297/hw4/model.zip

wget "${embedding}" -O ./temp.zip
unzip temp.zip
rm temp.zip

wget "${model}" -O ./temp.zip
unzip temp.zip
rm temp.zip

python3 hw4_test.py $1 $2
