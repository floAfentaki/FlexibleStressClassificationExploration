#!/bin/bash

base_dir=$(pwd)
parent_dir="./$1/$2/$3/${4}_synth"
cp csv.sh $parent_dir
cd $parent_dir
chmod +x csv.sh
./csv.sh
cd $base_dir