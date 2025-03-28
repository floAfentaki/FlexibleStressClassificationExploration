#!/bin/bash

# parent_dir="./MLP/wesad/fisher_score/synth"
parent_dir="./$1/$2/$3/${4}_synth"
for dataset_dir in "$parent_dir"/*/
do  
    dirs=$(echo "$dataset_dir" | cut -d '/' -f 2,3,4)
    dirs=$(echo "$dirs" | tr '/' '_')
    number=$(basename "$dataset_dir")
    # echo "${dirs}_${number}"
    screenName="${dirs}_${4}_${number}"
    echo $screenName
    screen -S "$screenName" -d -m bash
    screen -S "$screenName" -X stuff "cd $dataset_dir && make all$(echo -ne '\015')"      
done
