#!/bin/bash
cnt=0
# parent_dir="./MLP/wesad/fisher_score/synth"
parent_dir="./$1/$2/$3/${4}_synth"
for dataset_dir in "$parent_dir"/*/
do  
    dirs=$(echo "$dataset_dir" | cut -d '/' -f 2,3,4)
    dirs=$(echo "$dirs" | tr '/' '_')
    number=$(basename "$dataset_dir")
    if [ ! -f "$dataset_dir/reports/reports.txt" ]; then
        cnt=$((cnt + 1))
        cp 0_template_pragmatic/scripts/clean.sh "$dataset_dir/scripts"
        screenName="${dirs}_${4}_${number}_Synthesis"
        screen -S "$screenName" -d -m bash
        # screen -S "$screenName" -X stuff "cd $dataset_dir && make clean $(echo -ne '\015')" 
        screen -S "$screenName" -X stuff "cd $dataset_dir &&make all$(echo -ne '\015')" 


        # if [ ! -f "$dataset_dir/gate/top.sv" ]; then
        #     # echo $dataset_dir needs Synthesis
        #     screenName="${dirs}_${4}_${number}_Synthesis"
        #     cnt=$((cnt + 1))
        #     # echo $screenName
        #     # screen -S "$screenName" -d -m bash
        #     # screen -S "$screenName" -X stuff "cd $dataset_dir && make all$(echo -ne '\015')" 
        # else 
        #     # echo $dataset_dir needs Gate Simulation and Power Calculation
        #     screenName="${dirs}_${4}_${number}_Gate"
        #     # echo $screenName
        #     # cnt=$((cnt + 1))
        #     # cp 0_template_pragmatic/scripts/clean.sh "$dataset_dir/scripts"
        #     # screen -S "$screenName" -d -m bash
        #     # screen -S "$screenName" -X stuff "cd $dataset_dir && make gate_sim&& make power&&make results$(echo -ne '\015')" 
        #     # screen -S "$screenName" -X stuff "cd $dataset_dir && make clean &&make all$(echo -ne '\015')" 
        # fi
    fi     
    # cnt=$((cnt + 1))

done
echo $cnt