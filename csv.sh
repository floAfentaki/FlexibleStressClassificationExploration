#!/bin/bash

file="csv.csv"
echo "Dataset;FeatureSelection;Number;Accuracy;Area;Power;" > $file

# for dataset_path in $1/*; do 
path=$(pwd)
for dataset_path in ./*; do 

    if [[ "$dataset_path" == "./csv.csv" || "$dataset_path" == "./csv.sh" ]]; then
        continue
    fi
    dataset="$(cut -d'/' -f10 <<<"$path")"
    feature_selection="$(cut -d'/' -f11 <<<"$path")"
    number="$(cut -d'/' -f2 <<<"$dataset_path")"

    i=$dataset_path
    accuracy=$(cat $i/reports/reports.txt | grep 'Accuracy: '|cut -d':' -f2|cut -d"A" -f1| awk '{printf "%.0f", $1}')
  
    area_file="$i/reports/top_200000.0ns.area.rpt"
    power_file="$i/reports/top_200000.0ns.power.ptpx.rpt"
    area=$( [ -f "$area_file" ] && grep 'Total cell area:' $area_file | awk '{printf "%.5f", $4 / 100000000}' || echo "N/A" )
    power=$( [ -f "$power_file" ] && grep 'Total Power' $power_file | awk '{print $4 * 1000}' || echo "N/A" )

    echo $dataset";"$feature_selection";"$number";"$accuracy";"$area";"$power";" >> $file
done
