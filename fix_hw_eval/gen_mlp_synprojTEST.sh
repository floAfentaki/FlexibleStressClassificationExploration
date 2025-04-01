#!/bin/bash

if [ "$#" -ne 5 ]; then
       echo "expecting model dataset FSmethod optimization bitwidth"
       exit
fi


base_folder="$1/$2/$3"
models_folder=$base_folder/$4
synth_folder=$base_folder/"${4}_${5}_synth"

for i in $models_folder/*.joblib
do
  # new_filename=$(echo "$i" | sed 's/\./_/')
  # mv "$i" "$new_filename"
  # i=$new_filename
  number=$(basename "$i" | cut -d '.' -f 1)
  mkdir -p $synth_folder
  name="$synth_folder/$number"
  if [ -d "$name" ]; then
    rm -rf $name
  fi

  cp -r 0_template_pragmatic $name
	cp csv.sh $synth_folder


    python syn_projectTEST.py $2 $models_folder $number $i $5| tee ${2}py.rpt

  mv ${2}py.rpt $name/py.rpt

  regclf="clf"
  for f in "sim.Xtrain" "sim.Xtest" "sim.Ytest" "top_tb.v"
  do
  	cp $2$f $name/sim/$f 
  done

  cp ${2}exact.v ${name}/hdl/top.v
  mv ${2}exact.v ${name}/hdl/top.v.base

  for f in "${2}sim.Xtrain" "${2}sim.Xtest" "${2}sim.Ytest" "${2}top_tb.v"
  do
  	rm $f
  done

done

