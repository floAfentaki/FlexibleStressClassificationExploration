#!/bin/bash

./gen_mlp_synprojTEST.sh TEST_KOSTAS TEST fisher_score float_model $1
cd "TEST_KOSTAS/TEST/fisher_score/float_model_${1}_synth/5"
make all

echo ""
echo ""
echo "verilog results"
cat reports/reports.txt
echo ""
echo ""
echo "python results"
cat py.rpt