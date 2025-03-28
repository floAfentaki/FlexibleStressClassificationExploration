# Healthcare_PE

## feature selection training 
python3 flokesh.py

> change dataset paths 

## generate verilog and obtain results for 4-bits
/gen_dt_synproj.sh DT wesad disr float_models 4

./run_synth.sh DT wesad disr float_models_4

./run_csv.sh DT wesad disr float_models_4

> check run_stupido.sh

python run_get_csvs.py