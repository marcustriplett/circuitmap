#!/bin/sh

set -e
export PATH="/home/mat2245/anaconda3/bin:$PATH"

# userhome="/home/ubuntu"
# datastore="circuitmap/data"
# # outstore="ncapdata/localout"

echo "---- DOWNLOADING DATA ----"


conda init bash
conda activate jax

# neurocaas-contrib workflow get-data -f -o $userhome/$datastore/
# neurocaas-contrib workflow get-config -f -o $userhome/$datastore/

echo "---- PARSING PATHS ----"
datapath=$(neurocaas-contrib workflow get-datapath)
configpath=$(neurocaas-contrib workflow get-configpath)
resultpath=$(neurocaas-contrib workflow get-resultpath-tmp)

cd "~/circuit_mapping/circuitmap/"

echo "---- LAUNCHING CIRCUITMAP ----"
python "scripts/run_circuitmap_main.py" --data $datapath --config $configpath --out $resultpath

# echo "---- WRITING RESULTS ----"
# neurocaas-contrib workflow put-result -r $resultpath/