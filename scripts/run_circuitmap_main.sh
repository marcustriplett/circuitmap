#!/bin/bash

set -e
export PATH="/home/mat2245/anaconda3/bin:$PATH"
source activate pytorch_p38

userhome="/home/ubuntu/"
# datastore="circuitmap/data"
# # outstore="ncapdata/localout"

echo "---- DOWNLOADING DATA ----"

neurocaas-contrib workflow get-data
neurocaas-contrib workflow get-config

echo "---- PARSING PATHS ----"
datapath=$(neurocaas-contrib workflow get-datapath)
configpath=$(neurocaas-contrib workflow get-configpath)
resultpath=$(neurocaas-contrib workflow get-resultpath-tmp)

echo "---- LAUNCHING CIRCUITMAP ----"
cd $userhome/circuitmap
python ./scripts/run_circuitmap_main.py --data $datapath --config $configpath --out $resultpath

# echo "---- WRITING RESULTS ----"
# neurocaas-contrib workflow put-result -r $resultpath/