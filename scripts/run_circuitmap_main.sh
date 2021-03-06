#!/bin/bash

# @author Marcus A. Triplett (2022), Columbia University

set -e
source activate neurocaas
userhome="/home/ubuntu/"

echo "---- DOWNLOADING DATA ----"
neurocaas-contrib workflow get-data
neurocaas-contrib workflow get-config

echo "---- PARSING PATHS ----"
datapath=$(neurocaas-contrib workflow get-datapath)
configpath=$(neurocaas-contrib workflow get-configpath)
resultpath=$(neurocaas-contrib workflow get-resultpath-tmp)
dataname=$(neurocaas-contrib workflow get-dataname)
savname="${dataname%.*}_cm"
source deactivate

echo "---- LAUNCHING CIRCUITMAP ----"
source activate pytorch_p38
python $userhome/circuitmap/scripts/run_circuitmap_main.py --data $datapath --config $configpath --out $resultpath
cd $resultpath
zip -r $savname.zip *
source deactivate

echo "---- UPLOADING RESULTS ----"
cd $userhome
source activate neurocaas
neurocaas-contrib workflow put-result -r $resultpath/$savname.zip