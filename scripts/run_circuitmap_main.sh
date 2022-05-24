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
source deactivate

echo "---- LAUNCHING CIRCUITMAP ----"
cd $userhome/circuitmap
source activate pytorch_p38
python ./scripts/run_circuitmap_main.py --data $datapath --config $configpath --out $resultpath
zip -r out.zip $resultpath/*
source deactivate

echo "---- UPLOADING RESULTS ----"
source activate neurocaas
neurocaas-contrib workflow put-result -r out.zip