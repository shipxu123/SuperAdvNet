#!/bin/bash
T=`date +%m%d%H%M`

ROOT=~/xupeng/codes/super-adv-net1/prototype/
export PYTHONPATH=$ROOT:$PYTHONPATH

#PARTITION=$1
NUM_GPU=$1
CFG=./config.yaml
if [ -z $2 ];then
    NAME=default
else
    NAME=$2
fi

#g=$(($NUM_GPU<8?$NUM_GPU:8))
python3 -W ignore -u -m prototype.solver.bignas_cifar10_adv_solver \
  --config=$CFG \
  2>&1 | tee log.train.$NAME.$T
