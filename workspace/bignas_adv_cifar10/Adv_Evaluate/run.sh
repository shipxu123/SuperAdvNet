#!/bin/bash
T=`date +%m%d%H%M`

ROOT=/mnt/lustre/xupeng2/xupeng/codes/prototype/
export PYTHONPATH=$ROOT:$PYTHONPATH
#FACE=~/xupeng/codebase/pytorch-pod/plugins/face
#export PODPLUGINPATH=$FACE

#PARTITION=$1
NUM_GPU=$1
CFG=./config3.yaml
if [ -z $3 ];then
    NAME=default
else
    NAME=$3
fi

g=$(($NUM_GPU<8?$NUM_GPU:8))
#srun --mpi=pmi2 -p $PARTITION --comment=spring-submit -n$NUM_GPU --gres=gpu:$g --ntasks-per-node=$g --cpus-per-task=5 \
#    --job-name=$NAME \
#--phase evaluate_subnet
spring.submit run -n$g --gpu --job-name=--job-name=$NAME \
"python3 -u -m prototype.solver.bignas_cifar10_adv_solver \
  --config=$CFG \
  --phase evaluate_subnet \ 
  2>&1 | tee log.train.$NAME.$T"
