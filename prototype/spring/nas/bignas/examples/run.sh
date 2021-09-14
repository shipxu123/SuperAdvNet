T=`date +%m%d%H%M`
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=1 \
python main.py 2>&1 | tee log.train.$T