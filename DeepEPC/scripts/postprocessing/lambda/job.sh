#!/bin/bash
#SBATCH --job-name="deepEPC-lambda"
#SBATCH --output="%j.%N.out"
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH -n 64
#SBATCH --export=ALL
#SBATCH -t 4-00:00:00

set -e
ulimit -s unlimited
ulimit -l unlimited
export OMP_NUM_THREADS=1

# Load the MPI and Python environment required by your cluster.
touch RUNNING
python gen_qlist.py
mpirun -np 64 python lambda.py
mv RUNNING done
