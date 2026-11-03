#!/bin/bash
#SBATCH --job-name="deepEPC-sigma"
#SBATCH --output="%j.%N.out"
#SBATCH --partition=debug,parallel
#SBATCH --nodes=1
#SBATCH -n 16
#SBATCH --export=ALL
#SBATCH -t 4-00:00:00

set -e
ulimit -s unlimited
ulimit -l unlimited
export OMP_NUM_THREADS=1

# Load the MPI and Python environment required by your cluster.
touch RUNNING
python k_list.py
python gen_qlist.py
mpirun -np 15 python self_energy.py
mv RUNNING done
