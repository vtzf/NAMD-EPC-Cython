#!/bin/bash
#SBATCH --partition       sugon
#SBATCH -J namd-tyy
#SBATCH --time            30:00:00
#SBATCH --nodes           4
#SBATCH --ntasks-per-node 16
#SBATCH --cpus-per-task   1
#SBATCH --error           namd_%j.err
#SBATCH --output          namd_%j.log

ulimit -s unlimited

export HDF5_USE_FILE_LOCKING="FALSE"
export OMP_NUM_THREADS=1

echo "============================================================"
module list
env | grep "MKLROOT="
echo "============================================================"
echo "Job ID: $SLURM_JOB_NAME"
echo "Job name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of processors: $SLURM_NTASKS"
echo "Task is running on the following nodes:"
echo $SLURM_JOB_NODELIST
echo "OMP_NUM_THREADS = $SLURM_CPUS_PER_TASK"
echo "============================================================"
echo

srun --mpi=pmi2 namd-epc
