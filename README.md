# NAMD-EPC

This is an Cython implementation of  [NAMD in Momentum Space (NAMD_k)](https://github.com/ZhenfaZheng/NAMDinMomentumSpace). Using Cython with MPI4py, H5py and Intel MKL libraries, most of the basic functions can be effectively achieved just like the original code. Some algorithm optimizations and error corrections are made to generate more reliable simulation results.

## Before Running NAMD_k

To use this implementation, prepare Intel MKL library and C compiler with MPI. Please prepare the Python >= 3.9 interpreter. Install the following Python packages required:

* Cython
* NumPy
* H5py
* MPI4py >= 3.1.3

## Run NAMD_k

1. Set parameters in `inp` and `INICON` in rundir(./).
2. Use `make c` to generate C code files.
   Use `make so` to generate dynamic-link libraries from C code files.
   Use `make exe` to gnerate namd-epc target file.
   Or instead, use `make` or `make all` to finish the above three processes.
3. Run `mpirun -np ncore namd-epc` or `sbatch sub_namd`.

Before perform preprocessing and NAMD simulations, some parameters need to be specify in `inp`. We list all the parameters needing to be customized.

```python
# NAMD parameter in Args.py
EPMDIR   = '../namdepc/h5files'
EPMPREF  = 'graphene'
NPARTS   = 9
nqx      = 90
nqy      = 90
nqz      = 1

namddir  = 'output'  # output NAMD output file in namddir
POTIM    = 1.0       # MD time step (fs)
SIGMA    = 0.025

LHOLE    = False     # Hole/electron transfer

TEMP     = 300.0     # temperature in Kelvin
NSAMPLE  = 1         # INICON sample number
NSW      = 50        # time step for NAMD run
NELM     = 100       # electron time step (per fs)
NTRAJ    = 2000      # SH trajectories number

EMIN     = -5.0   # band minimum energy (eV)
EMAX     = 2.0    # band maximum energy (eV)
PHCUT    = 1.0    # phonon minimum energy (meV)
LTRANS   = 'L'    # epc symmetrization:
                  # 'U': use upper triangle
                  # 'L': use lower triangle
                  # 'S': use both by (epc+epc.T.conj())/2
```

For comparation, the NAMD `inp` file is also listed here.

```fortran
&NAMDPARA
  EMIN       = -5
  EMAX       = 2
  NBANDS     = 2
  NQX        = 90
  NQY        = 90
  NQZ        = 1

  NSW        = 100
  POTIM      = 1.0
  TEMP       = 300.0

  NSAMPLE    = 1
  NELM       = 100
  NTRAJ      = 2000
  LHOLE      = .F.

  NPARTS     = 9
  SIGMA      = 0.025
  EPMDIR     = '../namdepc/h5files'
  EPMPREF    = 'graphene'
  NAMDDIR    = 'output'
  LTRANS     = 'L'
/
```

## After Running NAMD_k

After job finishes, `cp namdplt.py postnamd.py NAMDDIR`. Use `python namdplt.py` to plot.
The output files are numpy array binary files. We list all the output files and their corresponding output files in original NAMD_k implementation.

```
epc-*.npy(nb,nb): EPTXT
epcec-*.npy(nb,nb): EPECTXT
epcph-*.npy(nm,nb): EPPHTXT
psi-*.npy(nt,nb): PSICT[:,2:nb+2]
fssh_e_psi-*.npy(nt): PSICT[:,1]
fssh_e_sh-*.npy(nt): SHPROP[:,1]
fssh_pop_sh-*.npy(nt,nb): SHPROP[:,2:nb+2]
fssh_e_ph-*.npy(nt,nm): PHPROP.*[:,1]
fssh_pop_ph-*.npy(nt,nm,nb): PHPROP.*[:,2:nb+2]
```

## Notes

1. This implementation might be not efficient enough in large CPU core number.
2. Range of k points should be the subset of range of q points  in all HDF5 files, which means that `nk<=nq`.
3. Whole 3D q points should be used with row-major order, which means that `nqx*nqy*nqz==nq` and `for iqx in range(nqx): for for iqy in range(nqy): for iqz in range(nqz)` should be used.
