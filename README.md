# NAMD-EPC

This is an Cython implementation of  [NAMD in Momentum Space (NAMD_k)](https://github.com/ZhenfaZheng/NAMDinMomentumSpace). Using Cython with MPI4py, H5py and Intel MKL libraries, most of the basic functions can be effectively achieved just like the original code. Some algorithm optimizations and error corrections are made to generate more reliable simulation results.

New version from DeepEPC branch interfaced with upcoming DeepEPC code is aimed to compute electron-phonon coupling (EPC) matrix elements using the numeric atom-centered orbital (NAO) basis. Simple multiple-electron method has also been added. Some integer overflow bugs are fixed to perform large-scale simulation. Some new options are also provided for small-scale and memory-friendly simulation.

## Before Running NAMD_k

To use this implementation, prepare Intel MKL library and C compiler with MPI. Please prepare the Python >= 3.9 interpreter. Install the following Python packages required:

* Cython
* NumPy
* HDF5 library (can be h5py or independent library, now need to specify the path)
* MPI4py >= 3.1.3

## Run NAMD_k

1. Set parameters in `inp` and `INICON` in rundir(./).
2. Use `make c` to generate C code files.
   Use `make so` to generate dynamic-link libraries from C code files.
   Use `make exe` to generate namd-epc target file `namd-epc`.
   Or instead, use `make` or `make all` to finish the above three processes.
   `makefile` checks the path of `Python.h`, `numpy/*.h` and `hdf5.h` header files from Python installation directory.
3. Run `mpirun -np ncore namd-epc` or `sbatch sub_namd`.

Before performing preprocessing and NAMD simulations, some parameters need to be specified in `inp`. We list all the parameters needing to be customized. An example of NAMD `inp` file is listed here.

Some new tags are added in new version under DeepEPC branch. 

```fortran
&NAMDPARA
  EMIN       = -5
  EMAX       = 2
** NBANDS     = 2 now removed, can be read from EPC **
  NQX        = 90
  NQY        = 90
  NQZ        = 1

  NSW        = 100
  POTIM      = 1.0
  TEMP       = 300.0
  LHDF5      = .T.
** Use .T. to interface with Perturbo **
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

  BANDDEG    = 1
** Band degeneracy (still some bugs), just set to 1 **
  LEPCSHM    = .F.
** If EPC matrix are smaller than memory of one node, **
** use .T. to speed up surface hopping calculation. **
  LPHSHM     = .T.
** For nk_s << nq, use .T. to store only one phonon data **
** on each node, which can save memory. **
  LSPLIT     = .F.
** If .T., the program will end after time propagation. **
** You need to manually resubmit job for suface hopping. **
** Useful when calculating different SurfHop with same TimeProp. **
  NM_BLOCK   = 6
** [1,nmodes], smaller number can save more memory when symmetrizing EPC**
/
```

## After Running NAMD_k

After job finishes, `cp namdplt.py postnamd.py NAMDDIR`. Use `python namdplt.py` to plot.
The output files are numpy array binary files. We list all the output files and their corresponding output files in original NAMD_k implementation.

```
inp
bassel.npy(nk,2): BASSEL
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
2. k list and q list should be on the same grid with (nqx,nqy,nqz) shape.
3. Range of k list should be the subset of range of q list in all HDF5 files, which means that `nk<=nq`.
