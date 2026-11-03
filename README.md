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
   `SHM_MAX` sets the per-node FSSH row-cache budget in bytes (default: 32212254720, 30 GiB). It is used when `LEPCSHM = .F.`; `LEPCSHM = .T.` stores the full EPC matrix in shared memory instead. The budget includes cache metadata and other shared arrays. Override it at build time, e.g. `make SHM_MAX=... so`.
3. Build NAMD-k working directory. Generate `inp` and `INICON` under this folder.  
   If EPC matrix is output by DeepEPC (`LHDF5 = .F.` in `inp`), please copy `config.ini` to here. Set `inDir` to the DeepEPC data directory relative to this folder, and check `epcDir` and `epcname`.  
   If `Ispin = 1`, add `_up` or `_dn` before the file extensions of `valname`, `vecname`, `basselname` and `epcname` to select a spin component. For example (spin-up):
   ```ini
   inDir = ../7_no_so
   valname = band_val_k-10_up.npy
   vecname = band_vec_k-10_up.npy
   basselname = bassel-10_up.npy
   epcDir = epc_all
   epcname = epc_all_p-10_up.dat
   ```
   When `IsAllKlist = False`, `_p` is added to `valname` and `vecname` automatically by NAMD-k.
5. Run `mpirun -np ncore namd-epc` or `sbatch sub_namd`.

Before performing preprocessing and NAMD simulations, some parameters need to be specified in `inp`. We list all the parameters needing to be customized. An example of NAMD `inp` file is listed here.

Some new tags are added in new version under DeepEPC branch. 

```python
&NAMDPARA
  EMIN       = -5
  EMAX       = 2
# NBANDS     = 2 now removed, can be read from EPC
  NQX        = 90
  NQY        = 90
  NQZ        = 1

  NSW        = 100
  POTIM      = 1.0
  TEMP       = 300.0
  SIGMA      = 0.025

  NSAMPLE    = 1
  NELM       = 100
  NTRAJ      = 2000
  LHOLE      = .F.

  LHDF5      = .T.         # .T.: interface with Perturbo
                           # .F.: interface with DeepEPC
                           # if .F., NPARTS, EPMDIR and EPMPREF are ignored
  NPARTS     = 9
  EPMDIR     = '../namdepc/h5files'
  EPMPREF    = 'graphene'
  NAMDDIR    = 'output'
  LTRANS     = 'L'         # 'U': build the upper triangle from the original lower triangle,
                           #      with conjugate completion
                           # 'L': build the lower triangle from the original upper triangle
                           # 'S': build both triangles by Hermitian symmetrization

  BANDDEG    = 1           # band degeneracy (still some bugs), just set to 1
  LEPCSHM    = .F.         # .T.: store the full EPC matrix in shared memory
                           # .F.: use the bounded EPC row cache controlled by SHM_MAX
  LPHSHM     = .T.         # for nk_s << nq, you can use .T. to store only 
                           # one phonon data on each node to save memory
  LSPLIT     = .F.         # if .T., the program will end after time propagation
                           # you need to manually resubmit job for suface hopping
                           # useful when calculating different SurfHop with same TimeProp
  NM_BLOCK   = 6           # [1,nmodes], smaller number can save more memory when symmetrizing EPC
/
```

Each row of `INICON` defines one sample. The first value is the initial phonon time step used to sample the initial phonon phases for ensemble averaging. It is followed by `(ik, ib)` pairs specifying the initial carrier occupations (1-based). Each `(ik-1, ib-1)` pair must be present in the `bassel` basis list. The number of rows must be at least `NSAMPLE`.

## After Running NAMD_k

After job finishes, `cp namdplt.py postnamd.py NAMDDIR` if `LHDF5 = .T.`. Also `cp config.ini NAMDDIR` if `LHDF5 = .F.`. Use `python namdplt.py` to plot.
The output files are numpy array binary files. We list all the output files and their corresponding output files in original NAMD_k implementation.

```
inp
bassel.npy(nk,2): BASSEL
epc-*.npy(nb,nb): EPTXT
epcec-*.npy(nb,nb): EPECTXT
epcph-*.npy(nm,nb): EPPHTXT
psi-*.dat: NSW frames of [nel,nk_a] complex128 coefficients, appended frame by frame
fssh_e_psi-*.npy(NSW,nel): PSICT[:,1]
fssh_e_sh-*.npy(nt): SHPROP[:,1]
fssh_pop_sh-*.npy(nt,nb): SHPROP[:,2:nb+2]
fssh_e_ph-*.npy(nt,nm): PHPROP.*[:,1]
fssh_pop_ph-*.npy(nt,nm,nb): PHPROP.*[:,2:nb+2]
```

## Notes

1. This implementation might be not efficient enough in large CPU core number.
2. k list and q list should be on the same grid with (nqx,nqy,nqz) shape.
3. Range of k list should be the subset of range of q list in all HDF5 files, which means that `nk<=nq`.
