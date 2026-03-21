# DeepEPC

DeepEPC framework in NAMD-k_DeepEPC uses finite-displacement method under numeric atomic orbital basis to calculate electron-coupling coupling (EPC) matrix elements. NAMD-EPC-Cython can read the EPC results to perform NAMD-k simulation. Now DeepEPC supports OpenMX (\*.scfout binary data) or DeepH (HDF5 data) to get finite-displacement perturbation Hamiltonian and band structure, together with OpenMX (\*.out) or DeePMD to get finite-displacement perturbation force, force constant and phonon dispersion.

## Get Input Parameters

Before performing preparing calculations before obtaining EPC matrix, some parameters need to be specified in `config.ini`. We list all the parameters needing to be customized and we try to expain their meanings afterwards.

```python
[epc]
# input files
Ispin = 1			# Ispin = 0: non-spin-polarized calculations
				# Ispin = 1: spin-polarized calculations (collinear)
				# Ispin = 2: spin-polarized calculations (non-collinear, SOC)
para_dic = {"Ni":"10.0 8.0  off"}	# OpenMX parameter in "Atoms.SpeciesAndCoordinates"
				# automatically added after corresponding atomic coordinates
ucellidx = [7,7,7]		# supercell size for finite-displacement phonon/EPC calculations
dQ = 0.01			# finite-displacement step size of phonon/EPC calculations
inDir = 7_no_so			# working directory of band/phonon/EPC calculations
poscar_ucell = POSCAR_u		# unitcell POSCAR file
infile_in_s = input_s.dat	# OpenMX supercell input file without atomic information of
				# "Atoms.SpeciesAndCoordinates" and "Atoms.UnitVectors"
infile_split_s = 21		# number of line to insert atomic information in supercell input file
infile_out = input.dat		# OpenMX unitcell/supercell output file with atomic information
				# automatically generated in "inDir" and "inDir/dhamilDir/*±*"
subfile_s = sub_openmx_s	# OpenMX supercell submit script

# band files
bandDir = epc_band		# band files store in "inDir/bandDir"
valname = band_val_k-10.npy	# band eigenvalue file name
vecname = band_vec_k-10.npy	# band eigenvector file name
basselname = bassel-10.npy	# selected basis file when IsAllKlist = False
spinname = spinDM-10.npy	# spin density matrix file when "Ispin = 2"

# phonon files
phononDir = phonon		# phonon files store in "inDir/phononDir"
ifcname = fc_avg.npy		# interatomic force constant file name
ifcRcut = 5.0			# cutoff radius of force constant
				# if cutoff radius is unnecessary, just leave it blank
phvalname = val-10.npy		# phonon eigenvalue file name
phvecname = vec-10.dat		# phonon eigenvector file name
phonon_method = F		# finite difference method in phonon calculation
				# F: forward difference; B: backward difference; C: central difference
# method: F B C

# epc calculation
dhamilDir = dhamil		# finite-displacemant Hamiltonian stores in "inDir/dhamilDir"
epcDir = epc_all		# EPC result stores in "inDir/epcDir"
epcname = epc_all_p-10.dat	# EPC matrix file name
IsH5 = False			# interface type of EPC calculation
				# True: DeepH HDF5 file; False: OpenMX scfout file
				# the following five "H5*" parameters are ignored if IsH5 = False
H5HamName = out/hamiltonians	# HDf5 supercell (perturbation) Hamiltonian files 
				# are named as "inDir/dhamilDir/*±*/H5HamName.H5"
H5OlpName = out/overlaps	# HDf5 supercell overlap files are named as "inDir/H5OlpName(_*).H5"
H5OlpNum = 1			# number of HDF5 files for supercell overlap matrix
				# if H5OlpNum = N > 1, each file is named as "H5OlpName_{0..N-1}.H5"
H5DrName = output/overlaps_d	# supercell momentum operator matrix files
				# are named as "inDir/H5DrName(_*).H5"
H5DrNum = 8			# number of HDF5 files for supercell momentum operator matrix
				# if H5DrNum = N > 1, each file is named as "H5DrName_{0..N-1}.H5"
nq = [10,10,10]			# 3D k/q grid size to output EPC data
atom = [1]			# number of each kind of atom, corresponding to poscar_ucell
orbital = [14]			# number of NAO in each kind of atom, corresponding to infile_in_u
EpcType = A			# EPC calculation type
				# A: IsAllKlist = True: all k-grid → all q-grid
				# A: IsAllKlist = False: selected k-grid → selected k-grid
				# K: Kpoint defined k-path → all q-grid
				# Q: all k-grid → Kpoint defined q-path
# type: A K Q
Kpoint = [[0,0,0],[0,1/2,0]]	# k/q-point of EpcType = K/Q calculation
				# k/q-path is obtained by interpolation under Kpoint and nq
IsAllVec = False			# True: get all energy bands in [bmin=0,bmax=dot(atom,orbital)-1]
				# False: get energy band with given index [bmin,bmax]
bmin = 8			# minimum energy band index starting from 0
bmax = 9			# maximum energy band index starting from 0
IsAllKlist = False		# True: get energy band of all k-grid (under given [bmin,bmax])
				# False: get energy band in [emin,emax] (under given [bmin,bmax])
emin = -6.21423134		# minimum band energy
emax = -5.21423134		# maximum band energy
dhamil_method = F		# finite difference method in EPC calculation, similar to phonon_method
# method: F B C

[ucell]
IsH5_u = False			# interface type of band calculation, similar to IsH5
H5HamName_u = out/hamiltonians	# HDf5 unitcell Hamiltonian files are named as
				# "inDir/dhamilDir/*±*/H5HamName_u.H5"
H5OlpName_u = output/overlaps	# HDf5 unitcell overlap files are named as "inDir/H5OlpName_u(_*).H5"
H5OlpNum_u = 4			# number of HDF5 files for unitcell overlap matrix
				# if H5OlpNum_u = N > 1, each file is named as "H5OlpName_u_{0..N-1}.H5"
infile_in_u = input_u.dat	# OpenMX unitcell input file without atomic information of
				# "Atoms.SpeciesAndCoordinates" and "Atoms.UnitVectors"
infile_split_u = 21		# number of line to insert atomic information in unitcell input file
subfile_u = sub_openmx_u	# OpenMX unitcell submit script

[sub]
SUB_NUM_HAMIL = 2		# number of jobs run simultaneously in sub_para.py script
TIME_MAX_HAMIL = 360		# maximum job running time in sub_para.py script
# second

[mpi]
DHAMIL_BLOCK = 4		# number of MPI processes used to read and store
				# each supercell Hamiltonian in epcsparse.py
				# 1 <= DHAMIL_BLOCK <= number of MPI processes in one nnode
NMODES_BLOCK = 3		# epcsparse: number of perturbation Hamiltonian
				# read in each calculation loop
				# epc_nl.py: number of phonon modes of EPC calculated in part2 calculation
				# 1 <= NMODES_BLOCK <= nmodes and smaller NMODES_BLOCK 
				# can save memory but may reduce computational efficiency
# scalapack parameter
M_BLOCK = 32			# scalapack row block size in epc_nl.py
N_BLOCK = 32			# scalapack column block size in epc_nl.py
				# here M_BLOCK = N_BLOCK is needed to avoid some bugs
				# norbital*ncell > M_BLOCK*nprocs/GCD(nprocs) is recommanded
```

## Perform preparing calculations with Python Scripts

There are some simple python scripts under `DeepEPC/scripts` folder. To run there scripts normally, please prepare the Python >= 3.9 interpreter. Install the following Python packages required:

* NumPy
* ASE
* H5py
* Cython
* MPI4py >= 3.1.3

Follow the above example to generate the config.ini file in the working directory. Build the `inDir` folder manually. Put five input files under `inDir`:
`poscar_ucell`, `infile_in_s`, `subfile_s`, `infile_in_u`, `subfile_u`

We will firstly show how to use the OpenMX interface of DeepEPC. Run the scripts in the following order

`python supercell.py`: Generate `inDir/infile_out` with fixed grid center.
`python makedir.py`: Build input folder structure.
`python sub_para.py`: Concurrently submit job tasks. When the number of tasks is small, you can manually submit by running `sbatch subfile`.
`mpirun -np N python epc_band_ucell_sparse(_nc).py`: Energy band calculation. If Ispin = 0 or 1, use epc_band_ucell_sparse.py, otherwise use epc_band_ucell_sparse_nc.py.
`python epc_fc_ase.py`: Interatomic force constant calculation.
`mpirun -np N python epc_phonon.py`: Phonon dispersion calculation.

You can also submit band/phonon tasks if computational cost is large.

## Perform EPC calculations

There are some simple Python/Cython scripts under `DeepEPC/src` folder. To run there scripts normally, copy them to the working directory. Run `make c` to interpret Cython files as C language files. Then run `make so` to compile C language files into dynamic link library (\*.so) files. The following libraries are required to finish compilation:

* HDF5 library >= 1.10.5 (H5PATH needs to be specified in makefile)
* mpiicc compiler & Intel MKL library

`mpirun -np N python epcsparse.py`: EPC part1 calculation.
`mpirun -np N python epc_nl.py`: EPC part2 calculation.
The EPC part1 calculation will firstly output result in `inDir/epcDir/epcname`, then EPC part2 calculation will add the rest result on the above outputting file.
Binary file `inDir/epcDir/epcname` has the following format:

`IsAllKlist = True`: (nk = nq = nq[0]\*nq[1]\*nq[2])
`EpcType = A`: [nk,nq,nmode,nband,nband]
`EpcType = K`: [nkpath,nq,nmode,nband,nband]
`EpcType = Q`: [nk,nqpath,nmode,nband,nband]

`IsAllKlist = False`: (nbasis = basselname.shape[0])
`EpcType = A`: [nbasis,nbasis,nmode]

You can also build submitting EPC job if computational cost is large.

## Data plotting

There are three simple python plotting scripts under `DeepEPC/scripts` folder: `band_plot.py`, `ph_plot.py` and `epc_plot.py`. Just refer to the `DeepEPC/example/Ni` for usage.

## DeepEPC interface with DeePMD/DeepH framework

You should also follow the above steps to build input folder structure. If the DeePMD deep potential model is generated, copy `graph.pb` file to the working directory. Run `epc_fc_dp.py` to obtain interatomic force constant. Run `epc_phonon.py` to obtain phonon dispersion.

If the DeepH deep learning Hamiltonian model is generated, you can adjust the `sub_para.py` script to perform overlap-only calculation (mpirun -np N openmx-dolp infile_out) in `inDir/ucell` and `inDir/dhamilDir/*±*`, then perform overlap-dr-only supercell calculation (mpirun -np N openmx-dolp infile_out -calcOLPmo) in `inDir`.

The overlap-dr-only OpenMX patch is in `DeepEPC/patch`. Just copy the two adjusted C source files in OpenMX `source` and compile with HDF5 library (similar as [overlap-only-OpenMX](https://github.com/mzjb/overlap-only-OpenMX)) to build the executable file `openmx-dolp`.

After overlap-(dr-)only calculation, run DeepH to obtain unitcell Hamiltonian and run `epc_band_ucell_sparse(_nc).py` to obtain energy band. Then run DeepH to obtain all the supercell perturbation Hamiltonian. Finally run `epcsparse.py` and `epc_nl.py` to finish EPC calculation.

## NAMD-k interface with DeepEPC

In the NAMD-EPC-Cython implementation, `inp` file simply adjusts a few parameters for interfacing with DeepEPC and obtaining higher computational efficiency on large-scale simulation. Please refer to `README.md` of NAMD-EPC-Cython for usage.



