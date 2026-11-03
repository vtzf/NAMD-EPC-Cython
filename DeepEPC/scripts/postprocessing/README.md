# DeepEPC Post-processing

## Contributor

The original post-processing workflows were developed and contributed by **Jiale Chen** and used for the data production of *Materials Today Physics* **67**, 102192 (2026), https://doi.org/10.1016/j.mtphys.2026.102192.

Affiliations:

- Key Laboratory of Polar Materials and Devices, Ministry of Education, East China Normal University
- NYU-ECNU Institute of Physics, NYU Shanghai
- School of Information and Electronic Engineering, East China Normal University

Contact: **jc12994@nyu.edu**

## Quick start

Prepare the merged EPC file, electronic bands, phonon frequencies, bassel states, and K/Q path files produced by DeepEPC. The self-energy workflow requires `EpcType=K`, and the coupling workflow requires `EpcType=Q`, both with `IsAllKlist=True`.

Install NumPy, mpi4py, ASE, and matplotlib in an MPI-enabled Python environment. Edit `postprocessing.ini` in the relevant subdirectory: set `nq`, `nmodes`, `nbands`, the data directories, EPC file names, and analysis parameters. `config.ini` is not read.

For `Ispin=1`, use `spin=up` or `spin=dn` to select the corresponding file suffix; other spin modes use no suffix. Check the `Nef` and spin-weight conventions for the system being studied.

### K-path self-energy

```bash
cd self_energy
# Edit postprocessing.ini first.
python k_list.py
python gen_qlist.py
mpirun -np N python self_energy.py
python plot_sigma.py
```

This produces `Sigma_nk` and `Sigma_nk.png`. `Sigma_nk` contains `Re[Sigma(epsilon_nk)-Sigma(Ef)]` and `Im[Sigma(epsilon_nk)]` in meV.

### Q-path coupling and spectral function

```bash
cd lambda
# Edit postprocessing.ini first.
python gen_qlist.py
mpirun -np N python lambda.py
python ../a2f/a2f.py
python ../a2f/plota.py
```

This produces `lambda_qnu_{nq[0]}`, `a2f.dat`, and `a2f.png`. `a2f.py` also prints total `lambda`, `omega_log`, and the Allen–Dynes `Tc`.

`job.sh` can be used instead of the manual commands after adapting its partition, MPI environment, and rank count to the local cluster.

## Workflows

### K-path self-energy (`self_energy/`)

1. `k_list.py` generates `k_all_list` and `q_all_list` from `postprocessing.ini`.
2. `gen_qlist.py` maps `epcDir/kpath.npy` and the q grid to `k_list` and `q_list`.
3. `self_energy.py` distributes k-path points over MPI ranks. Each rank reads only its own k-path and q blocks from the merged EPC file, computes `abs` on each block, and accumulates the real and imaginary self-energy contributions.
4. `plot_sigma.py` writes `Sigma_nk.png`, with one curve per selected band.

### Q-path coupling (`lambda/`)

1. `gen_qlist.py` generates `k_all_list` and `q_list`. It uses `eph.qpt` when present; otherwise it reads `epcDir/qpath.npy` and assigns weight 1 to every q point. Supply `eph.qpt` with a fourth weight column for custom q weights.
2. `lambda.py` distributes q points over MPI ranks. Each rank reads only its own q blocks and selected bassel states, computes `abs` on each block, and accumulates `lambda_qnu` and total `lambda`.

### Spectral function (`a2f/`)

1. `a2f.py` reads `lambda_qnu_{nq[0]}` and `lambda/postprocessing.ini`, then calculates total `lambda`, `alpha2F(omega)`, `omega_log`, and `Tc`. It writes `a2f.dat` and prints the summary.
2. `plota.py` reads `a2f.dat` and writes `a2f.png` with `alpha2F(omega)` and accumulated `lambda(omega)`.

The self-energy and coupling workflows are independent. The spectral-function workflow depends only on the lambda output.

## Configuration

The `[Sigma]` section contains the electronic grid, temperature, target bands, and lifetime broadening. The electronic grid `Sigma.nk` must be an integer multiple of `main.nq`. The `[lambda]` section contains the coupling smearing, DOS at the Fermi level, and Fermi-window width. The `[a2f]` section contains the spectral smearing, Coulomb pseudopotential, and number of frequency points.

## Grid input without generator scripts

The generator scripts can be skipped when equivalent grid files are supplied directly:

- `k_all_list`: three columns `kx ky kz`, in full-grid order.
- `q_list`: five columns `qx qy qz weight q_index`, with a zero-based full-grid index.
- `k_list` (self-energy only): five columns `kx ky kz weight k_index`, also with a zero-based full-grid index.

`q_all_list` is only needed by `gen_qlist.py`; `self_energy.py` itself uses `k_all_list`, `q_list`, and `k_list`. `lambda.py` uses `k_all_list` and `q_list`.

## Data safety

`job.sh` is a scheduler template, not part of the numerical implementation. It runs only the required grid preparation and MPI calculation steps. No `g_reduce_k.py` or `g_final` file is required, and the supplied job scripts do not delete the original EPC data or reduced results.
