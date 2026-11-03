# Electron–Phonon Coupling Post-processing

## Workflow

This suite calculates mode-resolved electron–phonon coupling constants `lambda_qnu` and total `lambda`.

1. `gen_qlist.py` generates the uniform k grid and the q-point/index mapping. It reads `eph.qpt` when present, otherwise it reads `epcDir/qpath.npy` and assigns weight 1 to each q point. Supply `eph.qpt` with a fourth weight column when nonuniform weights are required.
2. `lambda.py` assigns q points to MPI ranks. Each rank reads only its own q blocks and the initial states selected by `basselname`, computes `abs` on each block, and accumulates `lambda_qnu` and total `lambda`. No `g_final` file is generated or loaded.
3. `a2f/a2f.py` and `a2f/plota.py` use this output to generate `a2f.dat` and `a2f.png`.

## Configuration

Only `postprocessing.ini` is read; `config.ini` is not required. Set `nq`, `nmodes`, `nbands`, the data directories and file names in `[main]`. `EpcType=Q` with `IsAllKlist=True` is required. For `Ispin=1`, `spin=up` or `spin=dn` selects the corresponding file suffix; other spin modes use no suffix. Check the spin convention for `Nef` and the factor 2 used in `lambda.py` for the system being studied.

The `[lambda]` section contains the Gaussian smearing, DOS at the Fermi level, and Fermi-window width. The `[a2f]` section contains the spectral smearing, Coulomb pseudopotential, and number of frequency points.

## Inputs

- `epcDir/epcname` (with the selected spin suffix): merged Q-path EPC data.
- `epcDir/qpath.npy`, or `eph.qpt` when custom q weights are needed.
- `bandDir/basselname` (with spin suffix): selected initial states.
- `bandDir/valname` (with spin suffix): full-grid and selected-state band energies; the `_p.npy` file is derived automatically.
- `phononDir/phvalname`: phonon frequencies.

## Outputs

- `lambda_qnu_{nq[0]}`: mode-resolved coupling constants and total `lambda`.
- `a2f.dat`, `a2f.png`: spectral function and accumulated coupling.

## Run

Configure `postprocessing.ini` and `job.sh` for your environment, then use `sbatch job.sh`. Run `python ../a2f/a2f.py` and `python ../a2f/plota.py` to obtain the spectral function. The supplied job script does not delete EPC files or reduced results.

The original workflow was contributed by Jiale Chen and used for the data production of Materials Today Physics 67, 102192 (2026), https://doi.org/10.1016/j.mtphys.2026.102192.
