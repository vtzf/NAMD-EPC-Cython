# Electron–Phonon Self-Energy Post-processing

## Workflow

This suite calculates the electronic self-energy `Sigma_nk` along a selected k-point path using the merged EPC matrix produced by the current DeepEPC.

1. `k_list.py` generates the uniform k and q grids from `postprocessing.ini`.
2. `gen_qlist.py` maps `epcDir/kpath.npy` and the q grid to full-grid indices.
3. `self_energy.py` assigns k-path points to MPI ranks. Each rank reads only its own k-path and q blocks from the merged EPC file, computes `abs` on each block, and accumulates the real and imaginary self-energy contributions. No `g_final` file is generated or loaded.
4. `plot_sigma.py` writes `Sigma_nk.png`, with one curve per selected band.

## Configuration

Only `postprocessing.ini` is read; `config.ini` is not required. In `[main]`, set `nq`, `nmodes`, `nbands`, the data directories and file names. `EpcType=K` with `IsAllKlist=True` is required. For `Ispin=1`, `spin=up` or `spin=dn` selects the corresponding file suffix; other spin modes use no suffix.

The electronic grid `Sigma.nk` must be an integer multiple of the phonon grid `main.nq`. `Ef`, `T`, `bmin/bmax`, `delta`, `ph_cutoff`, and other analysis parameters remain in this file. Band indices refer to the bands stored in `valname`.

## Inputs

- `epcDir/epcname` (with the selected spin suffix): merged K-path EPC data.
- `epcDir/kpath.npy`: selected k-point path.
- `bandDir/valname` (with spin suffix): electronic energies on the full k grid.
- `phononDir/phvalname`: phonon frequencies.

## Outputs

- `Sigma_nk`: energies relative to the Fermi level and the real and imaginary self-energy contributions in meV. The real part is `Re[Sigma(epsilon_nk)-Sigma(Ef)]`.
- `Sigma_nk.png`: self-energy plot.

## Run

Configure `postprocessing.ini` and `job.sh` for your environment, then use `sbatch job.sh`, or run the Python steps directly. The supplied job script does not delete EPC files or reduced results.

The original workflow was contributed by Jiale Chen and used for the data production of Materials Today Physics 67, 102192 (2026), https://doi.org/10.1016/j.mtphys.2026.102192.
