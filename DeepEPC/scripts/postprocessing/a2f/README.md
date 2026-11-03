# Spectral Function Post-processing

`a2f.py` reads `../lambda/postprocessing.ini` and `../lambda/lambda_qnu_{nq[0]}`, then writes `a2f.dat` containing 500 frequency points by default. It also prints total `lambda`, logarithmic-average phonon frequency, and the Allen–Dynes estimate of `Tc`.

Set `sigma`, `mu`, and `nw` in the `[a2f]` section of `../lambda/postprocessing.ini`. Run:

```bash
python a2f.py
python plota.py
```

`plota.py` writes `a2f.png` with the spectral function and accumulated `lambda(omega)`.
