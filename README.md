# scientific-python-showcase
Concise demos for numerical methods, simulation, and data analysis in Python.

## What This Repo Demonstrates
- Numerical integration (leapfrog), simulation workflows, and basic validation checks
- Scientific Python stack (NumPy, Matplotlib, SciPy)
- Working with generated datasets (.npz, .txt) and reproducible runs (seeds)
- Writing scripts that are runnable end-to-end with clear inputs/outputs

## Setup
- Install Poetry (dependency management): https://python-poetry.org/docs/
- Install dependencies:
  - `poetry install`

## Run Commands
All scripts are run from the repo root with `PYTHONPATH=.`:

- Orbit integration:
  - `PYTHONPATH=. poetry run python scripts/orbits/numerisk_bane_integrasjon.py`
- Orbit plots (uses orbit output):
  - `PYTHONPATH=. poetry run python scripts/orbits/numerisk_bane_plot.py`
- Atmosphere model:
  - `PYTHONPATH=. poetry run python scripts/atmosphere/atmosphere_model.py`
- Trilateration (uses orbit output):
  - `PYTHONPATH=. poetry run python scripts/estimation/trilateration.py`
- Spectral data generation (600–1000 nm):
  - `PYTHONPATH=. poetry run python scripts/spectral/spectral_data.py`
- Spectral analysis (uses spectral outputs):
  - `PYTHONPATH=. poetry run python scripts/spectral/spectral_analysis.py`

## Scripts and Outputs
- `scripts/orbits/numerisk_bane_integrasjon.py`
  - Integrates planetary orbits with leapfrog.
  - Outputs: `scripts/orbits/outputs/orbits.npz`
- `scripts/orbits/numerisk_bane_plot.py`
  - Plots analytic vs simulated orbits, distance vs time, and Kepler checks.
  - Reads: `scripts/orbits/outputs/orbits.npz`
  - Outputs: interactive figures (no files saved by default).
- `scripts/atmosphere/atmosphere_model.py`
  - Computes a two-layer (adiabatic + isothermal) atmosphere profile.
  - Outputs: `scripts/atmosphere/outputs/atmosphere_data.npz`
- `scripts/estimation/trilateration.py`
  - Estimates spacecraft position by trilateration using planet positions.
  - Reads: `scripts/orbits/outputs/orbits.npz`
- `scripts/spectral/spectral_data.py`
  - Generates synthetic spectrum + noise data for 600–1000 nm.
  - Outputs:
    - `scripts/spectral/outputs/spectrum_600nm_1000nm.txt`
    - `scripts/spectral/outputs/sigma_noise.txt`
- `scripts/spectral/spectral_analysis.py`
  - Performs line fitting for O2 and H2O on the synthetic spectrum.
  - Reads:
    - `scripts/spectral/outputs/spectrum_600nm_1000nm.txt`
    - `scripts/spectral/outputs/sigma_noise.txt`
  - Outputs: plots + `scripts/spectral/outputs/line_search_results.csv`

## Reproducibility
Scripts that generate data use fixed seeds in their `main()` functions. Re-run
the same script to get identical outputs. If you change parameters, regenerate
dependent outputs.

## Troubleshooting (PYTHONPATH)
If you see `ModuleNotFoundError: No module named 'lib'`, you are likely missing
`PYTHONPATH=.`. Run scripts exactly as shown above.

## Plots (optional)
Example outputs live under the `outputs/` folders next to each script. After running,
you should see files like:
- `scripts/orbits/outputs/orbits.npz`
- `scripts/spectral/outputs/line_search_results.csv` and associated PNGs

## Note on AI Assistance
This repository is based on my own coursework and implementations. I wrote the core
numerical methods, analysis logic, and the scripts themselves. I also used ChatGPT as
a productivity tool during refactoring and repository setup (e.g., restructuring folders,
improving script entry points, clarifying imports, and drafting parts of documentation).
I reviewed and tested changes locally to ensure correctness.
