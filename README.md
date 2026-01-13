# scientific-python-showcase
Concise demos for numerical methods, simulation, and data analysis in Python.

## Context: AST2000
This repository is adapted from work done in the University of Oslo course **AST2000**.  
I reorganized and cleaned up the original work into standalone scripts with reproducible outputs so the methods can be reviewed outside the course framework.

## What this repo demonstrates
- Numerical integration (leapfrog), simulation workflows, and basic validation checks  
- Scientific Python stack (NumPy, Matplotlib, SciPy)  
- Working with generated datasets (`.npz`, `.txt`) and reproducible runs (fixed seeds)  
- Writing scripts that are runnable end-to-end with clear inputs/outputs  

## Repository layout
- `lib/` – shared helpers used by scripts (imported via `PYTHONPATH=.`)
- `scripts/`
  - `orbits/` – orbit integration + plotting
  - `atmosphere/` – simple atmosphere model
  - `estimation/` – trilateration using simulated ephemerides
  - `spectral/` – synthetic spectrum generation + line fitting
- `figures/` – optional curated plots for the README / CV
- `scripts/**/outputs/` – generated outputs (data + plots)

## Setup
- Install Poetry (dependency management): https://python-poetry.org/docs/
- Install dependencies:
  - `poetry install`

## Run commands
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

## Scripts and outputs
- `scripts/orbits/numerisk_bane_integrasjon.py`
  - Integrates planetary orbits with leapfrog.
  - Outputs: `scripts/orbits/outputs/orbits.npz`

- `scripts/orbits/numerisk_bane_plot.py`
  - Plots analytic vs simulated orbits, distance vs time, and Kepler checks.
  - Reads: `scripts/orbits/outputs/orbits.npz`
  - Outputs: interactive figures (no files saved by default).

- `scripts/atmosphere/atmosphere_model.py`
  - Computes a two-layer (adiabatic + isothermal) atmosphere profile.
  - Outputs: `scripts/atmosphere/outputs/atmosphere_data.npz` (if enabled in the script)

- `scripts/estimation/trilateration.py`
  - Estimates a spacecraft position by trilateration using planet positions.
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
dependent outputs (e.g. regenerate orbits before rerunning trilateration).

## Troubleshooting (PYTHONPATH)
If you see `ModuleNotFoundError: No module named 'lib'`, you are likely missing
`PYTHONPATH=.`. Run scripts exactly as shown above.

## Outputs
Outputs live under the `outputs/` folders next to each script. After running,
you should see files like:
- `scripts/orbits/outputs/orbits.npz`
- `scripts/spectral/outputs/line_search_results.csv` and associated PNGs

## Example plots

### Spectral analysis

<img width="697" height="400" alt="Orbit plot 1" src="https://github.com/user-attachments/assets/73f06065-0f18-4d23-a93d-61657ab06345" />
<img width="700" height="448" alt="Orbit plot 2" src="https://github.com/user-attachments/assets/efe191b2-607b-4abe-8b8a-0e13baeb3760" />

### Orbital simulation
<img width="1392" height="812" alt="Spectral plot 1" src="https://github.com/user-attachments/assets/aac03a97-d0f0-45b2-ac9e-44c0d8683ce7" />
<img width="1384" height="796" alt="Spectral plot 2" src="https://github.com/user-attachments/assets/63dbc074-8493-462c-9ea4-94d69a4400f7" />

### Simple atmosphere model
<img width="744" height="588" alt="Atmosphere plot" src="https://github.com/user-attachments/assets/5577ba8b-4e91-4f69-a3dd-f1ec9afbcd74" />

## Note on AI assistance
This repository is based on my own coursework and implementations. I wrote the core
numerical methods, analysis logic, and the scripts themselves. I also used ChatGPT as
a productivity tool during refactoring and repository setup (e.g., restructuring folders,
improving script entry points, clarifying imports, and drafting parts of documentation).
I reviewed and tested changes locally to ensure correctness.
