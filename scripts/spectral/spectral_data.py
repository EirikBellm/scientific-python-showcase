"""
Synthetic spectral data generator.

Generates two files in scripts/spectral/outputs:
- spectrum_600nm_1000nm.txt (2 cols: wavelength_nm, flux)
- sigma_noise.txt           (2 cols: wavelength_nm, sigma)

Design goals
------------
- Flux is "normalized-ish": continuum close to 1 with a tiny ripple.
- Absorption lines are Gaussian dips with Doppler-thermal widths
  consistent with the fitting model in spectral_analysis.py:
      sigma_lambda = lambda0 * sqrt(k*T/m) / c
- A single global Doppler shift (radial velocity) is applied to all lines.
- Temperature is allowed to vary per line (toy proxy for different atmospheric layers).

Notes
-----
This is still a toy generator (not a full radiative transfer model). The main point is
physics consistency between generator and fitter so the curve fits look sensible.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS_DIR = HERE / "outputs"
LAM_MIN_NM = 600.0
LAM_MAX_NM = 1000.0


# Physical constants (SI)
C_LIGHT_M_PER_S = 299792458.0
K_B_J_PER_K = 1.380649e-23
M_P_KG = 1.67262192595e-27


def generate_sigma_noise(wavelengths_nm, seed, base_sigma=0.05):
    """
    Returns array with shape (N, 2): [wavelength_nm, sigma]

    sigma is a smooth-ish baseline around base_sigma plus a small random jitter.
    """
    rng = np.random.default_rng(seed + 1_000_000)
    wiggle = 0.003 * np.sin((wavelengths_nm - wavelengths_nm[0]) / 250.0)
    jitter = 0.001 * rng.normal(size=wavelengths_nm.size)
    sigma = np.clip(base_sigma + wiggle + jitter, 0.005, None)
    return np.column_stack([wavelengths_nm, sigma])


def doppler_sigma_nm(lambda0_nm, temperature_K, molecular_mass_kg):
    """
    Thermal Doppler width (Gaussian sigma) in nm, consistent with the fitter.

    sigma_lambda = lambda0 * sqrt(kT/m) / c
    """
    return lambda0_nm * np.sqrt(K_B_J_PER_K * temperature_K / molecular_mass_kg) / C_LIGHT_M_PER_S


def generate_spectrum(
    wavelengths_nm,
    seed,
    lines,
    sigma_noise,
    vr_m_per_s=None,
    vr_range=(-8000.0, 8000.0),
    temp_range=(100.0, 500.0),
    fmin_range=(0.65, 1.0),
):
    """
    Returns array with shape (N, 2): [wavelength_nm, flux]

    Creates a synthetic spectrum with:
    - continuum near 1.0 (+ small ripple)
    - Gaussian absorption lines with Doppler-thermal widths
    - additive Gaussian noise with per-point sigma

    Parameters
    ----------
    wavelengths_nm : (N,) ndarray
        Wavelength grid [nm].
    seed : int
        Random seed for reproducibility.
    lines : list[dict]
        Each dict must have:
          - "name": str (for readability)
          - "lambda0_nm": float
          - "mass_kg": float
    sigma_noise : (N,) ndarray
        Per-point noise sigma for flux.
    vr_m_per_s : float or None
        If provided, applies a single global Doppler shift to all lines.
        If None, sampled uniformly from vr_range.
    vr_range : tuple(float, float)
        Range for global Doppler shift if vr_m_per_s is None.
    temp_range : tuple(float, float)
        Range for per-line temperatures [K].
    fmin_range : tuple(float, float)
        Range for line minima fmin (0 < fmin <= 1).
    """
    rng = np.random.default_rng(seed)

    # Continuum baseline near 1.0 with tiny ripple.
    continuum = 1.0 + 0.002 * np.sin((wavelengths_nm - wavelengths_nm[0]) / 120.0)
    flux = continuum.copy()

    # Global Doppler shift for all lines.
    if vr_m_per_s is None:
        vr_m_per_s = float(rng.uniform(vr_range[0], vr_range[1]))
    shift_factor = vr_m_per_s / C_LIGHT_M_PER_S

    # Apply each absorption line.
    for line in lines:
        lam0 = float(line["lambda0_nm"])
        mass = float(line["mass_kg"])

        # Shift line center by the same global vr.
        lam_center = lam0 * (1.0 + shift_factor)

        # Temperature varies per line (toy "different layers").
        T_K = float(rng.uniform(temp_range[0], temp_range[1]))

        # Line depth parameter fmin (min flux at line center).
        fmin = float(rng.uniform(fmin_range[0], fmin_range[1]))

        # Doppler-thermal Gaussian width (sigma) in nm.
        sigma_nm = doppler_sigma_nm(lam_center, T_K, mass)

        # Safety: avoid sigma going to exactly 0 in weird cases.
        sigma_nm = max(sigma_nm, 1e-6)

        gauss = np.exp(-0.5 * ((wavelengths_nm - lam_center) / sigma_nm) ** 2)
        model = 1.0 - (1.0 - fmin) * gauss

        # Multiply absorptions so multiple lines can coexist.
        flux *= model

    # Add measurement noise (additive).
    noisy_flux = flux + rng.normal(scale=sigma_noise, size=flux.size)
    return np.column_stack([wavelengths_nm, noisy_flux])


def ensure_spectral_files(
    data_dir=None,
    seed=12345,
    n_points=2_000_000,
    lam_min=LAM_MIN_NM,
    lam_max=LAM_MAX_NM,
    base_sigma=0.05,
    force=False,
    vr_m_per_s=None,
):
    """
    Creates the two required files if they don't exist (or if force=True):
      - spectrum_600nm_1000nm.txt (2 cols: nm, flux)
      - sigma_noise.txt (2 cols: nm, sigma)

    Returns
    -------
    (spectrum_path, noise_path) : tuple[Path, Path]
    """
    data_dir = OUTPUTS_DIR if data_dir is None else Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = HERE / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    spectrum_path = data_dir / f"spectrum_{int(lam_min)}nm_{int(lam_max)}nm.txt"
    noise_path = data_dir / "sigma_noise.txt"

    if (not force) and spectrum_path.exists() and noise_path.exists():
        return spectrum_path, noise_path

    wavelengths_nm = np.linspace(lam_min, lam_max, int(n_points), dtype=float)

    # Candidate lines + molecular masses (kg).
    # These masses match what spectral_analysis.py uses (atomic mass * proton mass).
    proton = M_P_KG
    lines = [
        {"name": "O2_632", "lambda0_nm": 632.0, "mass_kg": 32.0 * proton},
        {"name": "O2_690", "lambda0_nm": 690.0, "mass_kg": 32.0 * proton},
        {"name": "O2_760", "lambda0_nm": 760.0, "mass_kg": 32.0 * proton},
        {"name": "H2O_720", "lambda0_nm": 720.0, "mass_kg": 18.0 * proton},
        {"name": "H2O_820", "lambda0_nm": 820.0, "mass_kg": 18.0 * proton},
        {"name": "H2O_940", "lambda0_nm": 940.0, "mass_kg": 18.0 * proton},
    ]

    noise_arr = generate_sigma_noise(wavelengths_nm, seed, base_sigma=base_sigma)
    sigma_noise = noise_arr[:, 1]

    spec_arr = generate_spectrum(
        wavelengths_nm=wavelengths_nm,
        seed=seed,
        lines=lines,
        sigma_noise=sigma_noise,
        vr_m_per_s=vr_m_per_s,  # keep global shift reproducible if you pass it
    )

    np.savetxt(noise_path, noise_arr, fmt="%.8f %.9f")
    np.savetxt(spectrum_path, spec_arr, fmt="%.8f %.8f")

    return spectrum_path, noise_path


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # force=True ensures you actually regenerate when tuning parameters.
    spectrum_path, noise_path = ensure_spectral_files(
        seed=12345,
        force=True,
        n_points=2_000_000,
        # vr_m_per_s=0.0,  # uncomment if you want zero Doppler shift
    )

    print(f"Spectral data files ready in {OUTPUTS_DIR}")
    print(f"- {spectrum_path}")
    print(f"- {noise_path}")


if __name__ == "__main__":
    main()
