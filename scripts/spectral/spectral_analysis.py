"""
Spectral line analysis utilities.

Requires noise and flux files in scripts/spectral/outputs (see main()).
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS_DIR = HERE / "outputs"
DATA_DIR = OUTPUTS_DIR

C_LIGHT_M_PER_S = 299792458.0  # m/s
K_B_J_PER_K = 1.380649e-23  # J/K
M_P_KG = 1.67262192595e-27  # kg

# -----------------------------------------------------------
# Physics and statistics
# -----------------------------------------------------------

def dopplerskift(bølge0, maks_fart, lysfart):
    """
    Compute the maximum Doppler shift in wavelength for a line.

    Parameters
    ----------
    bølge0 : float
        Line center wavelength [nm].
    maks_fart : float
        Max radial speed |v| [m/s].
    lysfart : float
        Speed of light c [m/s].

    Returns
    -------
    float
        Maximum wavelength shift delta_lambda [nm].
    """
    return (maks_fart / lysfart) * bølge0


def dopplerskift_til_fart(delta_lamda, bølge0, lysfart):
    """
    Convert a wavelength shift to radial velocity.

    Parameters
    ----------
    delta_lamda : float
        Wavelength shift from lambda0 [nm].
    bølge0 : float
        Line center wavelength [nm].
    lysfart : float
        Speed of light c [m/s].

    Returns
    -------
    float
        Radial speed corresponding to the shift [m/s].
    """
    return (delta_lamda / bølge0) * lysfart


def hent_arrayer(
    bølge0,
    bølgelengder,
    fluks,
    noise,
    maks_fart,
    lysfart,
    dopplerbuffer,
):
    """
    Extract a wavelength window around a target line.

    Parameters
    ----------
    bølge0 : float
        Line center wavelength [nm].
    bølgelengder : (N,) array
        Wavelength axis [nm].
    fluks : (N,) array
        Normalized flux.
    noise : (N,) array
        Per-point sigma in the same units as flux.
    maks_fart : float
        Max |v_r| [m/s], used to set window size.
    lysfart : float
        Speed of light c [m/s].
    dopplerbuffer : float
        Factor (>1.0) to extend the window beyond the max shift.

    Returns
    -------
    (lambda_intervall, fluks_intervall, noise_intervall) : tuple of arrays
    """
    # Add a Doppler buffer to ensure the line is inside the window.

    skift = dopplerskift(bølge0, maks_fart, lysfart)
    intervall = skift * dopplerbuffer
    start_lamda = bølge0 - intervall
    slutt_lamda = bølge0 + intervall

    # Find window indices.
    start_indeks = int(np.searchsorted(bølgelengder, start_lamda, side="left"))
    slutt_indeks = int(np.searchsorted(bølgelengder, slutt_lamda, side="right"))


    return (
        bølgelengder[start_indeks:slutt_indeks],
        fluks[start_indeks:slutt_indeks],
        noise[start_indeks:slutt_indeks],
    )


def chi_kvadrat(
    fluks_intervall,
    noise_intervall,
    lambda_intervall,
    temperatur_grid,
    forskyvning_grid,
    fmin_grid,
    bølge0,
    masse,
    lysfart,
):
    """
    Brute-force chi^2 minimization over a 3D grid (T, delta_lambda, fmin).

    Method
    ------
    - Vectorize the 3D grid and evaluate the model for all combinations.
    - chi^2 = sum(((data - model) / sigma)^2) over wavelength points.
    - The best index gives (T_best, delta_lambda_best, fmin_best).

    Returns
    -------
    temperatur_best, forskyvning_best, fmin_best, chi2_min
    """
    fluks_intervall = np.asarray(fluks_intervall, dtype=float)
    noise_intervall = np.asarray(noise_intervall, dtype=float)
    lambda_intervall = np.asarray(lambda_intervall, dtype=float)

    # Vectorize: data arrays become 4D so axes are [T, delta_lambda, fmin, lambda].
    fluks = fluks_intervall[np.newaxis, np.newaxis, np.newaxis, :]
    noise = noise_intervall[np.newaxis, np.newaxis, np.newaxis, :]
    lamda = lambda_intervall[np.newaxis, np.newaxis, np.newaxis, :]

    # Parameters live on the first three axes and broadcast over lambda.
    temperatur = temperatur_grid[:, np.newaxis, np.newaxis, np.newaxis] 
    forskyvning = forskyvning_grid[np.newaxis, :, np.newaxis, np.newaxis]
    fmin = fmin_grid[np.newaxis, np.newaxis, :, np.newaxis]

    # Doppler width for all T.
    sigma_lamda = bølge0 * np.sqrt(K_B_J_PER_K * temperatur / masse) / lysfart

    # Gaussian peak is 1; line depth is controlled by fmin.
    gauss = np.exp(-0.5 * ((lamda - (bølge0 + forskyvning)) / sigma_lamda) ** 2)
    modell = 1.0 - (1.0 - fmin) * gauss

    # Sum over lambda so chi2 has shape (n_T, n_shift, n_fmin).
    chi2 = np.sum(((fluks - modell) / noise)**2, axis=-1)

    beste_indeks = np.unravel_index(np.argmin(chi2), chi2.shape)
    return (
        temperatur_grid[beste_indeks[0]],
        forskyvning_grid[beste_indeks[1]],
        fmin_grid[beste_indeks[2]],
        chi2[beste_indeks],
    )

# -----------------------------------------------------------
# Data handling
# -----------------------------------------------------------


def finn_parametre(gassnavn, bølge0, rutenett, oppsett):
    """
    Deprecated helper for a single-line brute-force chi^2 search.

    Steps:
    1) Extract the window around lambda0.
    2) Build 1D grids for T, delta_lambda, and fmin (same resolution).
    3) Run chi^2 and return the best values.
    """
    lambda_intervall, fluks_intervall, noise_intervall = hent_arrayer(
        bølge0,
        oppsett["bølgelengder"],
        oppsett["fluks"],
        oppsett["noise"],
        oppsett["maks_fart"],
        oppsett["lysfart"],
        oppsett["dopplerbuffer"],
    )

    # Temperature grid [K].
    temperatur_grid = np.linspace(
        oppsett["temperaturomrade"][0],
        oppsett["temperaturomrade"][1],
        rutenett,
    )

    # Delta-lambda grid from +/- max Doppler shift.
    maks_skift = dopplerskift(bølge0, oppsett["maks_fart"], oppsett["lysfart"])
    forskyvning_grid = np.linspace(-maks_skift, maks_skift, rutenett)

    # fmin grid: 0 < fmin <= 1 (1 = no absorption, lower = deeper line).
    fmin_grid = np.linspace(
        oppsett["fmin_omrade"][0],
        oppsett["fmin_omrade"][1],
        rutenett,
    )

    masse = oppsett["molekylmasser"][gassnavn]

    temperatur_best, forskyvning_best, fmin_best, chi2_min = chi_kvadrat(
        fluks_intervall,
        noise_intervall,
        lambda_intervall,
        temperatur_grid,
        forskyvning_grid,
        fmin_grid,
        bølge0,
        masse,
        oppsett["lysfart"],
    )

    return {
        "gass": gassnavn,
        "bølge0": bølge0,
        "temperatur": temperatur_best,
        "delta_lamda": forskyvning_best,
        "fmin": fmin_best,
        "chi2": chi2_min,
    }


def finn_parametre_iterativt(gassnavn, bølge0, rutenett, oppsett, antall=10, faktor=0.4):
    """
    Run iterative chi^2 searches with progressively narrower parameter ranges.

    Parameters
    ----------
    gassnavn : str
        Line identifier used to look up mass and global ranges.
    bølge0 : float
        Line center wavelength [nm].
    rutenett : int
        Grid points per dimension in each brute-force search.
    oppsett : dict
        Spectrum, noise, and physical constants (same structure as finn_parametre).
    antall : int, optional
        Number of iterations (default 10).
    faktor : float, optional
        Range scale per iteration (0 < faktor < 1). Default 0.4.

    Returns
    -------
    dict
        Best T, delta_lambda, fmin, and chi^2 from the final iteration.
    """

    # Extract the spectrum window around the target wavelength.
    lambda_intervall, fluks_intervall, noise_intervall = hent_arrayer(
        bølge0,
        oppsett["bølgelengder"],
        oppsett["fluks"],
        oppsett["noise"],
        oppsett["maks_fart"],
        oppsett["lysfart"],
        oppsett["dopplerbuffer"],
    )


    # Molecular mass for the chosen gas.
    masse = oppsett["molekylmasser"][gassnavn]

    # Global search ranges for temperature, Doppler shift, and fmin.
    temp_global = oppsett["temperaturomrade"]
    fmin_global = oppsett["fmin_omrade"]

    # Maximum Doppler shift.
    maks_skift = dopplerskift(bølge0, oppsett["maks_fart"], oppsett["lysfart"])
    shift_global = (-maks_skift, maks_skift)

    # Start with the full global ranges.
    temp_range = temp_global
    shift_range = shift_global
    fmin_range = fmin_global

    # Store the best result from the last iteration.
    beste_resultat = None

    # Iterative chi^2: coarse first, then narrow around the best value.
    for steg in range(antall):
        # Build 1D grids for the current ranges.
        temperatur_grid = np.linspace(temp_range[0], temp_range[1], rutenett)
        forskyvning_grid = np.linspace(shift_range[0], shift_range[1], rutenett)
        fmin_grid = np.linspace(fmin_range[0], fmin_range[1], rutenett)

        # Find the parameter set with minimum chi^2.
        (
            temperatur_best,
            forskyvning_best,
            fmin_best,
            chi2_min,
        ) = chi_kvadrat(
            fluks_intervall,
            noise_intervall,
            lambda_intervall,
            temperatur_grid,
            forskyvning_grid,
            fmin_grid,
            bølge0,
            masse,
            oppsett["lysfart"],
        )

        # Store best values from this iteration.
        beste_resultat = {
            "gass": gassnavn,
            "bølge0": bølge0,
            "temperatur": temperatur_best,
            "delta_lamda": forskyvning_best,
            "fmin": fmin_best,
            "chi2": chi2_min,
        }

        # Narrow the search ranges around the best values for the next iteration.
        if steg < antall - 1:
            temp_range = snevre_inn(temperatur_best, temp_range, temp_global, faktor)
            shift_range = snevre_inn(forskyvning_best, shift_range, shift_global, faktor)
            fmin_range = snevre_inn(fmin_best, fmin_range, fmin_global, faktor)
    return beste_resultat


def snevre_inn(besteverdi, gjeldende, globalt, faktor):
    """
    Narrow a range around a best value without crossing global bounds.

    Parameters
    ----------
    besteverdi : float
        Value to center the new range on.
    gjeldende : tuple(float, float)
        Current range (min, max) to scale.
    globalt : tuple(float, float)
        Absolute physical bounds for the range.
    faktor : float
        Scale for range width (0 < faktor < 1 narrows the range).

    Returns
    -------
    (nedre, ovre) : tuple(float, float)
        Updated range within the global bounds.

    """
    # Current range width.
    bredde = gjeldende[1] - gjeldende[0]

    # New, narrower width after scaling.
    ny_bredde = bredde * faktor

    # Half-width for a symmetric interval around the best value.
    halv = ny_bredde / 2.0

    # Lower bound, clamped to the global range.
    nedre = max(globalt[0], besteverdi - halv)

    # Upper bound, clamped to the global range.
    ovre = min(globalt[1], besteverdi + halv)

    return (nedre, ovre)


def søk(gass_liste, oppsett, plottmappe, skriptmappe, rutenett=25):
    """
    Run the full line search, save plots and CSV outputs.
    """
    resultater = []
    for gassnavn, linjer in gass_liste:
        for bølge0 in linjer:
            if bølge0 is None:
                continue
            resultat = finn_parametre_iterativt(gassnavn, bølge0, rutenett, oppsett)
            resultater.append(resultat)
            # Print, plot, and store results.
 
            plott(resultat, oppsett, plottmappe, lagre=True)

    # Save aggregate outputs.
    sprednings_sti = parameter_spredning(resultater, plottmappe, lagre=True)
    csv_sti = skriv_csv(resultater, skriptmappe)
    print(f"\nSaved {len(resultater)} results to {csv_sti}")
    print(f"Plots saved in {plottmappe}")
    if sprednings_sti:
        print(f"Parameter plot saved to {sprednings_sti}")


# -----------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------

def gauss_plott(
    lambda_intervall,
    bølge0,
    delta_lamda,
    temperatur,
    fmin,
    masse,
    boltzmann,
    lysfart,
):
    """
    Helper for plotting the Gaussian line model.
    """
    sigma_lamda = bølge0 * np.sqrt(boltzmann * temperatur / masse) / lysfart
    # Peak = 1 at lambda = lambda0 + delta_lambda.
    gauss = np.exp(-0.5 * ((lambda_intervall - (bølge0 + delta_lamda)) / sigma_lamda) ** 2)
    # Continuum is 1, line center is fmin.
    return 1.0 - (1.0 - fmin) * gauss


def plott(resultat, oppsett, plottmappe, lagre=True):
    """
    Plot data and best-fit model for one line window.
    """
    lambda_intervall, fluks_intervall, noise_intervall = hent_arrayer(
        resultat["bølge0"],
        oppsett["bølgelengder"],
        oppsett["fluks"],
        oppsett["noise"],
        oppsett["maks_fart"],
        oppsett["lysfart"],
        oppsett["dopplerbuffer"],
    )

    radial_hastighet = dopplerskift_til_fart(
        resultat["delta_lamda"],
        resultat["bølge0"],
        oppsett["lysfart"],
    )

    modell = gauss_plott(
        lambda_intervall,
        resultat["bølge0"],
        resultat["delta_lamda"],
        resultat["temperatur"],
        resultat["fmin"],
        oppsett["molekylmasser"][resultat["gass"]],
        oppsett["boltzmann"],
        oppsett["lysfart"],
    )
    if lambda_intervall.size == 0:
        print(
            f"{resultat['gass']} {resultat['bølge0']:.1f}nm: "
            "no points in interval; skipping plot"
        )
        return None
    print(
        f"{resultat['gass']} {resultat['bølge0']:.1f}nm: "
        f"N={len(lambda_intervall)} points, "
        f"range={lambda_intervall[0]:.5f}-{lambda_intervall[-1]:.5f} nm"
    )
    plt.figure(figsize=(7, 4))
    plt.plot(lambda_intervall, fluks_intervall, label="Flux", linewidth=1.0)
    plt.plot(lambda_intervall, modell, label="Model", linewidth=1.5)
    plt.fill_between(
        lambda_intervall,
        1 - noise_intervall,
        1 + noise_intervall,
        color="gray",
        alpha=0.3,
        label="Noise +/- sigma",
    )
    plt.axvline(
        resultat["bølge0"],
        color="k",
        linestyle="--",
        linewidth=0.8,
        label="lambda0",
    )
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Normalized flux")
    plt.title(
        f"{resultat['gass']} at {resultat['bølge0']:.1f} nm "
        f"(v_r={radial_hastighet/1000:+.2f} km/s)"
    )
    plt.legend()
    plt.tight_layout()

    if lagre:
        plottmappe.mkdir(parents=True, exist_ok=True)
        plott_sti = plottmappe / f"{resultat['gass']}_{int(round(resultat['bølge0']))}nm.png"
        plt.savefig(plott_sti)
        plt.close()
        return plott_sti

    plott_sti = None
    try:
        plt.show()
    except Exception as feil:
        print(
            f"Failed to show plot for {resultat['gass']} at "
            f"{resultat['bølge0']:.1f} nm: {feil}"
        )
    finally:
        plt.close()
    return plott_sti


def parameter_spredning(resultater, plottmappe, lagre=True):
    """
    Scatter plot of temperature vs radial velocity colored by relative flux (fmin).
    """
    if not resultater:
        return None

    temperaturer = [rad["temperatur"] for rad in resultater]
    hastigheter = [
        dopplerskift_til_fart(rad["delta_lamda"], rad["bølge0"], C_LIGHT_M_PER_S)
        for rad in resultater
    ]
    flukser = [rad["fmin"] for rad in resultater]

    plt.figure(figsize=(7, 4.5))
    spredning = plt.scatter(
        temperaturer,
        hastigheter,
        c=flukser,
        cmap="viridis",
        edgecolors="k",
        linewidths=0.4,
        s=60,
    )
    cbar = plt.colorbar(spredning)
    cbar.set_label("Relative flux (fmin)")
    plt.xlabel("Temperature [K]")
    plt.ylabel("Radial velocity [m/s]")
    plt.title("Parameter distribution across all lines")
    plt.tight_layout()

    if lagre:
        plottmappe.mkdir(parents=True, exist_ok=True)
        sprednings_sti = plottmappe / "parameter_spredning.png"
        plt.savefig(sprednings_sti)
        plt.close()
        return sprednings_sti

    sprednings_sti = None
    try:
        plt.show()
    except Exception as feil:
        print(f"Failed to show parameter plot: {feil}")
    finally:
        plt.close()
    return sprednings_sti


def skriv_csv(resultater, skriptmappe):
    """
    Save all line results as a CSV.
    """
    csv_sti = skriptmappe / "line_search_results.csv"
    with csv_sti.open("w", newline="") as fil:
        skriver = csv.writer(fil)
        skriver.writerow(
            ["gas", "lambda_nm", "temperature_K", "radial_speed_m_per_s", "fmin", "chi2"]
        )
        for rad in resultater:
            radial_hastighet = dopplerskift_til_fart(
                rad["delta_lamda"],
                rad["bølge0"],
                C_LIGHT_M_PER_S,
            )
            skriver.writerow(
                [
                    rad["gass"],
                    f"{rad['bølge0']:.4f}",
                    f"{rad['temperatur']:.4f}",
                    f"{radial_hastighet:.4f}",
                    f"{rad['fmin']:.6f}",
                    f"{rad['chi2']:.6f}",
                ]
            )
    return csv_sti


def main():
    """
    1) Load spectrum and noise (nm, flux, sigma).
    2) Define candidate lines.
    3) Set constants and molecular masses.
    4) Run the line search.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    skriptmappe = OUTPUTS_DIR
    plottmappe = OUTPUTS_DIR

    # Load spectrum and per-point noise from outputs.
    spectrum_path = DATA_DIR / "spectrum_600nm_1000nm.txt"
    if not spectrum_path.exists():
        spectrum_candidates = sorted(DATA_DIR.glob("spectrum_seed*_600nm_1000nm.txt"))
        if not spectrum_candidates:
            raise FileNotFoundError(
                f"No spectrum files found in {DATA_DIR}. Run spectral_data.py first."
            )
        spectrum_path = spectrum_candidates[-1]
    noise_path = DATA_DIR / "sigma_noise.txt"
    if not noise_path.exists():
        raise FileNotFoundError(
            f"{noise_path} not found. Run spectral_data.py first."
        )

    spektrumdata = np.loadtxt(spectrum_path)
    noisedata = np.loadtxt(noise_path)
    bølgelengder = spektrumdata[:, 0]
    fluks = spektrumdata[:, 1]
    noise = noisedata[:, 1]

    # Allowed line depth: fmin in [0.65, 1.0] (with a 0.05 buffer).
    fmin_omrade = (0.65, 1.0)

    # Candidate lines (nm). None means no second/third line.
    gass_liste = [
        ("O2", [632.0, 690.0, 760.0]),
        ("H2O", [720.0, 820.0, 940.0]),
    ]

    # Molecular masses: atomic mass * proton mass.
    protonmasse = M_P_KG
    molekylmasser = {
        "O2": 32.0 * protonmasse,
        "H2O": 18.0 * protonmasse,
    }

    oppsett = {
        "bølgelengder": bølgelengder,
        "fluks": fluks,
        "noise": noise,
        "temperaturomrade": (100, 500),   # K
        "fmin_omrade": fmin_omrade,
        "maks_fart": 10_000.0,               # m/s (radial, used for window and delta-lambda grid)
        "dopplerbuffer": 1.10,               # 10% margin around max shift
        "lysfart": C_LIGHT_M_PER_S,          # m/s
        "boltzmann": K_B_J_PER_K,            # J/K
        "molekylmasser": molekylmasser,
    }

    
    N = 35

    søk(gass_liste, oppsett, plottmappe, skriptmappe, rutenett=N)

if __name__ == "__main__":
    main()
