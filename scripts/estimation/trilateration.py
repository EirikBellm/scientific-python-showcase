import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]  # repo root
ORBITS_PATH = ROOT / "scripts" / "orbits" / "outputs" / "orbits.npz"

bane = None  # shape: [time, planet_index, (x, y)]
dt = None


def _load_orbits_npz(path=ORBITS_PATH):
    """
    Load planet positions from the orbital integration output.

    Parameters
    ----------
    path : str
        Path to the .npz file produced by the orbit integrator.

    Returns
    -------
    (bane, dt)
        bane: ndarray, shape [time, planet_index, (x, y)]
        dt: float, time step in years
    """
    data = np.load(path)
    bane_local = data["r"]  # shape: [time, planet_index, (x, y)]
    dt_local = float(data["dt"])
    return bane_local, dt_local


# Solve for spacecraft position using trilateration.
def trilateration(tid, avstander, planeter):
    """
    Estimate spacecraft position from measured distances.

    Parameters
    ----------
    tid : float
        Time of the measurement [yr].
    avstander : array-like
        Distances to reference bodies [AU].
    planeter : array-like
        Indices of reference planets.

    Returns
    -------
    (x_est, y_est), residuals
        Estimated position [AU] and least-squares residuals.
    """
    # Ensure orbit data has been loaded.
    if bane is None or dt is None:
        raise RuntimeError(
            "Orbit data not loaded. Run this file as a script (python ...py) "
            "so it can load scripts/orbits/outputs/orbits.npz, or load it "
            "yourself and set "
            "the globals `bane` and `dt` before calling trilateration()."
        )

    A = []
    B = []

    # Use the star as a fixed reference at (0, 0).
    x0, y0 = 0, 0
    r0 = avstander[-1]

    # Build A and B from the planet constraints.
    for idx, p_nr in enumerate(planeter):

        # Floor time to the nearest integration step.
        t_ned = math.floor(tid / dt)
        # Fractional offset between integration steps.
        rest = (tid - t_ned * dt) / dt

        # Linear interpolation of planet position.
        xy = (1 - rest) * bane[t_ned + 1, p_nr, :] + rest * bane[t_ned, p_nr, :]

        xi, yi = float(xy[0]), float(xy[1])

        ri = avstander[idx]

        Ai = [-2 * (xi - x0), -2 * (yi - y0)]
        Bi = (ri**2 - r0**2) - ((xi**2 - x0**2) + (yi**2 - y0**2))

        A.append(Ai)
        B.append(Bi)

    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    # Solve Ax = B with least squares.
    pos, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    x_est, y_est = pos

    return (x_est, y_est), residuals


# Test the trilateration method with synthetic data.
def test_trilaterasjon(antall_planeter, n_tester=100, pos_range=50):
    """
    Monte Carlo test using random positions and times.

    Parameters
    ----------
    antall_planeter : int
        Number of reference planets.
    n_tester : int, optional
        Number of trials.
    pos_range : float, optional
        Position range for each coordinate [AU].

    Returns
    -------
    None
        Prints the mean relative error over the trials.
    """
    # Ensure orbit data has been loaded.
    if bane is None or dt is None:
        raise RuntimeError(
            "Orbit data not loaded. Run this file as a script (python ...py) "
            "so it can load scripts/orbits/outputs/orbits.npz, or load it "
            "yourself and set "
            "the globals `bane` and `dt` before calling test_trilaterasjon()."
        )

    feil_liste = []

    # Add measurement noise to simulate realistic distances.
    støy_prosent = 0.01

    # Run trials.
    for i in range(n_tester):
        # Random positions, times, and planet indices.
        ekte_pos = np.random.uniform(-pos_range, pos_range, size=2)
        tid = np.random.choice(np.arange(50))
        planeter = np.random.choice(np.arange(5), size=antall_planeter, replace=False)

        avstander = []

        # Distances to reference bodies.
        for i in planeter:
            planet_pos = bane[int(tid / dt), i, :]
            avstand = np.linalg.norm(planet_pos - ekte_pos)
            stoy = np.random.normal(0, støy_prosent * avstand)
            avstander.append(avstand + stoy)

        avstand_stjerne = np.linalg.norm(ekte_pos - np.array([0, 0]))
        støy_stjerne = np.random.normal(0, støy_prosent * avstand_stjerne)
        avstander.append(avstand_stjerne + støy_stjerne)
        avstander = np.array(avstander)

        # Estimate position using trilateration.
        estimert_pos, _ = trilateration(tid, avstander, planeter)

        # Relative error for this trial.
        rel_feil = (np.linalg.norm((ekte_pos - estimert_pos)) / (np.linalg.norm(ekte_pos))) * 100
        feil_liste.append(rel_feil)

    gjennomsnittlig_feil = np.mean(feil_liste)

    print(f"Ran {n_tester} trials with {antall_planeter} random planets and times.")
    print(f"Mean relative error: {gjennomsnittlig_feil}%")


def main():
    global bane, dt

    # Load planet positions from the orbital integration output.
    bane, dt = _load_orbits_npz(ORBITS_PATH)

    # Example input data.
    tid = 1  # measurement time [yr]
    planeter = [0, 1, 2]
    avstander = [3, 14, 5]

    # Solve and print the estimate.
    pos, residualer = trilateration(tid, avstander, planeter)
    print(f"Estimated spacecraft position at t={tid}: {pos}")
    print(f"Residuals: {residualer}")

    # Run the test with a fixed number of reference planets.
    antall_planeter = 5  # 2+1 (star) is the minimum; more improves accuracy.
    test_trilaterasjon(antall_planeter)


if __name__ == "__main__":
    main()
