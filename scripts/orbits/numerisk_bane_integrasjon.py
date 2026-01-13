from pathlib import Path

import numpy as np

from lib.system import generate_system

HERE = Path(__file__).resolve().parent
OUTPUTS_DIR = HERE / "outputs"
ORBITS_PATH = OUTPUTS_DIR / "orbits.npz"

seed = 5289
system = generate_system(n_planets=5, seed=seed)
G_SOL = 4 * np.pi**2  # AU^3 / (yr^2 * M_sun)

def set_integration_parameters():
    """
    Set integration parameters using planet 0's orbital period as 1 year.

    Returns
    -------
    year : float
        Planet 0 period in Earth years.
    t_int : float
        Total integration time in Earth years.
    steps : int
        Number of integration steps.
    dt : float
        Step size in years.
    """
    Sm = system.star_mass


    a0 = system.semi_major_axes[0]
    year = np.sqrt((a0**3) / Sm)  # Kepler's 3rd law (scaled to this star mass)
    t_int = 25 * year
    steps = int(50 * 10000)
    dt = t_int / steps

    return(year, t_int, steps, dt)

def get_state():
    """
    Return initial positions and velocities for all planets.

    Returns
    -------
    r : ndarray, shape (N, 2)
        Initial positions in AU.
    v : ndarray, shape (N, 2)
        Initial velocities in AU/yr.
    """
    nr = system.number_of_planets
    r = np.zeros((nr, 2))
    v = np.zeros((nr, 2))
    for i in range(system.number_of_planets):
        r[i,:] = system.initial_positions[:, i]  # AU
        v[i,:] = system.initial_velocities[:, i]  # AU/yr
    return(r, v)

def leapfrog_one_step(r, v, dt, star_mass=None):
    """
    Perform one leapfrog integration step.

    Parameters
    ----------
    r : ndarray, shape (N, 2)
        Positions in AU.
    v : ndarray, shape (N, 2)
        Half-step velocities in AU/yr.
    dt : float
        Time step in years.
    star_mass : float, optional
        Star mass in Msun. Defaults to the module system's star mass.

    Returns
    -------
    r : ndarray, shape (N, 2)
        Updated positions at t + dt.
    v : ndarray, shape (N, 2)
        Updated velocities after acceleration update.
    a : ndarray, shape (N, 2)
        Accelerations at the new positions.
    """

    G = G_SOL
    Sm = system.star_mass if star_mass is None else star_mass

    r = r + v * dt
    a = -G * Sm * r / np.linalg.norm(r, axis = 1)[:, np.newaxis]**3
    v = v + a * dt
    return(r, v, a)

def planet_integration():
    """
    Integrate the system over [0, T] with leapfrog and save results.

    Writes "orbits.npz" to scripts/orbits/outputs containing:
    - r : positions over time
    - v : velocities over time
    - a : accelerations over time
    - dt : time step
    """
    year, t_int, steps, dt = set_integration_parameters()
    r, v = get_state()


    G = G_SOL
    Sm = system.star_mass


    # Initialize half-step velocity for leapfrog.
    a = -G  * Sm * r / np.linalg.norm(r, axis = 1)[:, np.newaxis]**3
    v_halv = v + a * (1/2) * dt

    # Initialize arrays.
    shape = (steps + 1, *r.shape)
    r_arr, v_arr, a_arr = (np.zeros(shape) for _ in range(3))
    r_arr[0], v_arr[0], a_arr[0] = r, v, a

    for i in range(steps):
        r, v_halv, a = leapfrog_one_step(r, v_halv, dt, star_mass=system.star_mass)

        # Store values for plotting.
        r_arr[i+1] = r
        a_arr[i+1] = a
        v_arr[i+1] = v_halv - 0.5*dt*a  # Velocity at full time step.

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ORBITS_PATH,
        r=r_arr,        
        v=v_arr,        
        a=a_arr,         
        dt=dt
        )

if __name__ == "__main__":
    # Only needs to run once to generate scripts/orbits/outputs/orbits.npz.
    planet_integration()

def get_anal_orbit(planet_idx):
    """
    Compute an analytic orbit for a single planet.

    Parameters
    ----------
    planet_idx : int
        Planet index in the system arrays.

    Returns
    -------
    (x, y) : tuple of arrays
        Orbit coordinates in AU with 1000 samples.
    """
    AA = system.aphelion_angles[planet_idx] + np.pi  # Convert aphelion to perihelion angle.
    e = system.eccentricities[planet_idx]
    a = system.semi_major_axes[planet_idx]
    theta = np.linspace(0, 2*np.pi, 1000)           # 0..2pi
    f_arr = theta - AA                              # Rotate by the orbit angle
    r_arr = a*(1 - e**2) / (1 + e * np.cos(f_arr))  # r(f)

    # Polar to Cartesian.
    x = r_arr * np.cos(theta) 
    y = r_arr * np.sin(theta) 
    return((x, y))
