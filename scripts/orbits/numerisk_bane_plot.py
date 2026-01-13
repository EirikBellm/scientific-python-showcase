
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from lib.system import generate_system
from scipy.signal import find_peaks

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
    t_int = 50 * year
    steps = int(50 * 10000)
    dt = t_int / steps

    return(year, t_int, steps, dt)

def get_anal_orbit(planet_idx):
    """
    Compute an analytic orbit for a single planet.

    Parameters
    ----------
    planet_idx : int
        Index of the planet in the system arrays.

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

def lesbar_plott():
    """
    Update plot defaults for readability.
    """

    plt.rcParams.update({
        "axes.labelsize": 15,
        "axes.titlesize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 10,
    })

def plot(x_y, idx):
    """
    Plot a single orbit curve and its initial position.
    """
    lesbar_plott()
    x, y = x_y
    posx, posy = system.initial_positions[:, idx]
    plt.plot(x, y, label=f"Planet {idx} orbit", linestyle="-")
    plt.scatter(posx, posy, s=20, label=f"Planet {idx} initial position")
    
    return

def plot_num_og_ana():
    """
    Plot simulated orbits against analytic orbits for all planets.

    Assumes:
    - "orbits.npz" exists at scripts/orbits/outputs/orbits.npz.
    - r has shape (T, N, 2) for time, planet, x/y.
    """
    
    data = np.load(ORBITS_PATH)
    bane = data["r"]


    N = system.number_of_planets
    for i in range(N):
        tuple = get_anal_orbit(i)
        plot(tuple, i)
        plt.plot(
            bane[:, i, 0],
            bane[:, i, 1],
            linestyle=(0, (16, 10)),
            label=f"Planet {i} (simulated)",
            lw=3,
        )

    plt.scatter(0, 0, s=100, c="yellow", marker="o", edgecolors="black", label="Star")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.axis("equal")
    plt.title("Analytic orbits compared to simulations")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

plot_num_og_ana()

def plott_r_av_r():
    """
    Plot distance r(t) from the star for all planets.
    """
    data = np.load(ORBITS_PATH)
    bane = data["r"]
    dt = float(data["dt"])
    tid = np.arange(bane.shape[0]) * dt
    N = system.number_of_planets
    for i in range(N):
        r_t = np.linalg.norm(bane[:, i, :], axis=1)[:, np.newaxis]
        plt.plot(tid, r_t, label=f"Planet {i} with e={system.eccentricities[i]:.5f}")
    plt.xlabel("Time [yr]")
    plt.ylabel("Distance from star [AU]")
    plt.title("Planet distance from the star over time")
    plt.legend()
    plt.tight_layout()
    plt.show()

plott_r_av_r()


def kep_2_test():
    """
    Test Kepler's second law by comparing swept areas near perihelion and aphelion.

    Requires:
    - scripts/orbits/outputs/orbits.npz with r positions.

    Returns
    -------
    None
        Prints a summary to stdout.
    """
    data = np.load(ORBITS_PATH)
    bane = data["r"]
    v = data["v"]
    dt = float(data["dt"])
    tid = np.arange(bane.shape[0]) * dt

    #
    r_norm = np.linalg.norm(bane[:, 0, :], axis = 1)
    v_norm = np.linalg.norm(v[:, 0, :], axis = 1)
    r_min = np.argmin(r_norm)  # perihelion index
    r_max = np.argmax(r_norm)  # aphelion index


    def areal(t0_idx, t_intervall):
        # t0 = t0 -0.5 * t_intervall
        areal = 0
        for i in range(int(t0_idx), int(t0_idx + t_intervall + 1)):
            delta_t = 1
            r = (bane[int(i), 0, :])
            r_delta_t = (bane[int(i) + delta_t, 0, :])

            # Triangle between r(i) and r(i+dt): |(r x r_next)|/2.
            areal += np.linalg.norm(0.5 * np.cross(r, r_delta_t))
        
        v_i_dt = v_norm[int(t0_idx) : int(t0_idx + t_intervall + 1)]
        avstand_i_dt = v_i_dt * dt
        avstand_dekket = np.sum(avstand_i_dt)
        v_mean = np.mean(v_i_dt)
        return areal, avstand_dekket, v_mean
    
    # Compare equal time windows around perihelion and aphelion.
    areal_rmin, avstand_rmin, mean_v_rmin =  areal(r_min, int(200))
    areal_rmax, avstand_rmax, mean_v_rmax  = areal(r_max, int(200))


    print(
        "Area swept from t0 to t1 at perihelion: "
        f"{areal_rmin} AU^2 - Distance: {avstand_rmin} - Mean speed: {mean_v_rmin}"
    )
    print(
        "Area swept from t0 to t1 at aphelion: "
        f"{areal_rmax} AU^2 - Distance: {avstand_rmax} - Mean speed: {mean_v_rmax}"
    )


# Compute orbital period and semi-major axis to compare with Kepler's 3rd law.
def periode(i):
    """
    Estimate orbital parameters from simulated positions and compare to theory.

    Parameters
    ----------
    i : int
        Planet index in the orbit array.

    Returns
    -------
    tuple
        (P_num, a, e, P_newton)
        where:
        - P_num: numerical period [yr]
        - a: semi-major axis [AU]
        - e: eccentricity [-]
        - P_newton: theoretical period from Newton's form [yr]

    """
    G = G_SOL
    Sm = system.star_mass

    data = np.load(ORBITS_PATH)
    bane = data["r"]
    dt = float(data["dt"])
    tid = np.arange(bane.shape[0]) * dt

    r_norm = np.linalg.norm(bane[:, i, :], axis = 1)
    # Find peaks in r(t) and estimate period from mean time spacing.
    topper, _ = find_peaks(r_norm)
    periode = np.mean(np.diff(tid[topper]))

    # Extremal distances.
    r_max = np.max(r_norm)
    r_min = np.min(r_norm)

    # Semi-axes and eccentricity for the ellipse.
    a = (r_max + r_min) / 2
    b = np.sqrt(r_max*r_min)
    e = np.sqrt(1 - (b/a)**2)
 
    # Theoretical period (Newton's form of Kepler's 3rd law).
    pn = np.sqrt((4*((np.pi)**2) * a**3 / (G*(system.masses[i] + Sm))))

    return periode, a, e, pn 

G = G_SOL
Sm = system.star_mass

N = system.number_of_planets
for i in range(N):
    P_num, a_num, e_num, P_newton = periode(i)
    a_ana = system.semi_major_axes[i]
    e_ana = system.eccentricities[i]
    avvik_P = abs((P_num**2 - (a_num**3 / Sm)) / P_num**2)
    avvik_a = abs(((a_num - a_ana) / a_ana) * 100)
    avvik_e = abs(((e_num - e_ana) / e_ana) * 100)
    avvik_PN = abs(((P_num - P_newton) / P_newton) * 100)
    print(f"Planet {i} period: {P_num:.2f} yr")
    print(f"Deviation from Kepler's 3rd law: {avvik_P:.5f}%")
    print(f"Deviation between analytic a and numeric a: {avvik_a:.5f}%")
    print(f"Deviation between analytic e and numeric e: {avvik_e:.5f}%")
    print(f"Deviation between Kepler P and Newton P: {avvik_PN:.5f}%")
