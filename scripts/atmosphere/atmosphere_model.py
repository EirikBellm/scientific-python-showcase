import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from lib.system import generate_system

HERE = Path(__file__).resolve().parent
OUTPUTS_DIR = HERE / "outputs"

# NOTE:
# Previously, `system = generate_system(...)` was executed at import time.
# That can cause side effects when this module is imported elsewhere.
# We keep the same features, but defer creation of `system` to main().
system = None

DATA_PATH = OUTPUTS_DIR / "atmosphere_data.npz"
G_SI = 6.6743e-11  # m^3 kg^-1 s^-2
M_SUN_KG = 1.9884754153381438e30  # kg
M_P_KG = 1.67262192595e-27  # kg
K_B_J_PER_K = 1.380649e-23  # J/K

def modell(mu, system_override=None, planet_index=1):
    """
    Compute density and temperature profiles by combining adiabatic and isothermal layers.

    Method
    ------
    Integrate adiabatically from the surface until T reaches T0/2, then
    continue isothermally with T fixed at T0/2 using Euler steps.

    Parameters
    ----------
    mu : float
        Mean molecular weight (atomic mass units).
    system_override : object, optional
        System object with masses, radii, and atmospheric_densities arrays.
    planet_index : int, optional
        Planet index to use for surface values (default 1).

    Returns
    -------
    tuple(np.ndarray, np.ndarray, np.ndarray, float)
        Density [kg/m^3], temperature [K], height [m], and transition height [m].
    """
    # Use override if provided; otherwise fall back to module-level system.
    system_use = system if system_override is None else system_override
    if system_use is None:
        raise RuntimeError(
            "No system is available. Either pass `system_override=...` to modell(), "
            "or run this file as a script so `main()` creates `system`."
        )

    # Surface gravity from Newton's law.
    M = system_use.masses[planet_index] * M_SUN_KG  # Solar masses -> kg
    R = system_use.radii[planet_index] * 1_000      # km -> m
    g0 = G_SI * M / R**2             # m/s^2

    # Surface values.
    rho0 = system_use.atmospheric_densities[planet_index]  # kg/m^3
    T0 = 295.2                              # K (reference surface temperature)

    # Adiabatic index for an ideal gas.
    gamma = 1.4

    # Physical constants.
    m_H = M_P_KG       # kg
    k = K_B_J_PER_K    # J/K

    # Precompute constants for the adiabatic relation and ideal gas law.
    A = (k * T0) / (mu * m_H * rho0**(gamma - 1))

    # Shared prefactors (gravity handled separately).
    B0 = - (mu * m_H * rho0**(gamma - 1)) / (gamma * k * T0)
    C0 = -(mu * m_H) / (k * (T0/2))

    # Numerical parameters.
    tol = 10**(-6)  # kg/m^3
    dh = 10         # m

    # Initialize arrays at h = 0.
    rho_list = [rho0]
    rho_ny = rho0
    T_list = [T0]
    h_list = [0]
    h_ny = 0

    # --- Adiabatic layer (T decreases with height) ---
    # Stop when T reaches T0/2, then switch to isothermal.
    while T_list[-1] > T0/2:
        g = G_SI * M / (R + h_ny)**2
        # Euler update for the adiabatic density gradient.
        rho_ny += (B0 * g) * (rho_ny**(2-gamma)) * dh

        # Temperature from the adiabatic relation.
        T_ny = A * (rho_ny**(gamma-1)) * mu * m_H / k

        # Advance height.
        h_ny += dh
        
        # Append results.
        rho_list.append(rho_ny)
        T_list.append(T_ny)
        h_list.append(h_ny)
    # Store height at the transition.
    h_half = h_ny
    # --- Isothermal layer (T fixed at T0/2) ---
    while rho_ny > tol:
        g = G_SI * M / (R + h_ny)**2
        # Euler update for the isothermal density gradient.
        rho_ny += (C0 * g) * rho_ny * dh 

        # Temperature is constant in the isothermal layer.
        T_ny = T0/2

        # Advance height.
        h_ny += dh

        # Safety cap: stop after 1000 km.
        if h_ny > 1e6:  # 1 000 km
            break
        
        # Append results.
        rho_list.append(rho_ny)
        T_list.append(T_ny)
        h_list.append(h_ny)

    return np.array(rho_list), np.array(T_list), np.array(h_list), h_half

def main():
    global system
    system = generate_system(n_planets=5, seed=5289)

    # Representative mean molecular weight for 1/3 * (CH4 + H2O + CO).
    mu = 20.65
    rho, T, h, h_half = modell(mu)

    # Save results for reuse.
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_PATH, rho=rho, T=T, h=h, mu=mu)


    # --- Plot results ---

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })

    # X-axis ticks every 25 km.
    x_max = (h / 1e3).max()
    x_ticks = np.arange(0, x_max + 25, 25)   # from 0 to max height, step 25 km

    # --- Temperature profile ---
    plt.figure(figsize=(8,6))
    plt.plot(h/1e3, T, color='orange')
    h_half_km = h_half / 1e3

    plt.axvline(x=h_half_km, color='pink', linestyle='--', linewidth=1,
                label=f"T = T0/2 at {h_half_km:.1f} km")
    plt.title('Temperature profile')
    plt.xlabel('Height above surface [km]')
    plt.ylabel('Temperature [K]')
    plt.grid(True)
    plt.xticks(x_ticks)
    plt.legend()

    # --- Density (log scale) ---
    plt.figure(figsize=(8,6))
    plt.plot(h/1e3, rho, color='darkblue')
    plt.axvline(x=h_half_km, color='pink', linestyle='--', linewidth=1,
                label=f"T = T0/2 at {h_half_km:.1f} km")
    plt.yscale('log')
    plt.title('Density profile (log scale)')
    plt.xlabel('Height above surface [km]')
    plt.ylabel('Density [kg/m^3]')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(x_ticks)
    plt.legend()

    # --- Density (linear scale) ---
    plt.figure(figsize=(8,6))
    plt.plot(h/1e3, rho, color='blue')
    plt.axvline(x=h_half_km, color='pink', linestyle='--', linewidth=1,
                label=f"T = T0/2 at {h_half_km:.1f} km")
    plt.title('Density profile (linear scale)')
    plt.xlabel('Height above surface [km]')
    plt.ylabel('Density [kg/m^3]')
    plt.grid(True)
    plt.xticks(x_ticks)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
