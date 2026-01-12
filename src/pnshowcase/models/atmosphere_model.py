##############################
##### IKKE BRUKT KODEMAL #####
##############################


import ast2000tools.constants as const
import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.solar_system import SolarSystem
from pathlib import Path

system = SolarSystem(5289)
DATA_PATH = Path(__file__).with_name("atmosphere_data.npz")

def modell(mu):
    """
    Beregner tetthets- og temperaturprofiler i atmosfæren ved å kombinere en adiabatisk og isotermisk modell.

    Metode
    ------
    Atmosfæren integreres først adiabatisk fra overflaten til temperaturen har falt til T0/2,
    deretter isotermisk med konstant T = T0/2 ved hjelp av Euler-steg.

    Parametre
    ---------
    mu : float
        Gjennomsnittlig molekylvekt (i atommasseenheter) for atmosfæren.

    Returnerer
    ----------
    tuple(np.ndarray, np.ndarray, np.ndarray)
        Arrays for tetthet [kg/m^3], temperatur [K] og høyde [m] fra bakken og oppover.
    """
    # Finner tyngdeakselerasjonen ved overflaten fra Newtons gravitasjonslov.
    M = system.masses[1] * const.m_sun  # Solmasser til kg
    R = system.radii[1] * 1_000         # km til m
    g0 = const.G * M / R**2             # Tyngdeakselerasjon ved bakken [m/s^2]

    # Startverdier ved overflaten:
    rho0 = system.atmospheric_densities[1]  # Tetthet ved overflaten [kg/m^3]
    T0 = 295.2                              # Estimert temperatur ved overflaten [K] fra del 3

    # Konstant gamma for adiabatisk idealgass
    gamma = 1.4

    # Fysiske konstanter:
    m_H = const.m_p  # Masse til hydrogenatom [kg] (elektronmassen neglisjeres)
    k = const.k_B    # Boltzmanns konstant [J/K = m^2 kg / (s^2 K)]

    # Konstanter vi bruker som kommer av at vi kombinerer:
    # - adiabatisk sammenheng T proposjonal med rho^{gamma-1}
    # - og idealgassloven P = (rho * k * T) / (mu * m_H)
    # A brukes til å regne T(rho) i den adiabatiske delen
    A = (k * T0) / (mu * m_H * rho0**(gamma - 1))

    # Konstant prefaktor uten gravitasjonsterm som brukes i begge modeller
    B0 = - (mu * m_H * rho0**(gamma - 1)) / (gamma * k * T0)
    C0 = -(mu * m_H) / (k * (T0/2))

    # Numeriske parametere:
    tol = 10**(-6)  # Avslutt når tettheten blir svært liten [kg/m^3]
    dh = 10         # Høydesteg [m]

    # Lager lister og fyller inn startverdier ved h = 0
    rho_list = [rho0]
    rho_ny = rho0
    T_list = [T0]
    h_list = [0]
    h_ny = 0

    # ------- Adiabatisk del (T avtar med høyde) -------
    # Stopper når temperaturen er falt til T0/2, da går vi over til isotermisk modell.
    while T_list[-1] > T0/2:
        g = const.G * M / (R + h_ny)**2
        # Euler-steg for tetthet fra den adiabatiske differensiallikningen:
        # rho_{ny} = rho + d rho/dh * dh
        rho_ny += (B0 * g) * (rho_ny**(2-gamma)) * dh

        # Temperatur fra adiabatisk formel
        T_ny = A * (rho_ny**(gamma-1)) * mu * m_H / k

        # Oppdater høyde
        h_ny += dh
        
        # Legg til nye verdier i listene
        rho_list.append(rho_ny)
        T_list.append(T_ny)
        h_list.append(h_ny)
    # lagrer h når vi bytter regime
    h_half = h_ny
    # ------- Isotermisk del (T konstant = T0/2) -------
    # Her antar vi at temperaturen har blitt så lav at videre kjøling er liten,
    # og vi modellerer resten av atmosfæren som isotermisk med T = T0/2.
    while rho_ny > tol:
        g = const.G * M / (R + h_ny)**2
        # Euler-steg for isotermisk atmosfære: d rho/dh = D * rho
        rho_ny += (C0 * g) * rho_ny * dh 

        # Temperaturen holdes konstant i den isotermiske delen
        T_ny = T0/2

        # Øker høyden
        h_ny += dh

        # Går ikke over 1000 km for å unngå uendelig løkke
        if h_ny > 1e6:  # 1 000 km
            break
        
        # Legger inn nye verdier
        rho_list.append(rho_ny)
        T_list.append(T_ny)
        h_list.append(h_ny)

    return np.array(rho_list), np.array(T_list), np.array(h_list), h_half

def main():
    # Velger gjennomsnittlig molekylvekt mu for atmosfære av 1/3 * (CH4 + H2O + CO)
    mu = 20.65
    rho, T, h, h_half = modell(mu)

    # Lagrer resultatene slik at de kan lastes inn i senere deloppgaver
    np.savez(DATA_PATH, rho=rho, T=T, h=h, mu=mu)


    # ---------- Plotting av resultatene ----------

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })

    # Definerer tall på x_aksen hvert 25. tall
    x_max = (h / 1e3).max()
    x_ticks = np.arange(0, x_max + 25, 25)   # from 0 to max height, step 25 km

    # --- Temperaturprofil ---
    plt.figure(figsize=(8,6))
    plt.plot(h/1e3, T, color='orange')
    h_half_km = h_half / 1e3

    plt.axvline(x=h_half_km, color='pink', linestyle='--', linewidth=1,
                label=f"T = T₀/2 ved {h_half_km:.1f} km")
    plt.title('Temperaturprofil')
    plt.xlabel('Høyde over overflaten [km]')
    plt.ylabel('Temperatur [K]')
    plt.grid(True)
    plt.xticks(x_ticks)
    plt.legend()

    # --- Tetthet i log-skala ---
    plt.figure(figsize=(8,6))
    plt.plot(h/1e3, rho, color='darkblue')
    plt.axvline(x=h_half_km, color='pink', linestyle='--', linewidth=1,
                label=f"T = T₀/2 ved {h_half_km:.1f} km")
    plt.yscale('log')
    plt.title('Tetthetsprofil (logaritmisk skala)')
    plt.xlabel('Høyde over overflaten [km]')
    plt.ylabel('Tetthet [kg/m³]')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(x_ticks)
    plt.legend()

    # --- Tetthet i lineær skala ---
    plt.figure(figsize=(8,6))
    plt.plot(h/1e3, rho, color='blue')
    plt.axvline(x=h_half_km, color='pink', linestyle='--', linewidth=1,
                label=f"T = T₀/2 ved {h_half_km:.1f} km")
    plt.title('Tetthetsprofil (lineær skala)')
    plt.xlabel('Høyde over overflaten [km]')
    plt.ylabel('Tetthet [kg/m³]')
    plt.grid(True)
    plt.xticks(x_ticks)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
