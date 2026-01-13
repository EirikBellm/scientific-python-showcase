import numpy as np

G_SOL = 4 * np.pi**2  # AU^3 / (Msun * yr^2)

class System:
    def __init__(self,
                 masses, radii, atmospheric_densities,
                 star_mass,
                 semi_major_axes, aphelion_angles, eccentricities,
                 initial_positions, initial_velocities,
                 seed):
        self.masses = np.array(masses, dtype=float)                        # (N,) Msun
        self.radii = np.array(radii, dtype=float)                          # (N,) km
        self.atmospheric_densities = np.array(atmospheric_densities, dtype=float)  # (N,) kg/m^3
        self.star_mass = float(star_mass)                                  # Msun
        self.semi_major_axes = np.array(semi_major_axes, dtype=float)      # (N,) AU
        self.aphelion_angles = np.array(aphelion_angles, dtype=float)      # (N,) rad
        self.eccentricities = np.array(eccentricities, dtype=float)        # (N,) dimensionless
        self.initial_positions = np.array(initial_positions, dtype=float)  # (2, N) AU
        self.initial_velocities = np.array(initial_velocities, dtype=float)# (2, N) AU/yr

        self.number_of_planets = int(self.masses.shape[0])
        self.seed = int(seed)  # store seed so you can reproduce the same system later

    def verify_planet_positions(self, simulation_duration, planet_positions):
        planet_positions = np.asarray(planet_positions, dtype=float)
        if simulation_duration <= 0:
            raise ValueError("simulation_duration must be positive")
        if planet_positions.shape[0] != 2:
            raise ValueError("planet_positions must have shape (2, N, T)")
        if planet_positions.shape[1] != self.number_of_planets:
            raise ValueError("planet_positions must have shape (2, N, T)")
        if planet_positions.shape[2] < 2:
            raise ValueError("planet_positions must have at least two time steps")
        return True


def generate_system(n_planets=5, seed=None, star_mass=1.0,
                    a_range=(0.5, 8.0), e_range=(0.0, 0.25)):
    """
    Generates a reproducible random system.

    Units:
      - distances: AU
      - time: years
      - masses: Msun
      - velocities: AU/yr
      - radii: km
      - atmospheric_densities: kg/m^3

    Initial conditions:
      - Places each planet at aphelion (r = a(1+e)) at angle aphelion_angle.
      - Tangential velocity magnitude at aphelion from vis-viva:
          v_ap = sqrt(mu * (1 - e) / (a * (1 + e)))
        where mu = G_SOL * star_mass.
    """
    if n_planets > 5:
        raise ValueError("n_planets must be <= 5.")
    # If no seed is provided, generate one and *store it* for reproducibility
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)

    rng = np.random.default_rng(seed)
    mu = G_SOL * star_mass

    # Semi-major axes (AU): log-uniform looks nicer than uniform for wide ranges
    a_min, a_max = a_range
    semi_major_axes = np.exp(rng.uniform(np.log(a_min), np.log(a_max), size=n_planets))

    # Eccentricities: uniform in a small range (keep < 1)
    e_min, e_max = e_range
    eccentricities = rng.uniform(e_min, e_max, size=n_planets)

    # Aphelion angles (rad): random orientation in the plane
    aphelion_angles = rng.uniform(0.0, 2*np.pi, size=n_planets)

    # Planet masses (Msun): log-uniform around Earth-ish to Jupiter-ish (tweak as you like)
    # Earth ~ 3e-6 Msun, Jupiter ~ 9.5e-4 Msun
    masses = np.exp(rng.uniform(np.log(3e-7), np.log(1e-3), size=n_planets))

    # Radii (km): very rough scaling (not physically exact), just to populate atmosphere model
    # Clamp to something reasonable
    radii = np.clip(6371.0 * (masses / 3e-6)**(1/3), 1500.0, 80000.0)

    # Atmospheric surface densities (kg/m^3): random-ish plausible range
    atmospheric_densities = rng.uniform(0.0, 5.0, size=n_planets)

    # Initial positions/velocities at aphelion
    r_ap = semi_major_axes * (1.0 + eccentricities)  # AU
    cos_t = np.cos(aphelion_angles)
    sin_t = np.sin(aphelion_angles)

    initial_positions = np.vstack([r_ap * cos_t, r_ap * sin_t])  # (2, N)

    # Speed at aphelion (AU/yr)
    v_ap = np.sqrt(mu * (1.0 - eccentricities) / (semi_major_axes * (1.0 + eccentricities)))

    # Tangential direction (counterclockwise): (-sin, cos)
    initial_velocities = np.vstack([-v_ap * sin_t, v_ap * cos_t])  # (2, N)

    return System(
        masses=masses,
        radii=radii,
        atmospheric_densities=atmospheric_densities,
        star_mass=star_mass,
        semi_major_axes=semi_major_axes,
        aphelion_angles=aphelion_angles,
        eccentricities=eccentricities,
        initial_positions=initial_positions,
        initial_velocities=initial_velocities,
        seed=seed,
    )


def main():
    """
    Optional: quick smoke-test / example usage.
    This does NOT run on import; only when executing this file directly.
    """
    system = generate_system(n_planets=8, seed=5289)
    print(f"Generated system with N={system.number_of_planets}, seed={system.seed}")


if __name__ == "__main__":
    main()
