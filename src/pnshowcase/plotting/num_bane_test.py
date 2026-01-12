import numpy as np

# ---------------------------
# Solar-system-like setup (AU, years, solar masses)
# ---------------------------
G_sol = 4 * np.pi**2          # AU^3 / (yr^2 * M_sun)
star_mass = 1.0               # M_sun

# 8 planets: Mercury ... Neptune (approx J2000-ish)
semi_major_axes = np.array([  # AU
    0.387098, 0.723332, 1.000000, 1.523679,
    5.2044,   9.5826,   19.2184,  30.11
], dtype=float)

eccentricities = np.array([
    0.2056, 0.0067, 0.0167, 0.0934,
    0.0489, 0.0565, 0.0457, 0.0113
], dtype=float)

# Keep it simple: align all orbits (angle in radians).
# (Your analytic routine treats this as "aphelion angle".)
aphelion_angles = np.zeros_like(semi_major_axes)

number_of_planets = len(semi_major_axes)

# Initial conditions: start each planet at perihelion on +x axis,
# with velocity in +y direction at perihelion (AU/yr).
r0 = semi_major_axes * (1.0 - eccentricities)
v0 = np.sqrt(G_sol * star_mass * (1.0 + eccentricities) / (semi_major_axes * (1.0 - eccentricities)))

initial_positions = np.zeros((2, number_of_planets), dtype=float)
initial_velocities = np.zeros((2, number_of_planets), dtype=float)

initial_positions[0, :] = r0           # x positions (AU)
initial_positions[1, :] = 0.0          # y positions (AU)
initial_velocities[0, :] = 0.0         # vx (AU/yr)
initial_velocities[1, :] = v0          # vy (AU/yr)


def set_integration_parameters():
    '''
    Forklaring:
    -----------
    Definerer integrasjons parametre. Bruker planet 0 sin periode som år og bruker 50 slike.

    Parametere:
    -----------
    ingen

    Output:
    -------
    year: float: planet 0 sin periode i jord-år
    t_int: float: totaltid i jord-år
    steps: int: antall integrasjonssteg
    dt: float: størelsen på ett tidssteg i jordår
    '''
    Sm = star_mass


    a0 = semi_major_axes[0]
    year = np.sqrt( (a0**3) / Sm )   # keplers 3, justert for vår "solmasse"
    t_int = 50 * year
    steps = int(50 * 10000)
    dt = t_int / steps

    return(year, t_int, steps, dt)

def get_state():
    '''
    Forklaring:
    -----------
    Henter initialtilstand for alle planetene i et system . 
    Lager r og v arrays og lagrer initialtilstanden til hver planet .

    Parametere:
    -----------
    Ingen.

    Output:
    -------
    r : ndarray shape (nr, 2)
        Startposisjoner for stjerne og planet i AU.
    v : ndarray shape (nr, 2)
        Starthastigheter for stjerne og planet i AU/år.
    '''
    nr = number_of_planets
    r = np.zeros((nr, 2))
    v = np.zeros((nr, 2))
    for i in range(number_of_planets):
        r[i,:] = initial_positions[:, i] #AU
        v[i,:] = initial_velocities[:, i] # AU/year
    return(r, v)

def leapfrog_one_step(r, v, dt):
    '''
    Utfører ett leapfrog-steg

    Parametre
    ---------
    r : ndarray, shape (N, 2)
        Nåværende posisjoner til legemene i AU målt fra sol/origo.
    v : ndarray, shape (N, 2)
        Nåværende hastigheter på halvsteg AU/år.
    dt : float
        Tidssteg i år

    Returnerer
    ----------
    r : ndarray, shape (N, 2)
        Oppdaterte posisjoner ved t + dt.
    v : ndarray, shape (N, 2)
        Oppdaterte hastigheter etter nytt akselerasjonssteg.
    a : ndarray, shape (N, 2)
        Akselerasjoner beregnet ved de nye posisjonene r.

    '''

    G = G_sol
    Sm = star_mass

    r = r + v * dt
    a = -G * Sm * r / np.linalg.norm(r, axis = 1)[:, np.newaxis]**3
    v = v + a * dt
    return(r, v, a)

def planet_integration():
    '''
    Integrer systemet over [0, T] med leapfrog og lagre resultat til fil.

    Returnerer
    ----------
    None
    men:
    Skriver en fil "orbits.npz" som inneholder:
    - r : posisjonsarray over tid
    - v : hastighetsarray over tid
    - a : akselerasjonsarray over tid
    - dt : skalar tidssteg

    '''
    year, t_int, steps, dt = set_integration_parameters()
    r, v = get_state()


    G = G_sol
    Sm = star_mass


    # Initialiserer V halvsteg for leapfrog integrasjon.
    a = -G  * Sm * r / np.linalg.norm(r, axis = 1)[:, np.newaxis]**3
    v_halv = v + a * (1/2) * dt

    # Initialiserer arrays
    shape = (steps + 1, *r.shape)
    r_arr, v_arr, a_arr = (np.zeros(shape) for _ in range(3))
    r_arr[0], v_arr[0], a_arr[0] = r, v, a

    for i in range(steps):
        r, v_halv, a = leapfrog_one_step(r, v_halv, dt)

        # Lagrer verdier for plotting
        r_arr[i+1] = r
        a_arr[i+1] = a
        v_arr[i+1] = v_halv - 0.5*dt*a # Lagrer fart ved helt tidsstedg.
    
    np.savez_compressed(
        "orbits.npz",
        r=r_arr,        
        v=v_arr,        
        a=a_arr,         
        dt=dt
        )

#trenger bare runne dette en gang
planet_integration()

def get_anal_orbit(planet_idx):
    '''
    Forklaring:
    -----------
    Tar in planetnummeret, henter tilhørende info fra system, regner ut analytiske baner og  gir bane arrays .

    Parametere:
    -----------
    planet_idx: Planetnummeret slik det er lagret i vår system instanse.

    Output:
    -------
    (x, y): en tupple med x og y array hver med 1000 verdier og representerer koordinater til banen.
    '''
    AA = aphelion_angles[planet_idx] + np.pi # Justerer slik at Aphelion angle blir Perehelion angle.
    e = eccentricities[planet_idx]
    a = semi_major_axes[planet_idx]
    theta = np.linspace(0, 2*np.pi, 1000)           # Array fra 0 - 2pi
    f_arr = theta - AA                              # Roterer f med samme vinkel som banene
    r_arr = a*(1 - e**2) / (1 + e * np.cos(f_arr))  # r(f)

    # Fra polar til kartesisk
    x = r_arr * np.cos(theta) 
    y = r_arr * np.sin(theta) 
    return((x, y))
