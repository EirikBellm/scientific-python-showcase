import numpy as np

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
    Sm = system.star_mass


    a0 = system.semi_major_axes[0]
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
    nr = system.number_of_planets
    r = np.zeros((nr, 2))
    v = np.zeros((nr, 2))
    for i in range(system.number_of_planets):
        r[i,:] = system.initial_positions[:, i] #AU
        v[i,:] = system.initial_velocities[:, i] # AU/year
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

    G = const.G_sol
    Sm = system.star_mass

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


    G = const.G_sol
    Sm = system.star_mass


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
    AA = system.aphelion_angles[planet_idx] + np.pi # Justerer slik at Aphelion angle blir Perehelion angle.
    e = system.eccentricities[planet_idx]
    a = system.semi_major_axes[planet_idx]
    theta = np.linspace(0, 2*np.pi, 1000)           # Array fra 0 - 2pi
    f_arr = theta - AA                              # Roterer f med samme vinkel som banene
    r_arr = a*(1 - e**2) / (1 + e * np.cos(f_arr))  # r(f)

    # Fra polar til kartesisk
    x = r_arr * np.cos(theta) 
    y = r_arr * np.sin(theta) 
    return((x, y))