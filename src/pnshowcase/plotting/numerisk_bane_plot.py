
import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from scipy.signal import find_peaks
seed = 5289
system = SolarSystem(seed)

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

def lesbar_plott():
    '''
    Forklaring:
    -----------
    Oppdaterer plott-setting for bedre lesbarhet.

    '''

    plt.rcParams.update({
        "axes.labelsize": 15,
        "axes.titlesize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 10,
    })

def plot(x_y, idx):
    '''
    Forklaring:
    -----------
    Tar in tupple med x og y arrays og plotter.

    Parametere:
    -----------
    x_y: en tupple med x og y arrays. må vøre samme lengde.
    planet_idx: Planetnummeret slik det er lagret i vår system instanse. Brukes for lable

    Output:
    -------
    ingen outputt. Bare plotter.

    '''
    lesbar_plott()
    x, y = x_y
    posx, posy = system.initial_positions[:, idx]
    plt.plot(x, y, label = f"Planet nr.{idx} bane", linestyle = "-")
    plt.scatter(posx, posy, s = 20, label = f"Initiell posisjon planet nr.{idx}")
    
    return

def plot_num_og_ana():
    '''
    Forklaring:

    Laster numeriske baner fra «orbits.npz» og plotter dem sammen med analytiske baner
    (for alle planeter i system). Viser også stjernens posisjon i (0, 0). Figuren får
    akse-etiketter i AU oglik skala på x/y.

    Parametere:

    None

    Forutsetninger
        -   Filen «orbits.npz» finnes i mappen.
        -   r: posisjoner med form (T, N, 2) (tid, planet, x/y)

    Retur

    None

    '''
    
    data = np.load("orbits.npz")
    bane = data["r"]


    N = system.number_of_planets
    for i in range(N):
        tuple = get_anal_orbit(i)
        plot(tuple, i)
        plt.plot(bane[:, i, 0], bane[:, i, 1], linestyle=(0, (16, 10)), label=f"Planet nr.{i} (simulert)", lw = 3)

    plt.scatter(0, 0, s=100, c="yellow", marker="o", edgecolors="black", label="Star")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.axis("equal")
    plt.title("Analytiske baner sammenliknet med simulerte baner")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

plot_num_og_ana()

def plott_r_av_r():
    '''
    Forklaring 
    -----------
    Plotter r(t) for alle planeter.
  
    Retur
    -----
    None

    '''
    data = np.load("orbits.npz")
    bane = data["r"]
    _, t_int, steps, _ = set_integration_parameters()
    tid = np.linspace(0, t_int, steps + 1)
    N = system.number_of_planets
    for i in range(N):
        r_t = np.linalg.norm(bane[:, i, :], axis=1)[:, np.newaxis]
        plt.plot(tid, r_t, label=f'Planet nr.{i} med eksentrisitet {system.eccentricities[i]:.5f} ')
    plt.xlabel("Tid [år]")
    plt.ylabel("Avstand fra stjerne [AU]")
    plt.title("Planetenes posisjon som funksjon av tid")
    plt.legend()
    plt.tight_layout()
    plt.show()

plott_r_av_r()


def kep_2_test():
    """
    Forjklaring:
    ------------
    Tester Keplers 2. lov (like arealer på like tider) ved å sammenligne arealet
    som r sveiper ut nær perihel og aphel.

    Forutsetter:
    ------------
    - Filen "orbits.npz" med"r" (posisjoner).

    Returnerer:
    -----------
    - Ingenting. Printer et sammendrag til stdout.
    """
    data = np.load("orbits.npz")
    bane = data["r"]
    v = data["v"]
    _, t_int, steps, dt = set_integration_parameters()
    tid = np.linspace(0, t_int, steps)

    #
    r_norm = np.linalg.norm(bane[:, 0, :], axis = 1)
    v_norm = np.linalg.norm(v[:, 0, :], axis = 1)
    r_min = np.argmin(r_norm) # perihel-indeks
    r_max = np.argmax(r_norm) # aphel-indeks


    def areal(t0,t_intervall):
        # t0 = t0 -0.5 * t_intervall
        areal = 0
        for i in range(int(t0), int(t0 + t_intervall + 1)):
            delta_t = 1
            r = (bane[int(i), 0, :])
            r_delta_t = (bane[int(i) + delta_t, 0, :])

            # Trekanten mellom r(i) og r(i+dt): |(r x r_next)|/2
            areal += np.linalg.norm(0.5 * np.cross(r, r_delta_t))
        
        v_i_dt = v_norm[ int(t0) : int(t0 + t_intervall + 1)]
        avstand_i_dt = v_i_dt * dt
        avstand_dekket = np.sum(avstand_i_dt)
        v_mean = np.mean(v_i_dt)
        return areal, avstand_dekket, v_mean
    
    # Sammenlign like tidsintervaller rundt perihel og aphel
    areal_rmin, avstand_rmin, mean_v_rmin =  areal(tid[r_min], int(200))
    areal_rmax, avstand_rmax, mean_v_rmax  = areal(tid[r_max], int(200))


    print(f'Areal sveipet over fra t0 til t1 ved perihelion: {areal_rmin} AU^2  -  Avstand tilbakelagt: {avstand_rmin} - gjenomsnittelig fart er {mean_v_rmin} ')
    print(f'Areal sveipet over fra t0 til t1 ved aphelion: {areal_rmax} AU^2  -  Avstand tilbakelagt: {avstand_rmax}- gjenomsnittelig fart er {mean_v_rmax}')


#Finner periode og semi-major-axis for å sammenlikne med Keplers 3. lov:
def periode(i):
    """
    Beregner den numeriske omløpsperioden og orbitalparametere for en valgt planet
    basert på simulerte posisjonsdata, og sammenligner deretter resultatene med
    teoretiske verdier fra Keplers 3. lov (Newtons form).

    Parametre:
    ----------
    i : int
        Indeksen til planeten i bane-arrayen.

    Returnerer:
    -----------
    tuple
        (P_num, a, e, P_newton)
        hvor
          P_num   : numerisk estimert omløpsperiode [år]
          a       : stor halvakse [AU]
          e       : eksentrisitet [-]
          P_newton: teoretisk periode fra Newtons formel [år]

    """
    G = const.G_sol
    Sm = system.star_mass

    data = np.load("orbits.npz")
    bane = data["r"]

    _, t_int, steps, dt = set_integration_parameters()
    tid = np.linspace(0, t_int, steps)

    r_norm = np.linalg.norm(bane[:, i, :], axis = 1)
    # Finn topper i r(t) og estimer periode fra gj.snittlig tidsdifferanse
    topper, _ = find_peaks(r_norm)
    periode = np.mean(np.diff(tid[topper]))

    # Ekstremavstander
    r_max = np.max(r_norm)
    r_min = np.min(r_norm)

    # Halvaksene og eksentrisitet (ellipse): a = (r_max + r_min)/2, b = sqrt(r_max*r_min)
    a = (r_max + r_min) / 2
    b = np.sqrt(r_max*r_min)
    e = np.sqrt(1 - (b/a)**2)
 
    # Teoretisk periode (Newtons form av Keplers 3. lov)
    pn = np.sqrt((4*((np.pi)**2) * a**3 / (G*(system.masses[i] + Sm))))

    return periode, a, e, pn 

G = const.G_sol
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
    print(f'Perioden til planet {i}: {P_num:.2f} jordår')
    print(f'Avvik fra Keplers 3. lov: {avvik_P:.5f}%')
    print(f'Avviket mellom analytisk a og numerisk a er: {avvik_a:.5f} % ')
    print(f'Avviket mellom analytisk e og numerisk e er: {avvik_e:.5f} % ')
    print(f'Avvik mellom Keplers P og Newtons P er: {avvik_PN:.5f} %')


def verify_orbit():
    _, t_int, steps, _ = set_integration_parameters()
    data = np.load("orbits.npz")
    r = data["r"]                       # (T, planeter, 2)
    arr = np.transpose(r, (2, 1, 0))    # endrer til (2, planeter, T)

    # Tar ut et mindre antall tidspunkter for å teste verifikasjon
    points = 5000   
    arr = arr[:, :, np.linspace(0, steps-1, points, dtype=int)]

    # Kall
    system.verify_planet_positions(
        simulation_duration=t_int,   # total varighet i år
        planet_positions=arr         # må ha form (2, antall_planeter, antall_tidspunkter)
    )

verify_orbit()