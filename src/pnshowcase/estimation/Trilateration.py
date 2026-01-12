##########################
### IKKE BRUKT KODEMAL ###
##########################

###################################################################################################
# FILAVHENGIGHETER 
#
# 1) orbits.npz   (OBLIGATORISK)
#    - Plassering: samme mappe som dette skriptet kjøres fra (working directory).
#    - Kommer fra: del2/Numerisk_bane_integrasjon.py 
###################################################################################################

import math
import numpy as np
from ast2000tools.solar_system import SolarSystem
seed = 5289
system = SolarSystem(seed)


#--- Henter data for planetenes posisjon fra integrasjonen i del 2 av prosjektet---
data = np.load("orbits.npz")
bane = data["r"]  #form: [tid, planet_nr, (x, y)]
dt = float(data['dt'])

# --- Bruker matriselikning til å finne rakettens posisjon med trilateration-metode ---
def trilateration(tid, avstander, planeter):
    """
    Forklaring:
    -----------
    Finner rakettens posisjon med trilateration med planeter/stjerne i systemet 

    Parametre:
    ----------
    Bestemt tid, målte avstander og liste over hvilke planeter som brukes 

    Returnerer:
    -----------
    Rakettens posisjon og residualene (feil)
    """ 
    A = []
    B = []

    #--- Bruker stjernen i systemet som referanse ---
    x0, y0 = 0, 0 #antar at stjernen er i ro og har dermed alltid posisjon (0,0)
    r0 = avstander[-1]  #alltid siste element

    # --- Løkke over resten av planetene som finner elementene til matrisen A og B ---
    for idx, p_nr in enumerate(planeter):


        ###################################################################
        #################### AVIK FRA FLYTKART ############################
        ###### Linjær interpolerer hvis t ikke er ett heltall av dt #######
        # Runder ned t
        t_ned = math.floor(tid/dt)
        # Finner hvor langt mellom de to punktene vi er
        rest = (tid - t_ned*dt) / dt

        xy = (1 - rest) * bane[t_ned + 1, p_nr, :] + rest * bane[t_ned, p_nr, :] #henter planetposisjoner med banesimuleringen fra del 2 av prosjektet med liniær interpolering

        xi, yi = float(xy[0]), float(xy[1])

        ri = avstander[idx]  #henter tilhørende avstand (radius i sikelen som dannes)

        Ai = [-2 * (xi - x0), -2 * (yi - y0)] #matrise A 
        Bi = (ri**2 - r0**2) - ((xi**2 - x0**2) + (yi**2 - y0**2)) #B

        A.append(Ai)
        B.append(Bi)

    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    #--- Løsning x av matriselikningen Ax=B ---
    pos, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None) #bruker minste kvadraters metode for å finne den beste løsningen av x
    x_est, y_est = pos

    return (x_est, y_est), residuals


#--- Definerer tid, planet numre og målte avstander ---
tid = 1 #setter inn tiden målingene ble gjort på slik at banesimuleringen blir riktig (vilkårlig tall her)
planeter = [0, 1, 2] #velger planetnummerene vi har målinger til (vilkårlige tall her)
avstander = [3, 14, 5] #setter inn målte avstander (vilkårlige tall her)

#--- Kaller på matrise-løsningen og printer posisjon og residualer (feil) ---
pos, residualer = trilateration(tid, avstander, planeter)
print(f'Estimert posisjon til rakett ved tid {tid}: {pos}')
print(f'Residualer: {residualer}')




#--- Tester trilaterasjon metoden med kjente posisjoner ---
def test_trilaterasjon(antall_planeter, n_tester=100, pos_range=50):
    """
    Forklaring:
    -----------
    Tester trilaterasjon metoden med 4 planeter 100 ganger med tilfeldig valgte ekte posisjoner og tider. 

    Parametre:
    ----------
    Antallet planeter som skal brukes spom referansepunkter, antall tester og intervallet til posisjons-komponentene.

    Returnerer:
    -----------
    Ingenting, men printer gjennomsnittlig relativ feil fra de 100 testene. 
    """

    feil_liste = []

    ##### AVVIK FRA FLYTKART: #####
    støy_prosent = 0.01 #legger til støy i avstandmålingene for å sjekke realistisk relativ feil
    ###############################

    #--- Gjennomfører testen 100 ganger ---
    for i in range(n_tester): 
        #--- Genererer tilfeldige posisjoner, tider og planet numre for hver test ---
        ekte_pos = np.random.uniform(-pos_range, pos_range, size=2) #tilfeldig posisjon innenfor intervallet
        tid = np.random.choice(np.arange(50)) #tilfeldig tidspunkt mellom 0 og 50 jordår 
        planeter = np.random.choice(np.arange(8), size=antall_planeter, replace=False) #ikke med tilbakelegging da en planet ikke kan brukes flere ganger

        avstander = []
        
        #--- Finner avstander til alle referansepunktene ---
        for i in planeter:
            planet_pos = bane[int(tid/dt), i, :] #bruker banesimuleringen fra del 2 av prosjektet (antar her at den er korrekt)
            avstand = np.linalg.norm(planet_pos - ekte_pos) #finner avstand mellom rakett og planet med vektorregning
            stoy = np.random.normal(0, støy_prosent * avstand) #AVVIK FRA FLYTKART
            avstander.append(avstand + stoy)
        
        avstand_stjerne = np.linalg.norm(ekte_pos - np.array([0, 0])) #finner avstand til stjerne 
        støy_stjerne = np.random.normal(0, støy_prosent * avstand_stjerne)
        avstander.append(avstand_stjerne + støy_stjerne)
        avstander = np.array(avstander) #gjør om til array slik at man kan regne ut gjennomsnittlig relativ feil

        #--- Beregner den estimerte posisjonen med trilaterasjon-metoden ---
        estimert_pos, _ = trilateration(tid, avstander, planeter) #kaller på trilateration funksjon for å hente estimerte posisjoner

        #--- Beregner relativ feil for alle testene ---
        rel_feil = (np.linalg.norm((ekte_pos-estimert_pos)) / (np.linalg.norm(ekte_pos))) * 100
        feil_liste.append(rel_feil)

    gjennomsnittlig_feil = np.mean(feil_liste) #finner gjennomsnittet av alle testenes relative feil

    print(f'Kjørte {n_tester} tester med {antall_planeter} tilfeldige planeter ved tilfeldige tider.')
    print(f'Gjennomsnittlig relativ feil: {gjennomsnittlig_feil}%')

#--- Kjører trilaterasjon testen med bestemt antall planeter (referansepunkter) ---
antall_planeter = 6 #velger 3 (2+1 med stjernen) fordi det er minste kravet for å få ett skjæringspunkt, kan også velge flere for økt nøyaktighet
test_trilaterasjon(antall_planeter)