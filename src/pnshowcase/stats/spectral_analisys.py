##############################
##### IKKE BRUKT KODEMAL #####
##############################

##############################
#### NB denne filen må kjøres#
# med en /data mappe som har #
# noise og fluks data i seg  #
# for 89 som to siste siffer #
#  i seed                    #
###############################


import csv
from pathlib import Path
import ast2000tools.constants as konst
import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------------------------
#                  FYSIKK OG STATISTIKK
#-----------------------------------------------------------

def dopplerskift(bølge0, maks_fart, lysfart):
    """
    Regner ut maksimal Doppler-forskyvning i bølgelengde for en gitt linje.

    Parametre
    ---------
    bølge0 : float
        Lambda0 for linjen nm.
    maks_fart : float
        Øvre grense for radialhastighet |v| [m/s].
    lysfart : float
        Lyshastighet c [m/s].

    Returnerer
    ----------
    float
        Maksimalt bølgelengdeskift Delta-Lambda i [nm].
    """
    return (maks_fart / lysfart) * bølge0


def dopplerskift_til_fart(delta_lamda, bølge0, lysfart):
    """
    Gjør om en Doppler-forskyvning i bølgelengde til en radialhastighet.
        Parametre
    ---------
    delta_lambda : float 
        bølgelengdeskift fra lambda 0 i nm.
    bølge0 : float
        Lambda0 for linjen nm.
    lysfart : float
        Lyshastighet c [m/s].

    Returnerer
    ----------
    float
        radialfart for tilhørende doplershift i m/s.
    """
    return (delta_lamda / bølge0) * lysfart


def hent_arrayer(
    bølge0,
    bølgelengder,
    fluks,
    noise,
    maks_fart,
    lysfart,
    dopplerbuffer,
):
    """
    Klipper ut et intervall rundt en forventet linje.

    Parametre
    ---------
    bølge0 : float
        Senterbølgelengde for linja [nm].
    bølgelengder : (N,) array
        Akse for spektrum [nm]
    fluks : (N,) array
        Normalisert fluks 
    noise : (N,) array
        Standardavvik (sigma) per datapunkt, i samme enhet som fluks.
    maks_fart : float
        Øvre grense på |v_r| [m/s] brukt til intervallstørrelse.
    lysfart : float
        c [m/s].
    dopplerbuffer : float
        > 1.0 for å gi litt ekstra margin rundt teoretisk maks skift.

    Returnerer
    ----------
    (lambda_intervall, fluks_intervall, noise_intervall) : tuple av arrays
    """
    ###################################
    ######## AVVIK FRA FLYTKART #######
    # la til dopler buffer for at vi  #
    # er sikre på at vi finner linja  #
    ###################################

    skift = dopplerskift(bølge0, maks_fart, lysfart)
    intervall = skift * dopplerbuffer
    start_lamda = bølge0 - intervall
    slutt_lamda = bølge0 + intervall

    # Finner intervallets indekser
    start_indeks = int(np.searchsorted(bølgelengder, start_lamda, side="left"))
    slutt_indeks = int(np.searchsorted(bølgelengder, slutt_lamda, side="right"))

    return (
        bølgelengder[start_indeks:slutt_indeks],
        fluks[start_indeks:slutt_indeks],
        noise[start_indeks:slutt_indeks],
    )


def chi_kvadrat(
    fluks_intervall,
    noise_intervall,
    lambda_intervall,
    temperatur_grid,
    forskyvning_grid,
    fmin_grid,
    bølge0,
    masse,
    lysfart,
):
    """
    Brute-force CHI^2-minimering over 3D rutenett (T, Delta-Lambda, fmin).

    METODE:
    - Vi vektoriserer 3D rutenettet og regner ut modell(Lambda) for alle kombinasjoner.
    - CHI^2 = sigma ((data - modell) / sigma)^2, summeres over bølgelengdepunkter.
    - Beste indeks gir estimerte parametre (T_best, Delta-Lambda_best, fmin_best).

    Returnerer
    ----------
    temperatur_best, forskyvning_best, fmin_best, chi2_min
    """
    fluks_intervall = np.asarray(fluks_intervall, dtype=float)
    noise_intervall = np.asarray(noise_intervall, dtype=float)
    lambda_intervall = np.asarray(lambda_intervall, dtype=float)

    # Vektorisering: L = antall bølgelengdepunkter, og rutenettet er
    # (n_T, n_shift, n_fmin). Vi gjør derfor data-arrays 4D slik at
    # aksene følger [T, Delta-Lambda, fmin, lambda]:
    fluks = fluks_intervall[np.newaxis, np.newaxis, np.newaxis, :]
    noise = noise_intervall[np.newaxis, np.newaxis, np.newaxis, :]
    lamda = lambda_intervall[np.newaxis, np.newaxis, np.newaxis, :]

    # Parametrene legges på de første tre aksene.
    # temperatur: shape (n_T, 1, 1, 1), forskyvning: (1, n_shift, 1, 1),
    # fmin: (1, 1, n_fmin, 1). Resultatene av operasjoner på disse blir automatisk
    # broadcastet over alle lambda-punkter (siste akse).
    temperatur = temperatur_grid[:, np.newaxis, np.newaxis, np.newaxis] 
    forskyvning = forskyvning_grid[np.newaxis, :, np.newaxis, np.newaxis]
    fmin = fmin_grid[np.newaxis, np.newaxis, :, np.newaxis]

    # Regner ut Doppler-bredde for alle T
    sigma_lamda = bølge0 * np.sqrt(konst.k_B * temperatur / masse) / lysfart

    # Gaussen har peak=1, og linjedybden styres av fmin
    gauss = np.exp(-0.5 * ((lamda - (bølge0 + forskyvning)) / sigma_lamda) ** 2)
    modell = 1.0 - (1.0 - fmin) * gauss

    # Chi2 summerer over lambda-aksen (axis=-1), slik at chi2 får
    # shape (n_T, n_shift, n_fmin) og kan minimeres direkte.
    chi2 = np.sum(((fluks - modell) / noise)**2, axis=-1)

    beste_indeks = np.unravel_index(np.argmin(chi2), chi2.shape)
    return (
        temperatur_grid[beste_indeks[0]],
        forskyvning_grid[beste_indeks[1]],
        fmin_grid[beste_indeks[2]],
        chi2[beste_indeks],
    )

#-----------------------------------------------------------
#                    DATABEHANDLING
#-----------------------------------------------------------


def finn_parametre(gassnavn, bølge0, rutenett, oppsett):
    """
    #####################################################
    ####### Denne funksjonen brukes ikke lenger #########
    #####################################################

    Finner beste T, Doppler-forskyvning og fmin for én linje ved brute-force CHI^2.

    1) Henter intervall rundt Lambda0 .
    2) Setter opp 1D rutenett for T, Delta-Lambda og fmin (alle med samme 'rutenett'-oppløsning).
    3) Kjører CHI^2 og returnerer beste verdier.
    """
    lambda_intervall, fluks_intervall, noise_intervall = hent_arrayer(
        bølge0,
        oppsett["bølgelengder"],
        oppsett["fluks"],
        oppsett["noise"],
        oppsett["maks_fart"],
        oppsett["lysfart"],
        oppsett["dopplerbuffer"],
    )

    # Setter temperaturgrid [K]
    temperatur_grid = np.linspace(
        oppsett["temperaturomrade"][0],
        oppsett["temperaturomrade"][1],
        rutenett,
    )

    # Setter Delta-Lambda-grid fra +- maks Doppler-skift
    maks_skift = dopplerskift(bølge0, oppsett["maks_fart"], oppsett["lysfart"])
    forskyvning_grid = np.linspace(-maks_skift, maks_skift, rutenett)

    # Setter fmin-grid: 0< fmin ≤1 (1 = ingen absorpsjon; lavere = dypere linje)
    fmin_grid = np.linspace(
        oppsett["fmin_omrade"][0],
        oppsett["fmin_omrade"][1],
        rutenett,
    )

    masse = oppsett["molekylmasser"][gassnavn]

    temperatur_best, forskyvning_best, fmin_best, chi2_min = chi_kvadrat(
        fluks_intervall,
        noise_intervall,
        lambda_intervall,
        temperatur_grid,
        forskyvning_grid,
        fmin_grid,
        bølge0,
        masse,
        oppsett["lysfart"],
    )

    return {
        "gass": gassnavn,
        "bølge0": bølge0,
        "temperatur": temperatur_best,
        "delta_lamda": forskyvning_best,
        "fmin": fmin_best,
        "chi2": chi2_min,
    }


def finn_parametre_iterativt(gassnavn, bølge0, rutenett, oppsett, antall=10, faktor=0.4):
    """
    Utfører flere CHI^2-søk med gradvis innsnevring av parameterområdene rundt besteverdien.

    ###########################################
    ########### AVVIK FRA FLYTKART ############
    # Denne funksjonen gjør ca det samme som  #
    # finn_parametre, men iterativt, slik at  #
    # vi kan forbedre nøyaktighet på en mer   #
    # tidseffektiv måte, vs å øke grid.       #
    ###########################################

    Parametre
    ---------
    gassnavn : str
        Identifikator for linjen, brukes til å hente masse og globale intervaller.
    bølge0 : float
        Senterbølgelengde [nm] for linjen som analyseres.
    rutenett : int
        Antall gridpunkter pr. dimensjon i hvert brute-force søk.
    oppsett : dict
        Samler spektrum, støy og fysiske konstanter (samme struktur som i finn_parametre).
    antall : int, optional
        Hvor mange iterasjoner som kjøres. Standard er 10.
    faktor : float, optional
        Hvor mye intervallet skaleres per trinn (0 < faktor < 1). Standard 0.4.

    Returnerer
    ----------
    dict
        Resultat med beste T, Δλ, fmin og χ² fra siste iterasjon.
 """

    # Henter utsnitt av spekteret rundt aktuell bølgelengde:
    lambda_intervall, fluks_intervall, noise_intervall = hent_arrayer(
        bølge0,
        oppsett["bølgelengder"],
        oppsett["fluks"],
        oppsett["noise"],
        oppsett["maks_fart"],
        oppsett["lysfart"],
        oppsett["dopplerbuffer"],
    )

    # Henter molekylmasse for valgt gass
    masse = oppsett["molekylmasser"][gassnavn]

    # Globale søkeområder for temperatur, dopplerskift og F_min
    temp_global = oppsett["temperaturomrade"]
    fmin_global = oppsett["fmin_omrade"]

    # Maksimalt dopplerskift
    maks_skift = dopplerskift(bølge0, oppsett["maks_fart"], oppsett["lysfart"])
    shift_global = (-maks_skift, maks_skift)

    # Starter med å søke over hele globale intervaller
    temp_range = temp_global
    shift_range = shift_global
    fmin_range = fmin_global

    # Holder på beste resultat fra siste iterasjon
    beste_resultat = None

    # Iterativt CHI^2-søk: grovt først, så snevrer vi inn rundt beste verdi
    for steg in range(antall):
        # Lager 1D-rutenett for hver parameter innenfor gjeldende intervall
        temperatur_grid = np.linspace(temp_range[0], temp_range[1], rutenett)
        forskyvning_grid = np.linspace(shift_range[0], shift_range[1], rutenett)
        fmin_grid = np.linspace(fmin_range[0], fmin_range[1], rutenett)

        # Finner parameterkombinasjonen som gir lavest chi^2 innenfor rutenettet
        (
            temperatur_best,
            forskyvning_best,
            fmin_best,
            chi2_min,
        ) = chi_kvadrat(
            fluks_intervall,
            noise_intervall,
            lambda_intervall,
            temperatur_grid,
            forskyvning_grid,
            fmin_grid,
            bølge0,
            masse,
            oppsett["lysfart"],
        )

        # Lagrer beste funn fra denne iterasjonen
        beste_resultat = {
            "gass": gassnavn,
            "bølge0": bølge0,
            "temperatur": temperatur_best,
            "delta_lamda": forskyvning_best,
            "fmin": fmin_best,
            "chi2": chi2_min,
        }

        # På alle iterasjoner unntatt den siste snevrer vi inn søkeintervallene
        # rundt de funnede besteverdiene for neste søk.
        if steg < antall - 1:
            temp_range = snevre_inn(temperatur_best, temp_range, temp_global, faktor)
            shift_range = snevre_inn(forskyvning_best, shift_range, shift_global, faktor)
            fmin_range = snevre_inn(fmin_best, fmin_range, fmin_global, faktor)
    return beste_resultat


def snevre_inn(besteverdi, gjeldende, globalt, faktor):
    """
    Snevrer inn et intervall rundt en ny besteverdi uten å gå utenfor globale grenser.

    #############################################
    ########## AVVIK FRA FLYTKART ###############
    # Denne funksjonen la vi til for å forbedre #
    # nøyaktigheten, se komentarer for tankegang#
    #############################################

    Parametre
    ---------
    besteverdi : float
        Verdien som skal ligge i sentrum av det nye intervallet.
    gjeldende : tuple(float, float)
        Nåværende intervall (min, max) som skal skaleres.
    globalt : tuple(float, float)
        Absolutte fysiske grenser intervallet må holde seg innenfor.
    faktor : float
        Skalering av intervallbredden (0 < faktor < 1 gir innsnevring).

    Returnerer
    ----------
    (nedre, øvre) : tuple(float, float)
        Oppdatert intervall innenfor det globale området.

    """
    # Nåværende bredde på intervallet (øvre - nedre grense)
    bredde = gjeldende[1] - gjeldende[0]

    # Ny, smalere bredde etter innsnevringsfaktoren
    ny_bredde = bredde * faktor

    # Halv bredde brukes for å lage et symmetrisk intervall rundt besteverdi
    halv = ny_bredde / 2.0

    # Forsøker å legge nedre grense symmetrisk rundt besteverdien,
    # men ikke utenfor det globale tillatte området
    nedre = max(globalt[0], besteverdi - halv)

    # Tilsvarende for øvre grense
    ovre = min(globalt[1], besteverdi + halv)

    return (nedre, ovre)


def søk(gass_liste, oppsett, plottmappe, skriptmappe, rutenett=25):
    """
    Kjører en full gjennomgang med finere rutenett og lagrer både plott og CSV.

    """
    resultater = []
    for gassnavn, linjer in gass_liste:
        for bølge0 in linjer:
            if bølge0 is None:
                continue
            resultat = finn_parametre_iterativt(gassnavn, bølge0, rutenett, oppsett)
            resultater.append(resultat)
            # printing, plotting og resultater
            print(
                f"{resultat['gass']:>4} lambda0={resultat['bølge0']:7.1f} nm | "
                f"T={resultat['temperatur']:7.2f} K | "
                f"Delta lambda={resultat['delta_lamda']:+8.4f} nm | "
                f"Fmin={resultat['fmin']:6.3f} | chi^2={resultat['chi2']:10.3f}"
            )
            plott(resultat, oppsett, plottmappe, lagre=True)

    # plotting og resultater
    sprednings_sti = parameter_spredning(resultater, plottmappe, lagre=True)
    csv_sti = skriv_csv(resultater, skriptmappe)
    print(f"\nLagret {len(resultater)} resultater i {csv_sti}")
    print(f"Plottene ligger i {plottmappe}")
    if sprednings_sti:
        print(f"Parameterplott lagret i {sprednings_sti}")


#-----------------------------------------------------------
#               PLOTTING OG HJELPEFUNKSJONER
#-----------------------------------------------------------

def gauss_plott(
    lambda_intervall,
    bølge0,
    delta_lamda,
    temperatur,
    fmin,
    masse,
    boltzmann,
    lysfart,
):
    """
    Denne funksjonen brukes bare til plotting.
    """
    sigma_lamda = bølge0 * np.sqrt(boltzmann * temperatur / masse) / lysfart
    # Gaussen har peak = 1 ved Lambda = Lambda0 + Delta-Lambda
    gauss = np.exp(-0.5 * ((lambda_intervall - (bølge0 + delta_lamda)) / sigma_lamda) ** 2)
    # Fluksmodellen er 1 i kontinua og fmin i sentrum (absorpsjon)
    return 1.0 - (1.0 - fmin) * gauss


def plott(resultat, oppsett, plottmappe, lagre=True):
    """
    Visualiserer data + beste modell for ett linjeintervall.

    """
    lambda_intervall, fluks_intervall, noise_intervall = hent_arrayer(
        resultat["bølge0"],
        oppsett["bølgelengder"],
        oppsett["fluks"],
        oppsett["noise"],
        oppsett["maks_fart"],
        oppsett["lysfart"],
        oppsett["dopplerbuffer"],
    )

    radial_hastighet = dopplerskift_til_fart(
        resultat["delta_lamda"],
        resultat["bølge0"],
        oppsett["lysfart"],
    )

    modell = gauss_plott(
        lambda_intervall,
        resultat["bølge0"],
        resultat["delta_lamda"],
        resultat["temperatur"],
        resultat["fmin"],
        oppsett["molekylmasser"][resultat["gass"]],
        oppsett["boltzmann"],
        oppsett["lysfart"],
    )

    plt.figure(figsize=(7, 4))
    plt.plot(lambda_intervall, fluks_intervall, label="Fluks", linewidth=1.0)
    plt.plot(lambda_intervall, modell, label="Modell", linewidth=1.5)
    plt.fill_between(
        lambda_intervall,
        1 - noise_intervall,
        1 + noise_intervall,
        color="gray",
        alpha=0.3,
        label="Støy +-sigma",
    )
    plt.axvline(
        resultat["bølge0"],
        color="k",
        linestyle="--",
        linewidth=0.8,
        label="lambda0",
    )
    plt.xlabel("Bølgelengde [nm]")
    plt.ylabel("Normalisert fluks")
    plt.title(
        f"{resultat['gass']} ved {resultat['bølge0']:.1f} nm "
        f"(v_r={radial_hastighet/1000:+.2f} km/s)"
    )
    plt.legend()
    plt.tight_layout()

    if lagre:
        plottmappe.mkdir(parents=True, exist_ok=True)
        plott_sti = plottmappe / f"{resultat['gass']}_{int(round(resultat['bølge0']))}nm.png"
        plt.savefig(plott_sti)
        plt.close()
        return plott_sti

    plott_sti = None
    try:
        plt.show()
    except Exception as feil:
        print(
            f"Klarte ikke å vise plott for {resultat['gass']} ved "
            f"{resultat['bølge0']:.1f} nm: {feil}"
        )
    finally:
        plt.close()
    return plott_sti


def parameter_spredning(resultater, plottmappe, lagre=True):
    """
    Tegner et spredningsplott med temperatur på x-aksen, radialhastighet på y-aksen
    og farger punktene etter relativ fluks (fmin) for å se mønstre.
    """
    if not resultater:
        return None

    temperaturer = [rad["temperatur"] for rad in resultater]
    hastigheter = [
        dopplerskift_til_fart(rad["delta_lamda"], rad["bølge0"], konst.c)
        for rad in resultater
    ]
    flukser = [rad["fmin"] for rad in resultater]

    plt.figure(figsize=(7, 4.5))
    spredning = plt.scatter(
        temperaturer,
        hastigheter,
        c=flukser,
        cmap="viridis",
        edgecolors="k",
        linewidths=0.4,
        s=60,
    )
    cbar = plt.colorbar(spredning)
    cbar.set_label("Relativ fluks (fmin)")
    plt.xlabel("Temperatur [K]")
    plt.ylabel("Radialhastighet [m/s]")
    plt.title("Parameterfordeling for alle linjer")
    plt.tight_layout()

    if lagre:
        plottmappe.mkdir(parents=True, exist_ok=True)
        sprednings_sti = plottmappe / "parameter_spredning.png"
        plt.savefig(sprednings_sti)
        plt.close()
        return sprednings_sti

    sprednings_sti = None
    try:
        plt.show()
    except Exception as feil:
        print(f"Klarte ikke å vise parameterplott: {feil}")
    finally:
        plt.close()
    return sprednings_sti


def skriv_csv(resultater, skriptmappe):
    """
    Lagrer alle linjeresultater som CSV for videre bruk / resultater.
    """
    csv_sti = skriptmappe / "line_search_results.csv"
    with csv_sti.open("w", newline="") as fil:
        skriver = csv.writer(fil)
        skriver.writerow(
            ["gass", "lambda_nm", "temperatur_K", "radial_speed_m_per_s", "Fmin", "chi2"]
        )
        for rad in resultater:
            radial_hastighet = dopplerskift_til_fart(
                rad["delta_lamda"],
                rad["bølge0"],
                konst.c,
            )
            skriver.writerow(
                [
                    rad["gass"],
                    f"{rad['bølge0']:.4f}",
                    f"{rad['temperatur']:.4f}",
                    f"{radial_hastighet:.4f}",
                    f"{rad['fmin']:.6f}",
                    f"{rad['chi2']:.6f}",
                ]
            )
    return csv_sti


def main():
    """
    1) Leser spektrum og støy (nm, fluks, sigma).
    2) Definerer kandidatlinjer (fra tabell i oppgaven).
    3) Setter fysikk-konstanter og molekylmasser.
    4) Kjører rask eller sakte
    """
    skriptmappe = Path(__file__).resolve().parent
    plottmappe = skriptmappe / "plots"

    # Leser inn spektrum og støy per punkt
    spektrumdata = np.loadtxt('prosjektfiler/del6/data/spectrum_seed89_600nm_3000nm.txt')
    noisedata = np.loadtxt('prosjektfiler/del6/data/sigma_noise.txt')
    bølgelengder = spektrumdata[:, 0]
    fluks = spektrumdata[:, 1]
    noise = noisedata[:, 1]

    # Definerer tillatt linjedybde: fmin [0.7, 1.0], la til 0.05 buffer
    fmin_omrade = (0.65, 1.0)

    # Lister kandidatlinjer (nm). None = ikke definert andre/tredje linje.
    gass_liste = [
        ("O2", [632.0, 690.0, 760.0]),
        ("H2O", [720.0, 820.0, 940.0]),
        ("CO2", [1400.0, 1600.0, None]),
        ("CH4", [1660.0, 2200.0, None]),
        ("CO", [2340.0, None, None]),
        ("N2O", [2870.0, None, None]),
    ]

    # Setter masse: atommasse * protonmasse 
    protonmasse = konst.m_p
    molekylmasser = {
        "O2": 32.0 * protonmasse,
        "H2O": 18.0 * protonmasse,
        "CO2": 44.0 * protonmasse,
        "CH4": 16.0 * protonmasse,
        "CO": 28.0 * protonmasse,
        "N2O": 44.0 * protonmasse,
    }

    oppsett = {
        "bølgelengder": bølgelengder,
        "fluks": fluks,
        "noise": noise,
        "temperaturomrade": (100, 500),   # K
        "fmin_omrade": fmin_omrade,
        "maks_fart": 10_000.0,               # m/s (radial, brukes til intervall og Delta-Lambda-grid)
        "dopplerbuffer": 1.10,               # 10% margin rundt maks skift
        "lysfart": konst.c,                  # m/s
        "boltzmann": konst.k_B,              # J/K
        "molekylmasser": molekylmasser,
    }

    
    N = 35

    søk(gass_liste, oppsett, plottmappe, skriptmappe, rutenett=N)

if __name__ == "__main__":
    main()
