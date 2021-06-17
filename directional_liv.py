
import numpy as np
import matplotlib.pyplot as plt


def P_mu_mu(
    E_GeV,
    L_km,
    declination,
    right_ascension,
    phi0,
    aX,
    aY,
    cX,
    cY,
) :
    '''
    Equations 2-5 from arXiv:1010.4096
    '''

    #TODO CPT odd

    L_per_GeV = L_km * 1e3 * 5076142.1319796955 * 1e9 # Convert to [1/eV], then [1/GeV]

    #TODO E unit?

    # Coordinate system angles
    theta = ( np.pi / 2 ) + declination
    phi =  np.pi + right_ascension

    # Unit vectors
    NX = np.sin(theta) * np.cos(phi)
    NY = np.sin(theta) * np.sin(phi)

    # Amplitudes
    As = ( NY * ( aX - ( 2 * E_GeV * cX ) ) ) - ( NX * ( aY - ( 2 * E_GeV * cY ) ) )
    Ac = ( -NX * ( aX - ( 2 * E_GeV * cX ) ) ) - ( NY * ( aY - ( 2 * E_GeV * cY ) ) )

    # print(As, Ac)
    # print(As)
    # print(Ac)

    # Transition prob
    Pmm = 1. - np.square( np.sin( L_per_GeV * ( (As * np.sin(right_ascension + phi0)) + (Ac * np.cos(right_ascension + phi0)) ) ) )

    assert np.all( Pmm >= 0. )
    assert np.all( Pmm <= 1. )

    return Pmm
    


def calc_declination_from_zenith_south_pole(zenith) :
    return zenith - np.deg2rad(90.)


def calc_path_length_from_coszen(cz, r=6371., h=15., d=1.) :
    '''
    Get the path length (baseline) for an atmospheric neutrino,
    given some cos(zenith angle).
    cz = cos(zenith) in radians, to be converted to path length in km
    r = Radius of Earth, in km
    h = Production height in atmosphere, in km
    d = Depth of detector, in km
    '''
    return -r*cz +  np.sqrt( (r*cz)**2 - r**2 + (r+h+d)**2 )




if __name__ == "__main__" :


    #
    # Define system
    #

    E_GeV = 1e3

    coszen = -0.8
    zenith = np.arccos(coszen)
    declination = calc_declination_from_zenith_south_pole(zenith)
    L_km = calc_path_length_from_coszen(coszen)

    print(coszen, zenith, declination, L_km)

    right_ascension_values = np.linspace(0., 2.*np.pi, num=100)
    phi0 = 0.

    P_values = P_mu_mu(
        E_GeV=E_GeV,
        L_km=L_km,
        declination=declination,
        right_ascension=right_ascension_values,
        phi0=phi0,
        aX=2e-23,
        aY=0.,
        cX=0.,
        cY=0.,
    )

    fig, ax = plt.subplots()

    ax.plot(np.rad2deg(right_ascension_values), P_values, color="orange", lw=3)

    ax.set(xlabel='RA [deg]', ylabel=r'$P_{\mu \mu}$', xlim=(0., 360.), ylim=(0., 1.) )
    ax.grid()

    figname = "directional_liv_1D.png"
    fig.savefig(figname)
    print("Saved: %s" % figname)

