
import datetime

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




import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

class CoordTransform(object) :

    def __init__(self, detector_lat_deg, detector_long_deg, detector_height_m) :

        self.detector_location = EarthLocation(lat=detector_lat_deg*u.deg, lon=detector_long_deg*u.deg, height=detector_height_m*u.m)


    def get_coszen(self, declination_deg, RA_deg, datetime, utcoffset_hr) : #TODO pass python datetime object
        '''
        Get the cosine of the zenith angle corresponding to the given declination (at the specified date/time)

        Depends on the detector location
        '''

        # Create observation time object
        obs_time = Time(datetime) - (utcoffset_hr*u.hour)

        # Create neutrino direction
        #TODO should this be the opposite direction (e.g. travel direction vs origin direction?)
        nu_dir = SkyCoord(ra=RA_deg*u.deg, dec=declination_deg*u.deg)

        # liv_dir_alt_az_values_day_icecube = liv_dir.transform_to(frame_values_day_icecube)

        # Convert to local coordinate system (zenith, azimuth)
        #TODO what is teh definition of azimuth?
        frame = AltAz(obstime=obs_time, location=self.detector_location)
        nu_dir_alt_az = nu_dir.transform_to(frame)

        # Get zenith angle and convert to cos(zen)
        coszen = np.cos( nu_dir_alt_az.zen.to(u.rad).value )

        #TODO also azimuth

        return coszen




if __name__ == "__main__" :

    #
    # Physics steering
    #

    E_GeV = 1e3

    aX = 2e-23
    aY = 0. #2e-23
    cX = 0.
    cY = 0.

    phi0 = 0.


    #
    # Basic 1D osc prob plot vs RA (IceCube)
    #

    coszen = -0.8
    zenith = np.arccos(coszen)
    declination = calc_declination_from_zenith_south_pole(zenith) #TODO can repalce with CoordTransform
    L_km = calc_path_length_from_coszen(coszen)

    right_ascension_values = np.linspace(0., 2.*np.pi, num=100)

    P_values = P_mu_mu(
        E_GeV=E_GeV,
        L_km=L_km,
        declination=declination,
        right_ascension=right_ascension_values,
        phi0=phi0,
        aX=aX,
        aY=aY,
        cX=cX,
        cY=cY,
    )

    fig, ax = plt.subplots( figsize=(6, 4) ) 

    fig.suptitle(r"IceCube : $E$ = %0.3g GeV, $\delta$ = %0.3g deg ($\rightarrow \cos(\theta)$ = %0.3g)" % (E_GeV, np.rad2deg(declination), coszen) )

    ax.plot(np.rad2deg(right_ascension_values), P_values, color="blue", lw=3)

    ax.set(xlabel='RA [deg]', ylabel=r'$P_{\mu \mu}$', xlim=(0., 360.), ylim=(0., 1.) )
    ax.grid()

    figname = "directional_liv_1D.png"
    fig.savefig(figname)
    print("Saved: %s" % figname)



    #
    # Compare IceCube and DUNE
    #

    #TODO declination to zenith, including time

    transform_icecube = CoordTransform(detector_lat_deg=-90., detector_long_deg=0., detector_height_m=-1e3)
    transform_dune_fd = CoordTransform(detector_lat_deg=44.3517, detector_long_deg=-103.7513, detector_height_m=-1.5e3)

    # Define the neutrino observation
    utcoffset_hr = -6. # Dune is US central, 6 hrs behind UTC. TODO What is Pole time?
    # nu_datetime = "2021-01-01 00:00:00"

    t0 = datetime.datetime(2021, 1, 1, 0, 0, 0, 0)
    dt_hrs = np.linspace(0., 24., num=100)
    nu_datetime = [ t0 + datetime.timedelta(hours=h) for h in dt_hrs ]
    nu_RA_deg = 0.

    nu_declination_deg_values = np.linspace(-90., +90., num=5)

    fig_cz, ax_cz = plt.subplots(nu_declination_deg_values.size, figsize=(6, 8))
    fig_P, ax_P = plt.subplots(nu_declination_deg_values.size, figsize=(6, 8))

    title = r"$E$ = %0.3g GeV, RA = %0.3g deg" % (E_GeV, nu_RA_deg)
    for fig in [fig_cz, fig_P] :
        fig.suptitle(title)

    for i, nu_declination_deg in enumerate(nu_declination_deg_values) :

        # ax[i].set_title(())

        # Get cos(zen)
        nu_coszen_icecube = transform_icecube.get_coszen(declination_deg=nu_declination_deg, RA_deg=nu_RA_deg, datetime=nu_datetime, utcoffset_hr=utcoffset_hr)
        nu_coszen_dune_fd = transform_dune_fd.get_coszen(declination_deg=nu_declination_deg, RA_deg=nu_RA_deg, datetime=nu_datetime, utcoffset_hr=utcoffset_hr)

        # Plot
        ax_cz[i].plot(dt_hrs, nu_coszen_icecube, color="blue", lw=3, label="IceCube")
        ax_cz[i].plot(dt_hrs, nu_coszen_dune_fd, color="red", lw=3, label="DUNE")

        # Get baseline
        L_icecube_km = calc_path_length_from_coszen(cz=nu_coszen_icecube, d=1.) #TODO correct detector depths
        L_dune_fd_km = calc_path_length_from_coszen(cz=nu_coszen_dune_fd, d=1.) #TODO correct detector depths

        # Get osc prob
        P_icecube = P_mu_mu(
            E_GeV=E_GeV,
            L_km=L_icecube_km,
            declination=np.deg2rad(nu_declination_deg),
            right_ascension=np.deg2rad(nu_RA_deg),
            phi0=phi0,
            aX=aX,
            aY=aY,
            cX=cX,
            cY=cY,
        )

        P_dune_fd = P_mu_mu(
            E_GeV=E_GeV,
            L_km=L_dune_fd_km,
            declination=np.deg2rad(nu_declination_deg),
            right_ascension=np.deg2rad(nu_RA_deg),
            phi0=phi0,
            aX=aX,
            aY=aY,
            cX=cX,
            cY=cY,
        )

        # Plot
        ax_P[i].plot(dt_hrs, P_icecube, color="blue", lw=3, label="IceCube")
        ax_P[i].plot(dt_hrs, P_dune_fd, color="red", lw=3, label="DUNE")

        # Labels
        physics_text = r"$\delta$ = %0.3g deg"%nu_declination_deg
        bbox = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax_cz[i].annotate(physics_text, xy=(23.8, -0.91), ha="right", va="bottom", fontsize=12, bbox=bbox)
        ax_P[i].annotate(physics_text, xy=(23.8, 0.05), ha="right", va="bottom", fontsize=12, bbox=bbox)

        # Format
        ax_cz[i].set(ylabel=r'$\cos(\theta)$', ylim=(-1.02, 1.02), xlim=(dt_hrs[0], dt_hrs[-1]) )
        ax_cz[i].grid()

        ax_P[i].set(ylabel=r'$P_{\mu \mu}$', ylim=(-0.02, 1.02), xlim=(dt_hrs[0], dt_hrs[-1]) )
        ax_P[i].grid()

    ax_cz[-1].set(xlabel="t [hr]")
    ax_cz[-1].legend(loc="lower left", fontsize=12)
    fig_cz.tight_layout(rect=(0., 0., 1., 0.96))

    ax_P[-1].set(xlabel="t [hr]")
    ax_P[-1].legend(loc="lower left", fontsize=12)
    fig_P.tight_layout(rect=(0., 0., 1., 0.96))

    figname = "coszen_vs_time.png"
    fig_cz.savefig(figname)
    print("Saved: %s" % figname)

    figname = "Pmm_vs_time.png"
    fig_P.savefig(figname)
    print("Saved: %s" % figname)

