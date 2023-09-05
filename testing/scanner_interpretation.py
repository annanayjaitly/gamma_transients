# from gamma_transients import core
from typing import Any
import dill
from os import cpu_count
from tqdm import tqdm
import concurrent.futures as cf

from astropy.table import Table, Column, Row
from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.spatial import KDTree
from scipy.stats import expon
from scipy.special import factorial

import numpy as np
from matplotlib import figure, axes
import pandas as pd
import numexpr

from tevcat import Source
import healpy as hp
from gammapy.data import DataStore, Observation
from gammapy.maps import WcsNDMap

# from lmfit.models import


from scipy.special import erfinv


def n_sigmas(p):
    return (2**0.5) * erfinv(1.0 - p)


def sphere_dist(ra1, dec1, ra2, dec2):
    """
    Haversine formula for angular distance on a sphere: more stable at poles.
    This version uses arctan instead of arcsin and thus does better with sign
    conventions.  This uses numexpr to speed expression evaluation by a factor
    of 2 to 3.

    :param ra1: first RA (deg)
    :param dec1: first Dec (deg)
    :param ra2: second RA (deg)
    :param dec2: second Dec (deg)

    :returns: angular separation distance (deg)

    https://stackoverflow.com/a/71510946
    """

    ra1 = np.radians(ra1).astype(np.float64)
    ra2 = np.radians(ra2).astype(np.float64)
    dec1 = np.radians(dec1).astype(np.float64)
    dec2 = np.radians(dec2).astype(np.float64)

    numerator = numexpr.evaluate(
        "sin((dec2 - dec1) / 2) ** 2 + "
        "cos(dec1) * cos(dec2) * sin((ra2 - ra1) / 2) ** 2"
    )

    dists = numexpr.evaluate("2 * arctan2(numerator ** 0.5, (1 - numerator) ** 0.5)")
    return np.degrees(dists)


@np.vectorize
def convert_angle(angle: float) -> float:
    """
    Convert angle from 0 to 360 to -180 to 180

    Parameters
    ----------
    angle : float
        Angle in degrees

    Returns
    -------
    float
        Converted angle
    """
    if angle > 180:
        return angle - 360
    return angle


def objectifyColumn(tab: Table, colname: str) -> None:
    """
    Turn the dtype of a column into `object`

    Parameters
    ----------
    tab : astropy.table.Table
        Table to modify

    colname : str
        Name of the column to modify

    """
    modcol = tab[colname].data.tolist()
    modcol[0].append(None)
    tab.replace_column(colname, modcol)
    tab[colname][0].pop()


def radec_to_SimbadCircle(ra: float, dec: float, radius_deg=0.2) -> str:
    """
    Return a simbad criteria string for a circle around a SkyCoord

    Parameters
    ----------
    ra_deg: float
        Right ascension in degrees

    dec_deg: float
        Declination in degrees

    radius_deg : float
        Radius of the circle in degrees

    Returns
    -------
    str
        Simbad criteria string
    """
    prefix = ""
    if dec >= 0:
        prefix += "+"
    return f"region(circle,{ra} {prefix}{dec},{radius_deg}d)"


def cat2hpx(
    lon: np.ndarray, lat: np.ndarray, nside: int, radec: bool = True
) -> np.ndarray:
    """
    Convert a catalogue to a HEALPix map of number counts per resolution
    element. (https://stackoverflow.com/a/50495134)

    Parameters
    ----------
    lon, lat : (ndarray, ndarray)
        Coordinates of the sources in degree. If radec=True, assume input is in the fk5
        coordinate system. Otherwise assume input is glon, glat

    nside : int
        HEALPix nside of the target map

    radec : bool
        Switch between R.A./Dec and glon/glat as input coordinate system.

    Return
    ------
    hpx_map : ndarray
        HEALPix map of the catalogue number counts in Galactic coordinates

    """

    npix = hp.nside2npix(nside)

    if radec:
        eq = SkyCoord(lon, lat, frame="fk5", unit="deg")
        l, b = eq.galactic.l.value, eq.galactic.b.value
    else:
        l, b = lon, lat

    # conver to theta, phi
    theta = np.radians(90.0 - b)
    phi = np.radians(l)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi)

    idx, counts = np.unique(indices, return_counts=True)

    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=int)
    hpx_map[idx] = counts

    return hpx_map


class Multiplets:
    """
    Class for handling multiplets

    Attributes
    ----------
    paths : list[str]
        List of paths to the multiplet folders

    table : astropy.table.Table
        Table of the multiplets

        colnames:
        'Nmax'
        'OBS_ID'
        'ID'
        'RA'
        'DEC'
        'TIME'
        'ENERGY'
        'dt'
        'da'

        (added by Multiplets.addCoordinateInfo()
        'MEDIAN_RA'
        'MEDIAN_DEC'
        'SkyCoord'
        'MEDIAN_GLAT'
        'MEDIAN_GLON'

        (added by Multiplets.searchTeVCat()
        'TEVCAT_SOURCE_NAME'
        'TEVCAT_SOURCE_TYPE'
        'TEVCAT_DISTANCES_DEG'

        (added MANUALLY, use in_which_container())
        'DS_INDEX'

        (added by Multiplets.addRMS()
        'ANGULAR_MEASURE_DEG'


    reduced : astropy.table.Table
        Table of the reduced multiplets


    """

    def __init__(self, path: str) -> None:
        self.paths = [path]
        self.loadMpletsBare(path)

    def loadMpletsBare(self, path) -> None:
        """
        load the multiplets from a path returned by the transient tool.

        Parameters
        ----------
        path : str
            Path to the multiplet folder

        """
        if path not in self.paths:
            self.paths.append(path)
            # try except
        with open(path + "/multiplets.pkl", "rb") as f:
            self.table = dill.load(f)
        self.table.sort("Nmax")

    def addCoordinateInfo(self) -> None:
        """
        Add coordinate information (MEDIAN_RA, MEDIAN_DEC, SkyCoord, MEDIAN_GLAT, MEDIAN_GLON) to Multiplets.table
        """
        self.table["MEDIAN_RA"] = [np.median(lst) for lst in self.table["RA"]]
        self.table["MEDIAN_DEC"] = [np.median(lst) for lst in self.table["DEC"]]

        self.table["SkyCoord"] = SkyCoord(
            self.table["MEDIAN_RA"],
            self.table["MEDIAN_DEC"],
            frame="fk5",
            unit="deg",
        )
        self.table["MEDIAN_GLAT"] = self.table["SkyCoord"].galactic.b
        self.table["MEDIAN_GLON"] = self.table["SkyCoord"].galactic.l

    def getAitoffFigure(self, bounds_deg: list = []) -> (figure.Figure, axes.Axes):
        """
        Get a scatterplot of the multiplet coordinats in aitoff projection

        Parameters
        ----------

        bounds_deg : list
            List of GLAT bounds to be plotted as dashed lines, represent the galactic plane.

        Returns
        -------

        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        """
        fig = figure.Figure()
        ax = fig.add_subplot(projection="aitoff")
        sc = ax.scatter(
            np.radians(convert_angle(self.table["MEDIAN_GLON"])),
            np.radians(self.table["MEDIAN_GLAT"]),
            s=0.5,
            c=self.table["Nmax"],
            marker="*",
            zorder=1,
            cmap="gist_ncar",
        )
        cbar = fig.colorbar(sc)
        cbar.set_label("Multiplet size")
        for item in bounds_deg:
            ax.plot(
                np.linspace(-np.pi, np.pi, 100),
                np.full(100, np.radians(item)),
                color="red",
                ls="--",
                zorder=3,
                lw=0.1,
            )
        ax.grid(True)
        ax.set_xlabel(r"MEDIAN_GLON [deg]")
        ax.set_ylabel(r"MEDIAN_GLAT [deg]")
        ax.tick_params(grid_alpha=0.3, colors="gray", zorder=3)

        return fig, ax

    def searchTeVCat(self, sources: list[Source]):
        """
        Perform a nearest neighbour search (k-dimensional tree) in the TeVCat database from the median multiplet coordinate.

        Parameters
        ----------

        sources : list[tevcat.Source]
            List of tevcat sources to search from. All available sources can be obtained by tevcat.TeVCat().sources

        """
        coordinates = np.asarray(
            [[source.getFK5().ra.deg, source.getFK5().dec.deg] for source in sources]
        )

        kdtree = KDTree(coordinates)

        distances, indices = kdtree.query(
            np.array(
                [
                    self.table["MEDIAN_RA"].data,
                    self.table["MEDIAN_DEC"].data,
                ]
            ).transpose()
        )
        identified_sources = np.asarray(sources)[indices]

        self.addTevCatSourceInfo(identified_sources, distances)

    def addTevCatSourceInfo(
        self, identified_sources: list[Source], distances: list[float]
    ) -> None:
        """
        add Source information to Multiplets.table based on the nearest TeVCat source

        Parameters
        ----------

        identified_sources : list[tevcat.Source]
            List of tevcat sources that were identified as nearest neighbours

        distances : list[float]
            List of distances to the nearest TeVCat source

        """
        self.table["TEVCAT_SOURCE_NAME"] = [
            source.canonical_name for source in identified_sources
        ]
        self.table["TEVCAT_SOURCE_TYPE"] = [
            source.source_type_name for source in identified_sources
        ]
        self.table["TEVCAT_DISTANCES_DEG"] = distances
        self.table["TEVCAT_FLUX_CRAB"] = [source.flux for source in identified_sources]

    def __len__(self) -> int:
        return len(self.table)

    def appendMultiplets(self, *others):
        """
        Append other Multiplets to self. Used to combine multiplets from different datasets.

        """
        df = pd.concat(
            [self.table.to_pandas(), *[other.table.to_pandas() for other in others]],
            ignore_index=True,
        )
        self.table = Table.from_pandas(df)
        self.table.sort("OBS_ID")
        self.addCoordinateInfo()

    def createReduced(
        self,
        min_source_dist_deg: float = 0.1,
        galactic_plane_halfwidth_deg: float = 5.0,
    ) -> None:
        """
        Create a reduced datast excluding TeVCat sources and sources close to the galactic plane.

        Parameters
        ----------

        min_source_dist_deg : float
            Minimum distance to a TeVCat source in degrees

        galactic_plane_halfwidth_deg : float
            Halfwidth of the galactic plane exclusion zone in degrees

        """
        noTeVCat_mask = self.table["TEVCAT_DISTANCES_DEG"] >= min_source_dist_deg
        xgal_mask = np.abs(self.table["MEDIAN_GLAT"]) > galactic_plane_halfwidth_deg

        self.reduced = self.table[noTeVCat_mask * xgal_mask]
        print(f"Reducement of {len(self)} to {len(self.reduced)} rows.")

    def objectifyColumns(self) -> None:
        """
        Turn relevant Multiplets.table columns into dtype `object`. To be used on Multiplets that contain but one multiplicity, causing a merge of columns via Multiplets.appendMultiplets with different multiplicities to fail.

        """
        for name in ["ID", "RA", "DEC", "TIME", "ENERGY"]:
            objectifyColumn(self.table, name)

    def setReduced(self, reduced: Table):
        self.reduced = reduced

    def addRMS(self):
        """
        Add a measure for the angular spread of the multiplet to Multiplets.table. Columname: ANGULAR_MEASURE_DEG
        This is done by calculating the mean distance of the multiplet members to the median multiplet coordinate.

        """
        measure = []
        workers = int(cpu_count() / 2)

        with cf.ProcessPoolExecutor(workers) as ex:
            future_rms = [
                ex.submit(
                    rms_worker,
                    row["RA"],
                    row["DEC"],
                    row["MEDIAN_RA"],
                    row["MEDIAN_DEC"],  ##
                )
                for row in tqdm(self.table)
            ]
        print("Finalising rms futures")
        for future in tqdm(future_rms):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error: {e}")
                result = np.nan
            measure.append(result)

        self.table["ANGULAR_MEASURE_DEG"] = measure


def rms_worker(ra: np.ndarray, dec: np.ndarray, cra: float, cdec: float) -> float:
    return np.quantile(sphere_dist(ra, dec, cra, cdec), 0.68)


class Reduced:
    """
    A reduced Multiplets class (not a subclass) that contains the reduced multiplets and methods to add information to it.

    Attributes
    ----------
    reduced : astropy.table.Table
        Table of the reduced multiplets

        colnames:
        see Multiplets.table

        (added by Reduced.addLambdaRatioSignificace())
        'LAMBDA_RATIO_SIGNIFICANCE'
        'BKG_DT_LAMBDA'
        'MPLET_DT_LAMBDA'

        (added by Reduced.addDistanceToPNT())
        'PNT_DISTANCE'

        (added by Reduced.addPNTAltitude())
        'ALT_PNT'

        (added by Reduced.addExposureCorrectedP())
        'BELL_FRACTION'
        'EXP_CORRECTED_P'
        'BERNOULLI_P'
        'BERNOULLI_SIGMA'



    datastores : list[DataStore]
        List of datastores used to create the reduced multiplets

    observation_per_ds : list[Observation]
        List of observations per datastore

    navtable : astropy.table.Table
        Table containing "navigational information".

        Columns:
        MPLET_INDEX: index of the multiplet in reduced
        DS_INDEX: index of the datastore in datastores
        OBS_ID: observation id
        OBS_INDEX_WITHIN_DS: index of the observation in observation_per_ds whn observations are retuned from the datastore, which is not   the same order as the OBS_IDs passed


    """

    def __init__(self, path: str) -> None:
        with open(path, "rb") as f:
            self.reduced = dill.load(f)

    def loadObservations(self, per_ds: bool = False) -> None:
        """
        Load the datastores to Reduced.datastores and the observations to Reduced.observation_per_ds, if per_ds is True.

        Parameters
        ----------

        per_ds : bool (default: False)
            If True, load the observations per datastore. If False, load all observations.
        """
        self.datastores = getDataStores()
        if per_ds:
            self.observation_per_ds = [
                ds.get_observations(
                    np.unique(
                        self.reduced["OBS_ID"].data[self.reduced["DS_INDEX"] == i]
                    ),
                    required_irf="all-optional",
                )
                for i, ds in enumerate(self.datastores)
            ]

    def loadNavtable(self, navpath: str = None) -> None:
        """
        Load the navigation table from a path. If no path is given, make a new one.

        Parameters
        ----------

        navpath : str (default: None)
            Path to the navigation table. If None, make a new one.
        """
        try:
            with open(navpath, "rb") as f:
                self.navtable = dill.load(f)
        except:
            print("Navtab not found, making from scratch.")
            self.navtable = self.makeNavigationTable()

    def makeNavigationTable(self) -> Table:
        """
        Make the navigation table from scratch.

        Returns
        -------
        astropy.table.Table
            Table containing metadata navigation information.

            Columns:
            MPLET_INDEX: index of the multiplet in reduced
            DS_INDEX: index of the datastore in datastores
            OBS_ID: observation id
            OBS_INDEX_WITHIN_DS: index of the observation in observation_per_ds whn observations are retuned from the datastore, which is not   the same order as the OBS_IDs passed
        """
        obs_id_to_index_per_ds = dict()
        for obs in self.observation_per_ds:
            obs_id_to_index_per_ds.update(
                {int(obs.ids[index]): index for index in range(len(obs))}
            )

        print("Building navigation table")
        navigation_table = Table(
            [
                range(len(self.reduced)),
                self.reduced["OBS_ID"],
                self.reduced["DS_INDEX"],
            ],
            names=("MPLET_INDEX", "OBS_ID", "DS_INDEX"),
            dtype=[int, int, int],
        )
        print("completing navigation table")
        temp = Column(
            name="OBS_INDEX_WITHIN_DS",
            data=list(map(obs_id_to_index_per_ds.get, navigation_table["OBS_ID"].data)),
            dtype=int,
        )
        navigation_table.add_column(temp)
        return navigation_table

    def addLambdaRatioSignificace(self) -> None:
        """
        Add a data driven significance to each multiplet. This is the probability of finding N >= Nm photons within burst time given a background rate. The burst time is not assumed to avoid biasing the significance, but rather rates are calculated from exponential fits on the differences in arrival times. (Since counting photons is assumed to be Poisson, the Delta(arrival time) is to be exponentially distributed.)

        Adds the following columns to Reduced.reduced:
            LAMBDA_RATIO_SIGNIFICANCE: the significance of the multiplet
        (if not already present)
            BKG_DT_LAMBDA: the background rate in Hz
            MPLET_DT_LAMBDA: the signal rate in GHz
        """
        if ("BKG_DT_LAMBDA" not in self.reduced.colnames) or (
            "MPLET_DT_LAMBDA" not in self.reduced.colnames
        ):
            print("Haven't found exponential fit parameters, will now start fitting.")
            self.addExponentialDtFits(
                getDataStores(), navpath="testing/pkl_jugs/navigationtable.pkl"
            )

        NormalizedLambda = (
            self.reduced["Nmax"]
            * self.reduced["BKG_DT_LAMBDA"]
            / 1e9
            / self.reduced["MPLET_DT_LAMBDA"]
        )
        self.reduced["LAMBDA_RATIO_SIGNIFICANCE"] = np.ones_like(
            NormalizedLambda
        ) - np.asarray(
            [
                np.sum(
                    np.power(lamb, np.arange(self.reduced[i]["Nmax"]))
                    * np.exp(-lamb)
                    / factorial(np.arange(self.reduced[i]["Nmax"])),
                )
                for i, lamb in enumerate(NormalizedLambda.data)
            ]
        )

    def addExponentialDtFits(self, radius_deg: float = 0.1):
        """
        Perform exponential fits on the differences in arrival times for the candidate signal and the local background. Local is interpreted as the support for 68% prob mass of a PSF (taken to be a circular region with (default) radius 0.1 deg) centered around the multiplet median coordinate.

        Arguments
        ---------
        radius_deg : float
            Radius of the support for the background exponential fit in degrees

        Adds the following columns to Reduced.reduced:
            BKG_DT_LAMBDA: the background rate in Hz
            MPLET_DT_LAMBDA: the signal rate in GHz
            BKG_PHOTONS: the number of photons in the background region

        """
        workercount = int(cpu_count() / 2)

        """
        ==============
        BKG EXPON FIT
        ==============
        """
        print("Getting photon rates.")
        expon_fit_parameters = []
        bkg_photon_count = []

        with cf.ProcessPoolExecutor(workercount) as ex:
            future_fitparams = [
                ex.submit(
                    run_expon_fit_worker,
                    self.observation_per_ds[row["DS_INDEX"]][
                        int(row["OBS_INDEX_WITHIN_DS"])
                    ],
                    self.reduced[row["MPLET_INDEX"]]["ID"],
                    self.reduced[row["MPLET_INDEX"]]["MEDIAN_RA"],
                    self.reduced[row["MPLET_INDEX"]]["MEDIAN_DEC"],
                    radius_deg,
                )
                for row in self.navtable
            ]
        print("Finalising run bkg fit futures")
        for future in tqdm(future_fitparams):
            try:
                result, nphot = future.result()
            except Exception as e:
                print(f"Error: {e}")
                result = np.nan
                nphot = np.nan
            expon_fit_parameters.append(result)
            bkg_photon_count.append(nphot)

        self.reduced["BKG_DT_LAMBDA"] = expon_fit_parameters
        self.reduced["BKG_PHOTONS"] = bkg_photon_count

        """
        ==============
        SIGNAL EXPON FIT
        ==============
        """
        print("Starting mplet expon fit")
        mplet_expon_fit_parameters = []
        with cf.ProcessPoolExecutor(workercount) as ex:
            future_fitparams = [
                ex.submit(
                    expon.fit,
                    np.diff(np.sort(np.asarray(row["TIME"]).astype(int))),
                    floc=0.0,
                )
                for row in self.reduced
            ]
        print("Finalising mplet dt fit futures")
        for future in tqdm(future_fitparams):
            try:
                result = 1.0 / future.result()[1]
            except Exception as e:
                print(f"Error: {e}")
                result = np.nan
            mplet_expon_fit_parameters.append(result)

        self.reduced["MPLET_DT_LAMBDA"] = mplet_expon_fit_parameters

    def addDistanceToPNT(self):
        """
        Add the radial distances from the median multiplet coordinate to the pointing coordinates of the observations in which the multiplet is present.

        Adds the following columns to Reduced.reduced:
            PNT_DISTANCE: the distance in degrees
        """

        distances = []

        workercount = int(cpu_count() / 2)

        with cf.ProcessPoolExecutor(workercount) as ex:
            future_distances = [
                ex.submit(
                    sphere_dist,
                    self.reduced[row["MPLET_INDEX"]]["MEDIAN_RA"],
                    self.reduced[row["MPLET_INDEX"]]["MEDIAN_DEC"],
                    self.observation_per_ds[row["DS_INDEX"]][
                        int(row["OBS_INDEX_WITHIN_DS"])
                    ].obs_info["RA_PNT"],
                    self.observation_per_ds[row["DS_INDEX"]][
                        int(row["OBS_INDEX_WITHIN_DS"])
                    ].obs_info["DEC_PNT"],
                )
                for row in tqdm(self.navtable)
            ]
        print("Finalising MPLET FOV DIST futures")
        for future in tqdm(future_distances):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error: {e}")
                result = np.nan
            distances.append(result)
        self.reduced["PNT_DISTANCE"] = distances

    def addPNTAltitude(self) -> None:
        """
        Add the pointing altitude to Reduced.reduced

        Adds the following columns to Reduced.reduced:
            ALT_PNT: the pointing altitude in degrees
            PNT_SOURCE: the source H.E.S.S. was pointing to

        """
        ## SUDDEN THOUGHT: compare with regular zenith distribution, maybe with the cumulative distributions // Smirnov test
        alt_pnt = []
        pnt_source = []
        for row in tqdm(self.navtable):
            alt = self.observation_per_ds[row["DS_INDEX"]][
                int(row["OBS_INDEX_WITHIN_DS"])
            ].obs_info["ALT_PNT"]
            alt_pnt.append(alt)

            sc = self.observation_per_ds[row["DS_INDEX"]][
                int(row["OBS_INDEX_WITHIN_DS"])
            ].obs_info["OBJECT"]
            pnt_source.append(sc)

        self.reduced["ALT_PNT"] = alt_pnt
        self.reduced["PNT_SOURCE"] = sc

    def addExposureCorrectedP(self, exposure: WcsNDMap) -> None:
        """
        Add two exposure corrected significances, one multiplies the significance by the ratio of the exposure that is contained in the multiplet region to the total H.E.S.S. exposure , the other considers observing a multiplet as a Bernoulli trial and corrects the significance for the number of trials, being the number of runs available in the datastores (which should be the sampled dataset). Both methods are approximative: each run is assumed to have the same exposure (both in time and area). Values of zero result from the multiplet not being contained in the exposure geometry (very edge of FoV).

        One should verify that these yield the same order of magnitude.

        Parameters
        ----------

        exposure : gammapy.maps.WcsNDMap
            Exposure map to use for the correction

        Adds the following columns to Reduced.reduced:
            BELL_FRACTION: the exposure fraction within the mplet region
            EXP_CORRECTED_P: the blind exposure corrected significance, which can rise above 1.
            BERNOULLI_P: the exposure corrected significance following the Bernoulli assumption
            BERNOULLI_SIGMA: p value converted into standard deviations.
        """
        bell_fraction = []
        print("Getting total runcount")
        runcount = get_total_runcount(self.datastores)
        print("Starting multiprocessed bell_ratio.")
        # get_bell_ratio(
        #     exposure,
        #     SkyCoord(*exposure.geom.center_coord).directional_offset_by(
        #         0.0 * u.deg, self.reduced[0]["PNT_DISTANCE"] * u.deg
        #     ),
        #     self.reduced[0]["da"],
        # )
        with cf.ProcessPoolExecutor(int(cpu_count() / 2)) as ex:
            future_expfactors = [
                ex.submit(
                    get_bell_ratio,
                    exposure,
                    SkyCoord(*exposure.geom.center_coord).directional_offset_by(
                        0.0 * u.deg, row["PNT_DISTANCE"] * u.deg
                    ),
                    row["ANGULAR_MEASURE_DEG"],
                )
                for row in tqdm(self.reduced)
            ]
        print("Analysing futures")
        for future in future_expfactors:
            try:
                result = future.result()
            except Exception as e:
                print(f"Error: {e}")
                result = np.nan
            bell_fraction.append(result)

        print("Saving to Reduced.reduced")
        bell_fraction = np.asarray(bell_fraction, dtype=np.double)

        self.reduced["BELL_FRACTION"] = bell_fraction
        self.reduced["EXP_CORRECTED_P"] = (
            runcount * self.reduced["LAMBDA_RATIO_SIGNIFICANCE"] / bell_fraction
        )
        self.reduced["BERNOULLI_P"] = 1.0 - np.power(
            1.0 - (self.reduced["LAMBDA_RATIO_SIGNIFICANCE"] / bell_fraction), runcount
        )

        self.reduced["BERNOULLI_SIGMA"] = n_sigmas(self.reduced["BERNOULLI_P"])

    def getCandidate(self, obs_id: int):
        return self.reduced[self.reduced["OBS_ID"] == obs_id]


class Candidate:
    def __init__(self, row: Row) -> None:
        if type(row) == Table:
            print(f"row is a Table of length {len(row)}, not a row. Taking first!")
            self.mplet = row[0]
        else:
            self.mplet = row
        self.obs_id = self.mplet["OBS_ID"]
        ds = getDataStores()
        try:
            self.obs = ds[0].obs(self.obs_id)
            print("hess1")
        except:
            self.obs = ds[1].obs(self.obs_id)
            print("hess1u")

    def ToAScatter(self, max_dist: float = 0.1) -> (figure.Figure, axes.Axes):
        phottable = self.obs.events.table

        run_dist = sphere_dist(
            phottable["RA"].data,
            phottable["DEC"].data,
            self.mplet["MEDIAN_RA"],
            self.mplet["MEDIAN_DEC"],
        )

        mask = run_dist < max_dist

        fig = figure.Figure()
        ax = fig.add_subplot()
        ax.scatter(phottable[mask]["TIME"], run_dist[mask], s=4, marker="x")
        ax.set_xlim(self.obs.obs_info["TSTART"], self.obs.obs_info["TSTOP"])
        return fig, ax


# def pxl_distance(i1: int, j1: int, i2: int, j2: int) -> float:
#     """
#     Compute Euclidean pixel distance between two pixels.

#     Parameters
#     ----------

#     i1, j1 : int
#         Pixel coordinates of the first pixel

#     i2, j2 : int
#         Pixel coordinates of the second pixel

#     Returns
#     -------
#     float
#         Euclidean pixel distance
#     """
#     return np.sqrt(np.square(i1 - j1) + np.square(i2 - j2))


def get_contained_indices(Nh: int, Nv: int, coord=tuple[int], center: tuple[int] = ()):
    """
    For an image of dimensions (Nh,Nv), get the image indices contained within a circle of radius pxl_distance(center,coord) around center.

    Parameters
    ----------
    Nh, Nv : int
        Image dimensions

    coord : tuple[int]
        Pixel coordinates of the center of the circle

    center : tuple[int] (default: None)
        Pixel coordinates of the center of the image. If None, use the center of the image (Nh/2, Nv/2).

    Returns
    -------

    contained_indices : tuple[np.ndarray]
        Tuple of two arrays containing respectively the x and y indices of the pixels contained within the circle.

    """
    if len(coord) != 2:
        raise ValueError(
            f"Coord should be a tuple of length 2 but has length {len(coord)}"
        )
    if len(center) != 2:
        print("Making new center")
        center = (int(Nh / 2), int(Nv / 2))
    x, y = np.meshgrid(np.arange(Nh), np.arange(Nv))
    distances = np.sqrt(np.square(x - center[0]) + np.square(y - center[1]))
    radius = np.sqrt(np.sum(np.square(center - coord)))

    contained_indices = np.where(distances <= radius)
    return contained_indices


def get_bell_ratio(exposure: WcsNDMap, signal: SkyCoord, signal_width_deg: float):
    """
    Get the probability mass of the normalized exposure contained in the tails of the distribution, defined as the pixels further than `signal` is from the center of the exposure map WCS geometry.

    Parameters
    ----------
    exposure : gammapy.maps.WcsNDMap
        Exposure map to use for the correction

    signal : astropy.coordinates.SkyCoord
        Coordinate of the signal

    Returns
    -------

    bell_ratio : float
    """
    spatial_exposure: WcsNDMap = exposure.sum_over_axes(keepdims=False)
    if not spatial_exposure.geom.contains(signal)[0]:
        print("get_bell_ratio: signal not contained in exposure geometry! Returning 0.")
        return 0.0

    signal_edgecoor: SkyCoord = signal.directional_offset_by(
        0.0 * u.deg, signal_width_deg * u.deg
    )

    contained_idx = get_contained_indices(
        *spatial_exposure.data.shape,
        np.array(spatial_exposure.geom.coord_to_idx(signal_edgecoor)).flatten(),
        np.array(spatial_exposure.geom.coord_to_idx(signal)).flatten(),
    )
    normalization = np.sum(spatial_exposure.data)
    contained_exposure = np.sum(spatial_exposure.get_by_idx(contained_idx))
    return contained_exposure / normalization


def get_total_runcount(ds: list[DataStore]):
    """
    Get the total amount of runs contained in all DataStores in ds.

    Parameters
    ----------
    ds : list[DataStore]
        List of DataStores

    Returns
    -------
    runcount : int
        Total amount of runs in ds
    """
    runcount = 0
    for store in ds:
        runcount += len(store.obs_ids)
    return runcount


def run_expon_fit_worker(
    obs: Observation, mplet_id: tuple[int], medra: float, meddec: float, radius_deg=0.1
):
    """ "
    Worker function for multiprocessing that fits the exponential dt distribution local to the multiplet.

    Parameters
    ----------
    obs: Observation
        The observation where the multiplet is present

    mplet_id: tuple[int]
        The photon id's of the multiplet members, masked in the fit.

    etc

    Returns
    -------

    run_parameters: tuple[float]
        loc,scale of the exponential distribution. Loc has been fixed to zero.

    n_phot: float
        Number of photons present in the region around the multiplet.

    """
    run_distances = sphere_dist(
        obs.events.table["RA"].data,
        obs.events.table["DEC"].data,
        medra,
        meddec,
    )

    on_region_mask = run_distances < radius_deg
    signal_mask = ~np.isin(obs.events.table["EVENT_ID"], mplet_id)

    bkg_photon_count = len(obs.events.table[on_region_mask * signal_mask])

    if bkg_photon_count == 0:
        return 0.0, bkg_photon_count  # no photons: rate==0
    elif bkg_photon_count == 1:
        return bkg_photon_count / obs.obs_info["LIVETIME"], bkg_photon_count

    # dt fit exluced the multiplet
    run_parameters = expon.fit(
        np.diff(
            np.sort(
                obs.events.table[
                    (run_distances < radius_deg)
                    * ~np.isin(obs.events.table["EVENT_ID"], mplet_id)
                ]["TIME"]
            )
        ),
        floc=0,
    )

    return 1.0 / run_parameters[1], bkg_photon_count


def in_which_container(item, containers: list[set]):
    """
    Query in which contained `item` is located. Using sets for the containers is recommended as this makes the search algorithm O(1).

    Parameters
    ----------

    item: object

    containers: list[set]
        The candidate containers to search in.

    Returns
    -------

    index: int
        The index of the container in which `item` is located.
    """
    sets = [set(c) for c in containers]
    for i, myset in enumerate(sets):
        if item in myset:
            return i
    raise LookupError("Item is not in any container. Please ensure it is.")


def getDataStores():
    hess1_datastore = DataStore.from_dir("$HESS1")
    hess1u_datastore = DataStore.from_dir("$HESS1U")

    return [hess1_datastore, hess1u_datastore]


def main():
    print("Mplets")
    mplets = bare_load_mplets()
    print("Mplet manips")
    rdpath = mplet_manips(mplets)
    print("Reduced manips")
    reduced_manips(rdpath)


def bare_load_mplets(
    prefix: str = "/lustre/fs22/group/hess/user/wybouwt/full_scanner_survey/nmax4_da_increased",
) -> Multiplets:
    bands = [
        "u_5_15",
        "u_15_25",
        "u_25_35",
        "u_35_45",
        "u_45_55",
        "u_55_65",
        "u_65_75",
        "u_75_90",
        "l_15_5",
        "l_25_15",
        "l_35_25",
        "l_45_35",
        "l_55_45",
        "l_65_55",
        "l_75_65",
        "l_90_75",
        "center",
    ]
    versions = ["hess1", "hess1u"]
    paths = [f"{prefix}/{version}/{band}" for band in bands for version in versions]
    mplet_list = [Multiplets(path) for path in paths]
    Nmax_list = [np.unique(mplets.table["Nmax"].data) for mplets in mplet_list]

    unique_nmax_bands = []
    for i, Nmax in enumerate(Nmax_list):
        if len(Nmax) == 1:
            unique_nmax_bands.append(i)

    for k in unique_nmax_bands:
        mplet_list[k].objectifyColumns()

    mplets = mplet_list[0]
    mplets.appendMultiplets(*mplet_list[1:])

    with open("testing/pkl_jugs/n4/mplets_bare.pkl", "wb") as f:
        dill.dump(mplets, f)
    return mplets


def mplet_manips(mplets: Multiplets) -> str:
    from tevcat import TeVCat

    tevcat = TeVCat()
    mplets.searchTeVCat(tevcat.sources)

    datastores = getDataStores()
    ds_mplet_indices = [
        in_which_container(id, containers=[ds.obs_ids for ds in datastores])
        for id in tqdm(mplets.table["OBS_ID"].data)
    ]

    mplets.table["DS_INDEX"] = ds_mplet_indices
    mplets.addRMS()

    mplets.createReduced()

    redpath = "testing/pkl_jugs/n4/reduced0.pkl"
    with open(redpath, "wb") as f:
        dill.dump(mplets.reduced, f)
    print(
        f"unique obs ids in reduced dataset: {len(np.unique(mplets.reduced['OBS_ID'].data))}"
    )
    return redpath


def reduced_manips(redpath: str):
    rd = Reduced(redpath)
    rd.loadObservations(per_ds=True)
    rd.loadNavtable()

    print("dumping navtable")
    with open("testing/pkl_jugs/n4/navtab.pkl", "wb") as f:
        dill.dump(rd.navtable, f)

    print("Adding PNT metadata.")
    rd.addPNTAltitude()
    rd.addDistanceToPNT()
    print("Adding exponential fits.")
    rd.addExponentialDtFits()
    print("Adding Lambda ratio significance.")
    rd.addLambdaRatioSignificace()
    print("Correcting for exposure.")
    with open(
        "testing/mc_scanner/real_dataset_dumps/hbl/stacked_datasets.pkl", "rb"
    ) as f:
        exposure: WcsNDMap = dill.load(f).exposure
    rd.addExposureCorrectedP(exposure)

    with open("testing/pkl_jugs/n4/reduced_complete.pkl", "wb") as f:
        dill.dump(rd.reduced, f)


# def main_correctP(reduced_path: str):
#     rd = Reduced(reduced_path)
#     print("Loading observations")
#     rd.loadObservations()
#     print("Loading exposure")
#     with open(
#         "testing/mc_scanner/real_dataset_dumps/hbl/stacked_datasets.pkl", "rb"
#     ) as f:
#         exposure: WcsNDMap = dill.load(f).exposure
#     print("Starting exposure correction")
#     rd.addExposureCorrectedP(exposure)

#     print("dumping rd.reduced")
#     with open("testing/pkl_jugs/reduced_correctedP_020923.pkl", "wb") as f:
#         dill.dump(rd.reduced, f)


# def setup_mplets():
#     bands = [
#         "u_5_15",
#         "u_15_25",
#         "u_25_35",
#         "u_35_45",
#         "u_45_55",
#         "u_55_65",
#         "u_65_75",
#         "u_75_90",
#         "l_15_5",
#         "l_25_15",
#         "l_35_25",
#         "l_45_35",
#         "l_55_45",
#         "l_65_55",
#         "l_75_65",
#         "l_90_75",
#         "center",
#     ]
#     versions = ["hess1", "hess1u"]
#     paths = [
#         f"/lustre/fs22/group/hess/user/wybouwt/full_scanner_survey/nmax4_da_increased/{version}/{band}"
#         for band in bands
#         for version in versions
#     ]
#     # mplets = scani.Multiplets(paths[0])
#     # mplets.appendMultiplets(*[scani.Multiplets(path) for path in paths[1:]])
#     mplet_list = [Multiplets(path) for path in paths]
#     Nmax_list = [np.unique(mplets.table["Nmax"].data) for mplets in mplet_list]

#     for i in range(len(Nmax_list)):
#         print(i, Nmax_list[i])

#     # unicorns = [9, 18]
#     # unicorns = [7, 16, 17, 24, 25, 32]
#     for mplet in mplet_list:
#         mplet.objectifyColumns()

#     mplets = mplet_list[0]
#     mplets.appendMultiplets(*mplet_list[1:])

#     from tevcat import TeVCat

#     tevcat = TeVCat()
#     mplets.searchTeVCat(tevcat.sources)

#     datastores = getDataStores()
#     ds_mplet_indices = [
#         in_which_container(id, containers=[ds.obs_ids for ds in datastores])
#         for id in tqdm(mplets.table["OBS_ID"].data)
#     ]

#     mplets.table["DS_INDEX"] = ds_mplet_indices
#     mplets.addRMS()

#     print("Dumping the mplets object.")
#     with open("testing/pkl_jugs/n4/mplets.pkl", "wb") as f:
#         dill.dump(mplets, f)

#     print("Creating reduced dataset")
#     mplets.createReduced()
#     print(
#         f"unique obs ids in reduced dataset: {len(np.unique(mplets.reduced['OBS_ID'].data))}"
#     )
#     print("Dumping bare reduced")
#     with open("testing/pkl_jugs/n4/reduced_bare.pkl", "wb") as f:
#         dill.dump(mplets.reduced, f)


def main_aitov(mplets):
    nosource_mask = mplets.table["TEVCAT_DISTANCES_DEG"] >= 0.2
    Nmin4_mask = mplets.table["Nmax"] >= 4
    Nmin5_mask = mplets.table["Nmax"] >= 5
    dt1sec_mask = mplets.table["dt"] <= 1e9

    fig = figure.Figure((5, 3))
    ax = fig.add_subplot(projection="aitoff")
    scattersize = 15
    current_mask = nosource_mask * Nmin4_mask * dt1sec_mask
    ax.scatter(
        -np.radians(convert_angle(mplets.table[current_mask]["MEDIAN_GLON"])),
        np.radians(mplets.table[current_mask]["MEDIAN_GLAT"]),
        s=scattersize,
        marker="v",
        zorder=1,
        color="m",
        label=r"$N=4, dt < 1\mathrm{s}$",
    )
    current_mask = nosource_mask * Nmin5_mask
    ax.scatter(
        -np.radians(convert_angle(mplets.table[current_mask]["MEDIAN_GLON"])),
        np.radians(mplets.table[current_mask]["MEDIAN_GLAT"]),
        s=scattersize,
        marker="^",
        zorder=2,
        color="c",
        label=r"$N=5, dt < 3\mathrm{s}$",
    )

    ax.fill_between(
        np.linspace(-np.pi, np.pi, 1000),
        np.radians(-5),
        np.radians(5),
        # color="k",
        alpha=0.2,
        label=r"$\mathrm{GLAT}: \pm 5\mathrm{deg}$",
    )
    ax.grid(True)
    ax.set_xlabel(r"MEDIAN_GLON [deg]")
    ax.set_ylabel(r"MEDIAN_GLAT [deg]")
    fig.suptitle(
        r"Skymap of multiplets, TeVCat sources ($\mathrm{distance} < 0.2\mathrm{deg}$) excluded"
    )
    ax.tick_params(grid_alpha=0.1, colors="gray", zorder=3, labelsize="x-small")
    fig.legend(loc="lower right")
    fig.savefig("testing/figures/combined/aitoff.pdf")


if __name__ == "__main__":
    main()
