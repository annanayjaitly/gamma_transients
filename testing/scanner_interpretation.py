# from gamma_transients import core
import dill
from os import cpu_count
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures as cf

from astropy.table import Table, Column, Row
from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.spatial import KDTree
from scipy.stats import expon
from scipy.special import factorial

import numpy as np
from matplotlib import figure, axes, colors
import pandas as pd
import numexpr

from tevcat import Source
import healpy as hp
from gammapy.data import DataStore, Observation
from gammapy.maps import WcsNDMap
from gammapy.datasets import MapDataset


# from astroquery.simbad import Simbad
import sys, os


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
def convert_angle(angle):
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


def cat2hpx(lon, lat, nside, radec=True):
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
    path : str
        Path to the multiplets file

    table : pandas.DataFrame
        Table containing the multiplets

    Methods
    -------

    loadMpletsBare()
        Load the multiplets from the multiplets directory created by gt_scanner

    addCoordinateInfo()
        Add coordinate information to the table: MEAN_RA, MEAN_DEC, SkyCoord, MEAN_GLAT, MEAN_GLON

    getAitoffFigure(bounds_deg: list = [])
        Plot the multiplets on an aitoff projection. If bounds_deg is given (as passed to gt_scanner), plot horizontal lines as well.

    searchTeVCat(sources: list[Source])
        Search TeVCat for the neares sources to each multiplet using KDTree nearest neighbor search.

    addSourceInfo(identified_sources: list[Source], distances: list[float])
        Add the source information to the mplet_table.

    """

    def __init__(self, path: str) -> None:
        self.paths = [path]
        self.loadMpletsBare(path)

    def inclSimbadSources(self, sources: list[str]) -> None:
        self.incl_simbad_sources += sources

    def loadMpletsBare(self, path) -> None:
        if path not in self.paths:
            self.paths.append(path)
            # try except
        with open(path + "/multiplets.pkl", "rb") as f:
            self.table = dill.load(f)
        self.table.sort("Nmax")

    def addCoordinateInfo(self) -> None:
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
        self.table["TEVCAT_SOURCE_NAME"] = [
            source.canonical_name for source in identified_sources
        ]
        self.table["TEVCAT_SOURCE_TYPE"] = [
            source.source_type_name for source in identified_sources
        ]
        self.table["TEVCAT_DISTANCES_DEG"] = distances

    def __len__(self) -> int:
        return len(self.table)

    def appendMultiplets(self, *others):
        df = pd.concat(
            [self.table.to_pandas(), *[other.table.to_pandas() for other in others]],
            ignore_index=True,
        )
        self.table = Table.from_pandas(df)
        self.table.sort("OBS_ID")
        self.addCoordinateInfo()

    def createReduced(
        self,
        min_source_dist_deg: float = 0.2,
        galactic_plane_halfwidth_deg: float = 5.0,
    ):
        noTeVCat_mask = self.table["TEVCAT_DISTANCES_DEG"] >= min_source_dist_deg
        xgal_mask = np.abs(self.table["MEDIAN_GLAT"]) > galactic_plane_halfwidth_deg

        self.reduced = self.table[noTeVCat_mask * xgal_mask]
        print(f"Reducement of {len(self)} to {len(self.reduced)} rows.")

    def objectifyColumns(self) -> None:
        for name in ["ID", "RA", "DEC", "TIME", "ENERGY"]:
            objectifyColumn(self.table, name)

    def setReduced(self, reduced: Table):
        self.reduced = reduced

    def addRMS(self):
        # self.table["ANGULAR_MEASURE_DEG"] = [
        #     np.mean(
        #         sphere_dist(row["RA"], row["DEC"], row["MEDIAN_RA"], row["MEDIAN_DEC"])
        #     )
        #     for row in self.table
        # ]
        measure = []
        workers = int(cpu_count() / 3)

        with cf.ProcessPoolExecutor(workers) as ex:
            future_rms = [
                ex.submit(
                    np.mean, row["RA"], row["DEC"], row["MEDIAN_RA"], row["MEDIAN_DEC"]
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


class Reduced:
    def __init__(self, path: str) -> None:
        with open(path, "rb") as f:
            self.reduced = dill.load(f)

    def loadObservations(self):
        self.datastores = getDataStores()
        self.observation_per_ds = [
            ds.get_observations(
                np.unique(self.reduced["OBS_ID"].data[self.reduced["DS_INDEX"] == i]),
                required_irf="all-optional",
            )
            for i, ds in enumerate(self.datastores)
        ]

    def loadNavtable(self, navpath: str = None):
        try:
            with open(navpath, "rb") as f:
                self.navtable = dill.load(f)
        except:
            print("Navtab not found, making from scratch.")
            self.navtable = self.makeNavigationTable()

    def makeNavigationTable(self):
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

    def addLambdaRatioSignificace(self):
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
        workercount = int(cpu_count() / 3)

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
        distances = []

        workercount = int(cpu_count() / 3)

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
        ## SUDDEN THOUGHT: compare with regular zenith distribution, maybe with the cumulative distributions // Smirnov test
        alt_pnt = []
        for row in tqdm(self.navtable):
            alt = self.observation_per_ds[row["DS_INDEX"]][
                int(row["OBS_INDEX_WITHIN_DS"])
            ].obs_info["ALT_PNT"]
            alt_pnt.append(alt)

        self.reduced["ALT_PNT"] = alt_pnt

    def addExposureCorrectedP(self, exposure: WcsNDMap) -> None:
        corrected_P = []
        runcount = get_total_runcount(self.datastores)
        with cf.ProcessPoolExecutor(int(cpu_count() / 3)) as ex:
            future_expfactors = [
                ex.sumit(get_bell_ratio(exposure, row["SkyCoord"]))
                for row in tqdm(self.reduced)
            ]
        for future in future_expfactors:
            try:
                result = future.result()
            except Exception as e:
                print(f"Error: {e}")
                result = np.nan
            corrected_P.append(result * runcount / result)

        self.exposure["EXP_CORRECTED_P"] = corrected_P


def pxl_distance(i1: int, j1: int, i2: int, j2: int):
    return np.sqrt(np.square(i1 - j1) + np.square(i2 - j2))


def get_contained_indices(
    Nh: int, Nv: int, coord=tuple[int], center: tuple[int] = None
):
    if not center:
        center = (int(Nh / 2), int(Nv / 2))
    x, y = np.meshgrid(np.arange(Nh), np.arange(Nv))
    distances = np.sqrt(np.square(x - coord[0]) + np.square(y - coord[1]))
    radius = pxl_distance(*center, *coord)

    contained_indices = np.where(distances <= radius)
    return contained_indices


def get_bell_ratio(exposure: WcsNDMap, signal: SkyCoord):
    spatial_exposure = exposure.sum_over_axes(keepdims=False)
    contained_idx = get_contained_indices(*spatial_exposure.data.shape, signal)
    normalization = np.sum(spatial_exposure.data)
    contained_exposure = np.sum(spatial_exposure.get_by_idx(contained_exposure))
    return 1 - contained_exposure / normalization


def get_total_runcount(ds: list[DataStore]):
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
    sets = [set(c) for c in containers]
    for i, myset in enumerate(sets):
        if item in myset:
            return i
    raise LookupError("Item is not in any container. Please ensure it is.")


def getDataStores():
    hess1_datastore = DataStore.from_dir("$HESS1")
    hess1u_datastore = DataStore.from_dir("$HESS1U")

    return [hess1_datastore, hess1u_datastore]


def main_fits():
    print("Loading mplets")
    with open("testing/pkl_jugs/n4/mplets.pkl", "rb") as f:
        mplets: Multiplets = dill.load(f)

    print("Got mplets, getting datastores.")
    datastores = getDataStores()
    print("Creating reduced dataset")
    mplets.createReduced()
    print(
        f"unique obs ids in reduced dataset: {len(np.unique(mplets.reduced['OBS_ID'].data))}"
    )

    with open("testing/pkl_jugs/n4/reduced.pkl", "wb") as f:
        dill.dump(mplets.reduced, f)

    print("Adding local background rate to mplets.reduced")
    mplets.addExponentialDtFits(
        datastores, navpath="testing/pkl_jugs/n4/navigationtable.pkl"
    )

    ## estimation based on poisson
    ## estimation based on simulation, maybe incorportate in addLocalBackgroundRate since you have the observations there
    # with open("testing/pkl_jugs/reduced_with_bkg.pkl", "wb") as f:
    #     dill.dump(mplets.reduced, f)

    mplets.addLambdaRatioSignificace()
    with open("testing/pkl_jugs/reduced_with_significance.pkl", "wb") as f:
        dill.dump(mplets.reduced, f)


def main_signifiance():
    print("Loading mplets")
    with open("testing/pkl_jugs/mplets.pkl", "rb") as f:
        mplets: Multiplets = dill.load(f)

    with open("testing/pkl_jugs/reduced_with_bkg.pkl", "rb") as f:
        mplets.setReduced(dill.load(f))
        print(f"Reduced length: {len(mplets.reduced)}")

    mplets.addLambdaRatioSignificace()

    print("dumping")
    with open("testing/pkl_jugs/reduced_with_significance.pkl", "wb") as f:
        dill.dump(mplets.reduced, f)


def main_fixparam():
    with open("testing/pkl_jugs/reduced_with_bkg.pkl", "rb") as f:
        reduced: Table = dill.load(f)

    mplet_fitparams = reduced["MPLET_DT_LAMBDA"]
    print(f"mplet fitparams sample {mplet_fitparams[0]}")

    lamb = [1.0 / par[1] for par in mplet_fitparams]
    reduced.remove_column("MPLET_DT_LAMBDA")
    reduced["MPLET_DT_LAMBDA"] = lamb
    with open("testing/pkl_jugs/reduced_with_bkg.pkl", "wb") as f:
        dill.dump(reduced, f)


def setup_mplets():
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
    paths = [
        f"/lustre/fs22/group/hess/user/wybouwt/full_scanner_survey/nmax4_da_increased/{version}/{band}"
        for band in bands
        for version in versions
    ]
    # mplets = scani.Multiplets(paths[0])
    # mplets.appendMultiplets(*[scani.Multiplets(path) for path in paths[1:]])
    mplet_list = [Multiplets(path) for path in paths]
    Nmax_list = [np.unique(mplets.table["Nmax"].data) for mplets in mplet_list]

    for i in range(len(Nmax_list)):
        print(i, Nmax_list[i])

    # unicorns = [9, 18]
    # unicorns = [7, 16, 17, 24, 25, 32]
    for mplet in mplet_list:
        mplet.objectifyColumns()

    mplets = mplet_list[0]
    mplets.appendMultiplets(*mplet_list[1:])

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

    print("Dumping the mplets object.")
    with open("testing/pkl_jugs/n4/mplets.pkl", "wb") as f:
        dill.dump(mplets, f)

    print("Creating reduced dataset")
    mplets.createReduced()
    print(
        f"unique obs ids in reduced dataset: {len(np.unique(mplets.reduced['OBS_ID'].data))}"
    )
    print("Dumping bare reduced")
    with open("testing/pkl_jugs/n4/reduced_bare.pkl", "wb") as f:
        dill.dump(mplets.reduced, f)


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
    generate_new_mplets = True

    if generate_new_mplets:
        setup_mplets()
