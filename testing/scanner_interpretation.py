# from gamma_transients import core
import dill
from astropy.table import Table, Column, Row
from astropy.coordinates import SkyCoord
from scipy.spatial import KDTree
from scipy.stats import expon
import numpy as np
from tevcat import Source
from matplotlib import figure, axes, colors
import healpy as hp
import pandas as pd
import numexpr
from gammapy.data import DataStore, Observation
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures as cf

# from astroquery.simbad import Simbad
import sys, os


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


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

    def __init__(self, path: str, simbad_sources_init: bool = True) -> None:
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

    def __getitem__(self, index):
        return self.table[index]

    def setReduced(self, reduced: Table):
        self.reduced = reduced

    # def calcRoughSignificance(self)
    def addLocalBackgroundRate(
        self, datastores: list[DataStore], radius_deg: float = 0.1, navpath: str = None
    ):
        """
        ==============================
        Creating the navigation table
        ==============================
        """
        try:
            with open(navpath, "rb") as f:
                navigation_table = dill.load(f)
            print(f"Loaded navtable from {navpath}.")
            print("Getting ds observations.")
            blockPrint()
            observation_per_ds = [
                ds.get_observations(
                    np.unique(self.reduced["OBS_ID"].data),
                    skip_missing=True,
                    required_irf="all-optional",
                )
                for ds in datastores
            ]
            enablePrint()
            print("Got observations")

        except TypeError:
            print("Not loading navtable.")

            print("Getting ds observations.")

            observation_per_ds = [
                ds.get_observations(
                    np.unique(
                        self.reduced["OBS_ID"].data[self.reduced["DS_INDEX"] == i]
                    ),
                    required_irf="all-optional",
                )
                for i, ds in enumerate(datastores)
            ]
            print("Got observations")

            # mapping obs id to index within the relevant datastore. Note that this is non-injective
            obs_id_to_index_per_ds = dict()
            for obs in observation_per_ds:
                obs_id_to_index_per_ds.update(
                    {int(obs.ids[index]): index for index in range(len(obs))}
                )

            with open("testing/pickles/obs_id_to_index_per_ds_dict.pkl", "wb") as f:
                dill.dump(obs_id_to_index_per_ds, f)

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
                data=list(
                    map(obs_id_to_index_per_ds.get, navigation_table["OBS_ID"].data)
                ),
                dtype=int,
            )
            navigation_table.add_column(temp)

            with open("testing/pickles/navigationtable.pkl", "wb") as f:
                dill.dump(navigation_table, f)

        """
        =========================
        Getting the photon rates
        =========================
        """
        print("Getting photon rates.")
        expon_fit_parameters = []
        # error_counter = 0
        # for row in tqdm(navigation_table):
        #     obs = observation_per_ds[row["DS_INDEX"]][int(row["OBS_INDEX_WITHIN_DS"])]
        #     run_distances = sphere_dist(
        #         obs.events.table["RA"].data,
        #         obs.events.table["DEC"].data,
        #         self.reduced[row["MPLET_INDEX"]]["MEDIAN_RA"],
        #         self.reduced[row["MPLET_INDEX"]]["MEDIAN_DEC"],
        #     )
        #     #     # expon_fit_parameters.append(
        #     #     #     len(obs.events.table[run_distances < radius_deg])
        #     #     #     / obs.obs_info["LIVETIME"]
        #     #     # )
        #     try:
        #         parameters = expon.fit(
        #             np.diff(
        #                 np.sort(obs.events.table[run_distances < radius_deg]["TIME"])
        #             )
        #         )
        #     except ValueError:
        #         print(f"Empty array around the multiplet for OBS_ID {obs.obs_id}")
        #         parameters = (np.nan, np.nan)
        #         error_counter += 1
        #     expon_fit_parameters.append(parameters)
        # self.reduced["LOCAL_BKG_PHOT_DT_EXPON_FIT"] = expon_fit_parameters

        # print(f"Getting mplet median right ascensions")
        # MPLET_MEDIAN_RA = [
        #     self.reduced[row["MPLET_INDEX"]]["MEDIAN_RA"]
        #     for row in tqdm(navigation_table)
        # ]
        # print(f"Getting mplet median declinations")
        # MPLET_MEDIAN_DEC = [
        #     self.reduced[row["MPLET_INDEX"]]["MEDIAN_DEC"]
        #     for row in tqdm(navigation_table)
        # ]
        # print(f"Getting flattened observations")
        # OBSERVATIONS_FLAT = [
        #     observation_per_ds[row["DS_INDEX"]][int(row["OBS_INDEX_WITHIN_DS"])]
        #     for row in tqdm(navigation_table)
        # ]

        # print("Making rough masks")
        # rough_mask = [obs.event.table["RA"] < ]

        # print("Getting run ra")
        # RA = [obs.events.table["RA"] for obs in tqdm(OBSERVATIONS_FLAT)]
        # print("Getting run dec")
        # DEC = [obs.events.table["DEC"] for obs in tqdm(OBSERVATIONS_FLAT)]
        # print("Getting run arrival times")
        # TIME = [obs.events.table["TIME"] for obs in tqdm(OBSERVATIONS_FLAT)]

        # # for row in tqdm(navigation_table):
        # #     obs = observation_per_ds[row["DS_INDEX"]][int(row["OBS_INDEX_WITHIN_DS"])]
        # #     RA.append(obs.events.table["RA"].data)
        # #     DEC.append(obs.events.table["DEC"].data)
        # #     MPLET_MEDIAN_RA.append(self.reduced[row["MPLET_INDEX"]]["MEDIAN_RA"])
        # #     MPLET_MEDIAN_DEC.append(self.reduced[row["MPLET_INDEX"]]["MEDIAN_DEC"])
        # print("Got cf input arrays")

        # from os import cpu_count

        # print("Multiprocessing angular distances of run photons to mplet")
        # with cf.ProcessPoolExecutor(int(cpu_count() / 4)) as executor:
        #     future_rundistances = [
        #         executor.submit(sphere_dist, ra, dec, medra, meddec)
        #         for ra, dec, medra, meddec in zip(
        #             RA, DEC, MPLET_MEDIAN_RA, MPLET_MEDIAN_DEC
        #         )
        #     ]
        # print("Finished multiprocessing distances.")
        # distances = [future.result() for future in future_rundistances]
        # local_dt = [
        #     np.diff(np.sort(TIME[rd < radius_deg]))
        #     for obs, rd in zip(OBSERVATIONS_FLAT, distances)
        # ]
        # print("Multiprocessing expon fitting")
        # with cf.ProcessPoolExecutor(int(cpu_count() / 4)) as executor:
        #     future_fitparams = [executor.submit(expon.fit, data) for data in local_dt]

        # print("Finished fitting dt.")
        # for future in future_fitparams:
        #     try:
        #         result = future.result()
        #     except Exception as e:
        #         print(f"Error: {e}")
        #         result = (np.nan, np.nan)
        #     expon_fit_parameters.append(result)

        from os import cpu_count

        workercount = int(cpu_count() / 4)
        with cf.ProcessPoolExecutor(workercount) as ex:
            future_fitparams = [
                ex.submit(
                    worker,
                    observation_per_ds[row["DS_INDEX"]][
                        int(row["OBS_INDEX_WITHIN_DS"])
                    ],
                    self.reduced[row["MPLET_INDEX"]]["MEDIAN_RA"],
                    self.reduced[row["MPLET_INDEX"]]["MEDIAN_DEC"],
                )
                for row in navigation_table
            ]
        for future in future_fitparams:
            try:
                result = future.result()
            except Exception as e:
                print(f"Error: {e}")
                result = (np.nan, np.nan)
            expon_fit_parameters.append(result)

        self.reduced["LOCAL_BKG_PHOT_DT_EXPON_FIT"] = expon_fit_parameters

        with open("testing/pickles/reduced_with_bkg.pkl", "wb") as f:
            dill.dump(self.reduced, f)

        # return error_counter


def worker(obs: Observation, medra: float, meddec: float, radius_deg=0.1):
    run_distances = sphere_dist(
        obs.events.table["RA"].data,
        obs.events.table["DEC"].data,
        medra,
        meddec,
    )

    parameters = expon.fit(
        np.diff(np.sort(obs.events.table[run_distances < radius_deg]["TIME"]))
    )

    return parameters


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


def main():
    print("Loading mplets")
    with open("testing/pickles/mplets.pkl", "rb") as f:
        mplets: Multiplets = dill.load(f)

    print("Got mplets, getting datastores.")
    datastores = getDataStores()
    print("Creating reduced dataset")
    mplets.createReduced()
    print(
        f"unique obs ids in reduced dataset: {len(np.unique(mplets.reduced['OBS_ID'].data))}"
    )
    print("Adding local background rate to mplets.reduced")
    ec = mplets.addLocalBackgroundRate(
        datastores, navpath="testing/pickles/navigationtable.pkl"
    )
    print(f"error counter: {ec} out of ~25k")

    # fig = figure.Figure()
    # ax = fig.add_subplot()
    # ax.hist(mplets.reduced["LOCAL_BKG_PHOT_RATE"], bins="fd", histtype="step")

    # ax.set_xlabel(r"Local photon background rate [$\mathrm{s}^{-1}$]")
    # ax.set_ylabel(r"Counts")

    # fig.savefig("testing/figures/combined/reduced_phot_bkg_rate.png", facecolor="white")


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
        f"/lustre/fs22/group/hess/user/wybouwt/full_scanner_survey/{version}/{band}"
        for band in bands
        for version in versions
    ]
    # mplets = scani.Multiplets(paths[0])
    # mplets.appendMultiplets(*[scani.Multiplets(path) for path in paths[1:]])
    mplet_list = [Multiplets(path) for path in paths]

    unicorns = [9, 18]
    for j in unicorns:
        mplet_list[j].objectifyColumns()

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

    with open("testing/pickles/mplets.pkl", "wb") as f:
        dill.dump(mplets, f)


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
    generate_new_mplets = False

    if generate_new_mplets:
        setup_mplets()

    main()
