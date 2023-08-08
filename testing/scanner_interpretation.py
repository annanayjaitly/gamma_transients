from gamma_transients import core
import dill
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from scipy.spatial import KDTree
import numpy as np
from tevcat import Source
from matplotlib import figure, axes, colors
import healpy as hp
import pandas as pd


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


def cat2hpx(lon, lat, nside, radec=True):
    """
    Convert a catalogue to a HEALPix map of number counts per resolution
    element. (https://stackoverflow.com/a/50495134)

    Parameters
    ----------
    lon, lat : (ndarray, ndarray)
        Coordinates of the sources in degree. If radec=True, assume input is in the icrs
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
        eq = SkyCoord(lon, lat, frame="icrs", unit="deg")
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
        self.addCoordinateInfo()

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
            frame="icrs",
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

    def searchTeVCat(
        self, sources: list[Source]
    ) -> (list[Source], list[float], list[int]):
        coordinates = np.asarray(
            [[source.getICRS().ra.deg, source.getICRS().dec.deg] for source in sources]
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

        return identified_sources, distances, indices

    def addTevCatSourceInfo(
        self, identified_sources: list[Source], distances: list[float]
    ) -> None:
        self.table["TEVCAT_SOURCE"] = identified_sources
        self.table["TEVCAT_DISTANCES_DEG"] = distances

    def __len__(self) -> int:
        return len(self.table)

    def appendMultiplets(self, *others):
        df = pd.concat(
            [self.table.to_pandas(), *[other.table.to_pandas() for other in others]],
            ignore_index=True,
        )
        self.table = Table.from_pandas(df)
        self.table.sort("Nmax")

    def objectifyColumns(self) -> None:
        for name in ["ID", "RA", "DEC", "TIME", "ENERGY"]:
            objectifyColumn(self.table, name)

    def __getitem__(self, index):
        return self.table[index]


def main():
    from os import getcwd

    band = "test"
    band_bounds = [-80, -78]

    print(f"Current working directory: {getcwd()}")

    # print("Loading TeVCat.")
    # tevcat = TeVCat()
    # sources = tevcat.getTeVCatSources()

    path = f"/lustre/fs22/group/hess/user/wybouwt/full_scanner_survey/{band}"

    print("Initializing Multiplets.")
    mplets = Multiplets(path)
    print(f"Found {len(mplets)} multiplets.")

    # print("Creating figure.")
    # fig, ax = mplets.getAitoffFigure(bounds_deg=band_bounds)

    # fig.savefig(f"testing/figures/multiplets_aitoff_{band}.png", facecolor="white")
    # print("Done")

    print(mplets.dataframe["OBS_ID"])
    print(np.unique(mplets.dataframe["Nmax"]))


if __name__ == "__main__":
    main()
