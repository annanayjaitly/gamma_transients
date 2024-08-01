import numpy as np

import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack


from .photon import Photon, PhotonSim
from .smallest_enc_circ import make_circle

from typing import Union


def in_(p, m):
    """
    Checks if photon p is in multiplet m.
    """
    if type(m) == list:
        return p in m
    return p == m


def append_(m, p):
    """
    Appends photon p to multiplet m.
    """
    if type(m) == list:
        return m.copy().append(p)
    return [m, p]


def check(m, dt, da):
    """
    input: (multiplets, dt in nanosec ,da in deg)
    It checks if the photons in the supposed multiplet really satisfy the condition
    on dt and on the angular separation to be a real multiplet.
    """
    points = [(p.ra, p.dec) for p in m]
    return (m[-1].t - m[0].t < dt) and make_circle(points)[2] * 2 < da


def f_N(photons, multiplets, dt, da, Nmin: int = 2, Nmax: Union[int, None] = None, verbose=False):
    """
    input: f(photons,[[p] for p in photons], dt in nanosec, da in deg, Nmin, Nmax)
    The function stops when there are no higher-order multiplets or when Nmax is reached.
    """
    multiplets_cycle = []
    if Nmax is not None:
        N1 = Nmax - 1
        if verbose:
            print("f will be called max. ", Nmax, " times")
    else:
        N1 = Nmax
        if verbose:
            print("f will be called until multiplets are detected")

    if Nmax is None or Nmax > 0:
        for m in multiplets:
            if Nmax is None or len(m) < Nmax:
                for p in photons:
                    if p > m[-1]:
                        m_ = m.copy()
                        m_.append(p)
                        if check(m_, dt, da):
                            multiplets_cycle.append(m_)

        if len(multiplets_cycle) > 0:
            if verbose:
                print(f"f is called -> multiplets number: {len(multiplets_cycle)}")
            return f_N(photons, multiplets_cycle, dt, da, Nmin, Nmax)  # changed N1 to Nmax
        else:
            return multiplets
    else:
        return multiplets


def multiplet_scanner(
    target,
    events_lists,
    dt_threshold,
    r_deg,
    r_area,
    Nmin: int = 2,
    Nmax: Union[int, None] = None,
    verbose: bool = False,
    remove_zero_coordinates: bool = False,
    convert_time = True,
):
    """Scans for Nmax-multiplets in specified sky region around target, blind search not implemented yet.

    Args:
        target (SkyCoord): Target to scan around
        events_lists (List of gammapy EventLists): one EventList for each observation
        dt_threshold (float): dt threshold for multiplets in ns
        r_deg (float): max angular separation between multiplet members
        r_area (float): scan radius around target
        Nmin (integer): Nmin-multiplets to look for.
        Nmax (integer): Nmax-multiplets to look for.
        verbose (bool, optional): Defaults to False.

    Returns:
        [astropy.Table]: multiplets_table
    """

    rows_multiplets = []

    doublets_save = []

    n_scanned = 0
    hits = 0

    da_cut = r_deg

    if not target:
        raise TypeError("Blind search not implemented yet, "
                        "'target' should be specified")

    for counter, event_list in enumerate(events_lists):
        events_table = event_list.table.copy()

        events_table["coord"] = event_list.radec  # create a SkyCoord column
        if convert_time:
            events_table["TIME"] = event_list.time.datetime64
        else:
            events_table["TIME"] = event_list.time

        events_table["fTIME"] = (
            events_table["TIME"].copy().astype(np.float64)
        )  # create a float64 time data column

        events_table = events_table[
            target.separation(events_table["coord"]).degree <= r_area
        ]  # select events within r_area of source

        if remove_zero_coordinates:
            events_table = events_table[(events_table["RA"] != 0.0) & (events_table["DEC"] != 0.0)]

        n_scanned += len(events_table)

        events_table.sort(keys="TIME")  # sort events by timestamp

        photons = [
            Photon(
                t=event["fTIME"],
                ra=event["RA"],
                dec=event["DEC"],
                id_=event["EVENT_ID"],
            )
            for event in events_table
        ]

        t_vector = events_table["fTIME"]

        coord_vector = events_table["coord"]

        dt_matrix = np.triu(
            t_vector - t_vector.reshape(-1, 1), k=1
        )  # create dt_matrix
        sep_matrix = np.triu(
            coord_vector.separation(coord_vector.reshape(-1, 1)).degree, k=1
        )

        doublet_ids = np.argwhere(
            (dt_matrix <= dt_threshold)
            & (dt_matrix > 0)
            & (sep_matrix <= da_cut)
            & (sep_matrix > 0)
        )  # find doublets, correspond to first and last event in multiplet if Nmax >2

        # get indices of first and second part of doublets
        doublet_ids1, doublet_ids2 = doublet_ids[:, 0], doublet_ids[:, 1]
        if verbose:
            print(f"{len(doublet_ids1)} doublets found)\r")
        # print('doublet_ids1, doublet_ids2: ', doublet_ids1, doublet_ids2)
        doublets_save.append(doublet_ids1)
        doublets_save.append(doublet_ids2)

        multiplets_max = f_N(
            photons,
            [
                [photons[id1], photons[id2]]
                for id1, id2 in zip(doublet_ids1, doublet_ids2)
            ],
            dt_threshold,
            r_deg,
            Nmin,
            Nmax,
        )

        if len(multiplets_max) > 0:
            for multiplet in multiplets_max:
                if Nmin <= len(multiplet) and (
                    (Nmax is None) or (len(multiplet) <= Nmax)
                ):
                    names = [
                        name
                        for name in events_table.colnames
                        if len(events_table[name].shape) <= 1
                    ]  # In case table has multidimensional columns, we must remove them
                    events_df = events_table[names].to_pandas()
                    find_photons = events_df.isin(
                        {"EVENT_ID": [photon.id_ for photon in multiplet]}
                    )
                    find_photons = find_photons.any(axis="columns").to_numpy()
                    multiplet_table = events_table[find_photons]

                    dt = multiplet[-1].t - multiplet[0].t
                    da = (
                        make_circle(
                            np.c_[multiplet_table["RA"], multiplet_table["DEC"]]
                        )[2]
                        * 2
                    )
                    if (da < r_deg) & (dt < dt_threshold):
                        hits += 1
                        rows_multiplets.append(
                            [
                                len(multiplet),
                                multiplet_table["OBS_ID"][0],
                                multiplet_table["EVENT_ID"].data,
                                multiplet_table["RA"].data,
                                multiplet_table["DEC"].data,
                                multiplet_table["TIME"].data,
                                multiplet_table["ENERGY"].data,
                                dt,
                                da,
                            ]
                        )

        print(
            f"\rScan no. {counter + 1}/{len(events_lists)} ({100 * (counter + 1) / len(events_lists):2.2f}%) \t| scanned {n_scanned} events so far \t| {Nmax}-multiplets: {hits} hits so far\r",
            end="\r",
        )

    return Table(
        rows=rows_multiplets,
        names=("Nmax", "OBS_ID", "ID", "RA", "DEC", "TIME", "ENERGY", "dt", "da"),
    )


def multiplet_scanner_simulated_table(
    target,
    events_lists,
    dt_threshold,
    r_deg,
    r_area,
    Nmin: int = 2,
    Nmax: Union[int, None] = None,
    verbose=False,
):
    """Scans for Nmax-multiplets in specified sky region around target, blind search not implemented yet.

    Args:
        target (SkyCoord): Target to scan around
        events_lists (List of astropy tables): one EventList for each simulation
        dt_threshold (float): dt threshold for multiplets in ns
        r_deg (float): max angular separation between multiplet members
        r_area (float): scan radius around target
        Nmin (integer): Nmin-multiplets to look for.
        Nmax (integer): Nmax-multiplets to look for.
        verbose (bool, optional): Defaults to False.

    Returns:
        [astropy.Table]: multiplets_table
    """

    rows_multiplets = []

    doublets_save = []

    n_scanned = 0
    hits = 0

    da_cut = r_deg

    if target is not None:
        for counter, event_list in enumerate(events_lists):
            events_table = event_list.copy()
            # print(events_table["OBS_ID"][0])

            events_table["coord"] = SkyCoord(
                ra=events_table["RA"], dec=events_table["DEC"], unit="deg"
            )
            events_table["fTIME"] = (
                events_table["TIME"].copy().astype(np.float64)
            )  # create a float64 time data column

            events_table = events_table[
                target.separation(events_table["coord"]).degree <= r_area
            ]  # select events within r_area of source
            events_table = events_table[
                (events_table["RA"] != 0.0) & (events_table["DEC"] != 0.0)
            ]

            n_scanned += len(events_table)

            events_table.sort(keys="TIME")  # sort events by timestamp

            photons = [
                PhotonSim(t=event["fTIME"], ra=event["RA"], dec=event["DEC"])
                for event in events_table
            ]

            t_vector = events_table["fTIME"]

            coord_vector = events_table["coord"]

            dt_matrix = np.triu(
                t_vector - t_vector.reshape(-1, 1), k=1
            )  # create dt_matrix
            sep_matrix = np.triu(
                coord_vector.separation(coord_vector.reshape(-1, 1)).degree, k=1
            )

            doublet_ids = np.argwhere(
                (dt_matrix <= dt_threshold)
                & (dt_matrix > 0)
                & (sep_matrix <= da_cut)
                & (sep_matrix > 0)
            )  # find doublets, correspond to first and last event in multiplet if Nmax >2

            # get indices of first and second part of doublets
            doublet_ids1, doublet_ids2 = doublet_ids[:, 0], doublet_ids[:, 1]
            if verbose:
                print(f"{len(doublet_ids1)} doublets found)\r")
            # print('doublet_ids1, doublet_ids2: ', doublet_ids1, doublet_ids2)
            doublets_save.append(doublet_ids1)
            doublets_save.append(doublet_ids2)

            multiplets_max = f_N(
                photons,
                [
                    [photons[id1], photons[id2]]
                    for id1, id2 in zip(doublet_ids1, doublet_ids2)
                ],
                dt_threshold,
                r_deg,
                Nmin,
                Nmax,
            )

            if len(multiplets_max) > 0:
                for multiplet in multiplets_max:
                    if Nmin <= len(multiplet) and (
                        (Nmax is None) or (len(multiplet) <= Nmax)
                    ):
                        multiplet_table = Table(
                            rows=[
                                [event.ra, event.dec, event.t] for event in multiplet
                            ],
                            names=("RA", "DEC", "TIME"),
                        )
                        dt = multiplet[-1].t - multiplet[0].t
                        da = (
                            make_circle(
                                np.c_[multiplet_table["RA"], multiplet_table["DEC"]]
                            )[2]
                            * 2
                        )
                        if (da < r_deg) & (dt < dt_threshold):
                            hits += 1
                            rows_multiplets.append(
                                [
                                    len(multiplet),
                                    multiplet_table["RA"].data,
                                    multiplet_table["DEC"].data,
                                    multiplet_table["TIME"].data,
                                    dt,
                                    da,
                                ]
                            )
            if verbose:
                print(
                    f"\rScan no. {counter + 1}/{len(events_lists)} ({100 * (counter + 1) / len(events_lists):2.2f}%) \t| scanned {n_scanned} events so far \t| {Nmax}-multiplets: {hits} hits so far\r",
                    end="\r",
                )

    if len(rows_multiplets) > 0:
        multiplets_table = Table(
            rows=rows_multiplets,
            names=("Nmax", "RA", "DEC", "TIME", "dt", "da"),
        )
        return multiplets_table
    else:
        print("No multiplets found!")
        return Table(
            names=("Nmax", "ID", "RA", "DEC", "TIME", "dt", "da"),
        )


def worker(
    target, event_list, dt_threshold, r_deg, r_area, Nmin: int = 2, Nmax: Union[int, None] = None, remove_zero_coordinates: bool = False
):
    """A single instance of the muktiplet scanner for one observation run, used in parallelization.

    Args:
        target (SkyCoord): Target to scan around
        events_lists (List of astropy tables): one EventList for each simulation
        dt_threshold (float): dt threshold for multiplets in ns
        r_deg (float): max angular separation between multiplet members
        r_area (float): scan radius around target
        Nmin (integer): Nmin-multiplets to look for.
        Nmax (integer): Nmax-multiplets to look for.

    Returns:
        [astropy.Table]: multiplets_table
    """

    rows = []
    da_cut = r_deg
    events_table = event_list.table.copy()
    # print(events_table["OBS_ID"][0])

    events_table["coord"] = event_list.radec  # create a SkyCoord column
    events_table["TIME"] = event_list.time.datetime64
    events_table["fTIME"] = (
        events_table["TIME"].copy().astype(np.float64)
    )  # create a float64 time data column

    events_table = events_table[
        target.separation(events_table["coord"]).degree <= r_area
    ]  # select events within r_area of source

    if remove_zero_coordinates:
        events_table = events_table[(events_table["RA"] != 0.0) & (events_table["DEC"] != 0.0)]

    events_table.sort(keys="TIME")  # sort events by timestamp

    photons = [
        Photon(
            t=event["fTIME"], ra=event["RA"], dec=event["DEC"], id_=event["EVENT_ID"]
        )
        for event in events_table
    ]

    t_vector = events_table["fTIME"]

    coord_vector = events_table["coord"]

    dt_matrix = np.triu(t_vector - t_vector.reshape(-1, 1), k=1)  # create dt_matrix
    sep_matrix = np.triu(
        coord_vector.separation(coord_vector.reshape(-1, 1)).degree, k=1
    )

    doublet_ids = np.argwhere(
        (dt_matrix <= dt_threshold)
        & (dt_matrix > 0)
        & (sep_matrix <= da_cut)
        & (sep_matrix > 0)
    )  # find doublets, correspond to first and last event in multiplet if Nmax >2

    # get indices of first and second part of doublets
    doublet_ids1, doublet_ids2 = doublet_ids[:, 0], doublet_ids[:, 1]
    # print('doublet_ids1, doublet_ids2: ', doublet_ids1, doublet_ids2)

    multiplets_max = f_N(
        photons,
        [[photons[id1], photons[id2]] for id1, id2 in zip(doublet_ids1, doublet_ids2)],
        dt_threshold,
        r_deg,
        Nmin,
        Nmax,
    )

    if len(multiplets_max) > 0:
        for multiplet in multiplets_max:
            if Nmin <= len(multiplet) and ((Nmax is None) or (len(multiplet) <= Nmax)):
                # events_df = events_table.to_pandas()
                names = [
                    name
                    for name in events_table.colnames
                    if len(events_table[name].shape) <= 1
                ]  # In case table has multidimensional columns, we must remove them
                events_df = events_table[names].to_pandas()
                find_photons = events_df.isin(
                    {"EVENT_ID": [photon.id_ for photon in multiplet]}
                )
                find_photons = find_photons.any(axis="columns").to_numpy()
                multiplet_table = events_table[find_photons]

                # print(multiplet_table)

                dt = multiplet[-1].t - multiplet[0].t
                da = (
                    make_circle(np.c_[multiplet_table["RA"], multiplet_table["DEC"]])[2]
                    * 2
                )
                if (da < r_deg) & (dt < dt_threshold):
                    rows.append(
                        [
                            len(multiplet),
                            multiplet_table["OBS_ID"][0],
                            np.asarray(multiplet_table["EVENT_ID"].data),
                            np.asarray(multiplet_table["RA"].data),
                            np.asarray(multiplet_table["DEC"].data),
                            np.asarray(multiplet_table["TIME"].data),
                            np.asarray(multiplet_table["ENERGY"].data),
                            dt,
                            da,
                        ]
                    )

    if len(rows) > 0:
        return Table(
            rows=rows,
            names=("Nmax", "OBS_ID", "ID", "RA", "DEC", "TIME", "ENERGY", "dt", "da"),
        )
    return Table(
        names=("Nmax", "OBS_ID", "ID", "RA", "DEC", "TIME", "ENERGY", "dt", "da")
    )


import concurrent.futures as cf


def scanner_parallel(
    target,
    events_lists,
    dt_threshold,
    r_deg,
    r_area,
    Nmin: int = 2,
    Nmax: Union[int, None] = None,
    verbose=False,
):
    """Runs the multiplet scanner in parallel for each observation run. Initialises the worker function for each run and stacks the results when all are done.

    Args:
        target (SkyCoord): Target to scan around
        events_lists (List of astropy tables): one EventList for each simulation
        dt_threshold (float): dt threshold for multiplets in ns
        r_deg (float): max angular separation between multiplet members
        r_area (float): scan radius around target
        Nmin (integer): Nmin-multiplets to look for.
        Nmax (integer): Nmax-multiplets to look for.
        verbose (bool, optional): Defaults to False.

    Returns:
        [astropy.Table]: multiplets_table
    """
    with cf.ProcessPoolExecutor() as executor:
        future_multiplets = [
            executor.submit(
                worker, target, event_list, dt_threshold, r_deg, r_area, Nmin, Nmax
            )
            for event_list in events_lists
        ]
    if len(future_multiplets) > 0:
        return vstack(
            [
                future.result()
                for future in future_multiplets
                if future.result() is not None
            ]
        )
    else:
        print("No multiplets found!")
        return None


#####################
# Plotting functions #
#####################


def triplet_hist2d_plotter(table, bins=50, position_src=None, save=False):
    """Plots a 2D histogram of the triplet (RADEC coords) distribution.
    if position_src is given, the histogram is centered on the source position.

    Args:
        table (astropy.table): the table obtained by running the scanner on the events list(s)
        bins (_type_): the number of bins for the histogram. Defaults to 50.
        position_src (_type_, optional): Source position. Defaults to None.
    """
    figure, ((ax0, ax1, ax2)) = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        gridspec_kw={"width_ratios": [1, 1, 1], "height_ratios": [1]},
    )

    if position_src is not None:
        ra_avg = np.average(table["RA"], axis=1) - position_src.ra.deg

        dec_avg = np.average(table["DEC"], axis=1) - position_src.dec.deg

        ra_1, dec_1 = (
            table["RA"][:, 0] - position_src.ra.deg,
            table["DEC"][:, 0] - position_src.dec.deg,
        )

        ra_2, dec_2 = (
            table["RA"][:, -1] - position_src.ra.deg,
            table["DEC"][:, -1] - position_src.dec.deg,
        )

        range_ra = np.max(
            np.abs(
                np.array(
                    np.max(table["RA"]) - position_src.ra.deg,
                    np.min(table["RA"]) - position_src.ra.deg,
                )
            )
        )
        range_dec = np.max(
            np.abs(
                np.array(
                    np.max(table["DEC"]) - position_src.dec.deg,
                    np.min(table["DEC"]) - position_src.dec.deg,
                )
            )
        )
        range_var = np.max([range_ra, range_dec])
        # rint(table['RA'].shape[1])
        range_ = [[-range_var, range_var], [-range_var, range_var]]

    else:
        ra_avg = np.average(table["RA"], axis=1)

        dec_avg = np.average(table["DEC"], axis=1)

        ra_1, dec_1 = table["RA"][:, 0], table["DEC"][:, 0]

        ra_2, dec_2 = table["RA"][:, -1], table["DEC"][:, -1]

        range_ra = np.max(np.abs(np.array(np.max(table["RA"]), np.min(table["RA"]))))
        range_dec = np.max(np.abs(np.array(np.max(table["DEC"]), np.min(table["DEC"]))))
        range_var = np.max([range_ra, range_dec])
        print(range_var)
        range_ = None

    N = table["RA"].shape[1]

    counts, xedges, yedges, im0 = ax0.hist2d(
        ra_avg, dec_avg, bins=bins, cmap=plt.cm.jet, range=range_
    )

    ax0.set_title(f"Averaged Coordinates of {N}-Multiplets", weight="bold")
    ax0.set_ylabel("DEC", weight="bold")
    ax0.set_xlabel("RA", weight="bold")

    counts, xedges, yedges, im1 = ax1.hist2d(
        ra_2,
        dec_2,
        bins=bins,
        cmap=plt.cm.jet,
        # norm=norm
        range=range_,
    )

    ax1.set_title("Last", weight="bold")
    ax1.set_ylabel("DEC", weight="bold")
    ax1.set_xlabel("RA", weight="bold")

    counts, xedges, yedges, im2 = ax2.hist2d(
        ra_1,
        dec_1,
        bins=bins,
        cmap=plt.cm.jet,
        # norm=norm
        range=range_,
    )

    ax2.set_title("First", weight="bold")
    ax2.set_ylabel("DEC", weight="bold")
    ax2.set_xlabel("RA", weight="bold")

    figure.colorbar(im0, ax=ax0)
    figure.colorbar(im1, ax=ax1)
    figure.colorbar(im2, ax=ax2)

    plt.suptitle(f"{N}-Multiplets", weight="bold")
    plt.tight_layout(pad=1.5)
    plt.show()


from matplotlib import colors


def da_dt_hist2d_plotter(table, bins=50):
    """plots a 2D histogram of the da vs dt distribution

    Args:
        table (astropy.table): obtained by running the scanner on the events list(s)
        bins (_type_): number of bins for the histogram. Defaults to 50.
    """
    figure, ax0 = plt.subplots(1, 1, figsize=(7, 6))

    log10dt = np.log10(table["dt"])

    da = table["da"]

    counts, xedges, yedges, im0 = ax0.hist2d(log10dt, da, bins=bins, cmap=plt.cm.jet)

    ax0.set_title("da vs dt", weight="bold")
    ax0.set_ylabel("da", weight="bold")
    ax0.set_xlabel("log10[dt/ns]", weight="bold")
    ax0.set_facecolor("dimgrey")

    figure.colorbar(im0, ax=ax0, norm=colors.LogNorm())

    plt.show()


def triplet_hist2d_saver(table, bins, position_src=None):
    """Returns a 2D histogram of the triplet distribution as a figure object.
    If position_src is given, the histogram is centered on the source position.

    Args:
        table (astropy.table): the table obtained by running the scanner on the events list(s)
        bins (_type_): the number of bins for the histogram. Defaults to 50.
        position_src (_type_, optional): Source position. Defaults to None.

    Returns:
        plt.figure: the figure object
    """
    figure, ax0 = plt.subplots(1, 1, figsize=(5, 4))

    if position_src is not None:
        ra_avg = np.average(table["RA"], axis=1) - position_src.ra.deg

        dec_avg = np.average(table["DEC"], axis=1) - position_src.dec.deg

        range_ra = np.max(
            np.abs(
                np.array(
                    np.max(table["RA"]) - position_src.ra.deg,
                    np.min(table["RA"]) - position_src.ra.deg,
                )
            )
        )
        range_dec = np.max(
            np.abs(
                np.array(
                    np.max(table["DEC"]) - position_src.dec.deg,
                    np.min(table["DEC"]) - position_src.dec.deg,
                )
            )
        )
        range_var = np.max([range_ra, range_dec])
        # rint(table['RA'].shape[1])
        range_ = [[-range_var, range_var], [-range_var, range_var]]

    else:
        ra_avg = np.average(table["RA"], axis=1)

        dec_avg = np.average(table["DEC"], axis=1)

        range_ = None

    N = table["RA"].shape[1]

    _,_,_, im0 = ax0.hist2d(
        ra_avg, dec_avg, bins=bins, cmap=plt.cm.jet, range=range_
    )

    ax0.set_title(f"Averaged Coordinates of {N}-Multiplets", weight="bold")
    ax0.set_ylabel("DEC", weight="bold")
    ax0.set_xlabel("RA", weight="bold")

    figure.colorbar(im0, ax=ax0)
    return figure
