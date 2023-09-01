import dill
import secrets

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pathlib import Path
from argparse import ArgumentParser

from astropy import units as u
from astropy.coordinates import SkyCoord

from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.datasets import MapDataset
from gammapy.maps import WcsGeom, MapAxis
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker

argp = ArgumentParser()

argp.add_argument(
    "-n", action="store", dest="name", required=True, help="source name for folders"
)
argp.add_argument(
    "-i", action="store", dest="input_data_path", required=True, help="input_data_path"
)
argp.add_argument(
    "-o",
    action="store",
    dest="output_storage_path",
    required=True,
    help="output_storage_path",
)
argp.add_argument(
    "-l",
    action="store",
    dest="load",
    required=False,
    default=1,
    help="load dataset? set to 0 to generate (real) dataset pickle files from scratch (which adds ~20 min per job)",
    type=int,
)
argp.add_argument(
    "-s",
    action="store",
    dest="N_sets",
    required=True,
    help="No. of simulated sets to generate and analyse per job, keep ~10",
    type=int,
)
argp.add_argument(
    "-c",
    action="store",
    dest="config_path",
    required=False,
    help="path to gammapy analysis config file, needed if load=0",
    type=int,
)

argp.add_argument(
    "--RA", action="store", dest="src_RA", required=True, help="Source RA", type=float
)
argp.add_argument(
    "--DEC",
    action="store",
    dest="src_DEC",
    required=True,
    help="Source DEC",
    type=float,
)
argp.add_argument(
    "--da-cut",
    action="store",
    dest="da_cut",
    required=False,
    default=0.5,
    type=float,
    help="max da_cut applied to speed up search",
)
argp.add_argument(
    "--dt-max",
    action="store",
    dest="dt_threshold",
    required=False,
    default=1e9,
    type=float,
    help="max dt_cut applied to speed up search",
)
argp.add_argument(
    "--multi_N",
    action="store",
    dest="multi_N",
    required=False,
    default=2,
    type=int,
    help="N for N-multiplet to search for",
)
argp.add_argument(
    "--r-area-cut",
    action="store",
    dest="r_area",
    required=False,
    default=1,
    type=float,
    help="max r_area around source to search in",
)

argp.add_argument(
    "--OBS_ID",
    action="store",
    dest="OBS_ID",
    required="false",
    default=None,
    type=int,
    help="If given, only select this run instead of based on pointing coordinates.",
)

args = argp.parse_args()

if args.config_path is not None:  # TODO, IMPLEMENT
    raise Exception(
        "Feature not implemented, please do not use '--config-file' parameter in current version"
    )

load, N_sets, da_cut, dt_threshold, multi_N, r_area, src_DEC, src_RA = (
    args.load,
    args.N_sets,
    args.da_cut,
    args.dt_threshold,
    args.multi_N,
    args.r_area,
    args.src_DEC,
    args.src_RA,
)

data_store = DataStore.from_dir(args.input_data_path)
dump_path = Path(args.output_storage_path) / "real_dataset_dumps" / f"{args.name}"
dump_path_lists = (
    Path(args.output_storage_path)
    / "MC_dumps"
    / f"{args.name}"
    / f"N{multi_N}_da({da_cut})_dt({dt_threshold / 1e6} ms)"
)

position_src = SkyCoord(src_RA, src_DEC, unit="deg", frame="icrs").icrs

import warnings

warnings.filterwarnings("ignore")


########## PREPARING DATA W/ GAMMAPY ###################################################


# Reduced IRFs are defined in true energy (i.e. not measured energy).
def make_dataset_from_config():
    energy_axis = MapAxis.from_energy_bounds(0.4, 10, 10, unit="TeV")

    geom = WcsGeom.create(
        skydir=(position_src.ra.degree, position_src.dec.degree),
        binsz=0.02,
        width=(5, 5),
        frame="icrs",
        proj="CAR",
        axes=[energy_axis],
    )

    # Reduced IRFs are defined in true energy (i.e. not measured energy).
    energy_axis_true = MapAxis.from_energy_bounds(
        0.2, 20, 30, unit="TeV", name="energy_true"
    )

    stacked = MapDataset.create(
        geom=geom, energy_axis_true=energy_axis_true, name="sc-stacked"
    )

    offset_max = 3 * u.deg
    maker = MapDatasetMaker()
    maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=offset_max)

    circle = CircleSkyRegion(center=position_src, radius=0.5 * u.deg)

    data = geom.region_mask(regions=[circle], inside=False)
    exclusion_mask = ~geom.region_mask(regions=[circle])
    maker_fov = FoVBackgroundMaker(method="scale", exclusion_mask=exclusion_mask)

    if not args.OBS_ID:
        selection_sc = dict(
            type="sky_circle",
            frame="icrs",
            lon=f"{position_src.ra.degree}",
            lat=f"{position_src.dec.degree}",
            radius="5 deg",
        )

        sc_obs_table = data_store.obs_table.select_observations(selection_sc)

        obs_list = sc_obs_table["OBS_ID"]

        observations = data_store.get_observations(obs_list)
    else:
        observations = data_store.get_observations([args.OBS_ID])

    events_sc = []

    for obs in tqdm(observations[:]):
        if obs.bkg != None:
            try:
                # First a cutout of the target map is produced
                cutout = stacked.cutout(
                    obs.pointing_radec, width=2 * offset_max, name=f"obs-{obs.obs_id}"
                )
                # A MapDataset is filled in this cutout geometry
                dataset = maker.run(cutout, obs)
                # The data quality cut is applied
                dataset = maker_safe_mask.run(dataset, obs)
                dataset = maker_fov.run(dataset)
                # fit background model
                # print(
                # f"\rBackground norm obs {obs.obs_id}: {dataset.background_model.spectral_model.norm.value:2.2f}\r", end = '\r'
                # )
                stacked.stack(dataset)

                events_sc.append(data_store.obs(obs.obs_id).events)

            except Exception:
                print(f"\nbad obs.! obs. {obs.obs_id}\n")
                # bad_obs_list.append(obs.obs_id)

    exposure = stacked.exposure.sum_over_axes()
    exposure.plot(stretch="sqrt", add_cbar=True)
    plt.savefig(dump_path / "exposure.pdf")
    plt.clf()
    excess = stacked.excess.sum_over_axes()
    excess.smooth("0.04 deg").plot(stretch="sqrt", add_cbar=True)
    plt.savefig(dump_path / "excess.pdf")
    print("dumped PDFs")
    return events_sc, observations, stacked


dump_path.mkdir(parents=True, exist_ok=True)

if load == 0:
    print("Making Dataset")
    vars_ = make_dataset_from_config()  # [events_sc, observations, stacked]
    vars_names = ["events_lists", "observations", "stacked_datasets"]

    for var, name in zip(vars_, vars_names):
        fname = Path(dump_path / f"{name}.pkl")
        fname.touch()

        with open(fname, "wb") as file:
            dill.dump(var, file)
        print(f"dumped {name}")

    print("dumped real data, can run ensemble jobs now, force quiting..")

    quit()

else:
    vars_ = []
    vars_names = ["events_lists", "observations", "stacked_datasets"]

    for name in vars_names:
        fname = Path(dump_path / f"{name}.pkl")

        with open(fname, "rb") as file:
            var = dill.load(file)
            vars_.append(var)
        print(f"loaded {name}")

    events_sc, observations, stacked = vars_

############ ANAlYSIS  ####################################

from mc import Sampler
from gamma_transients import core

print("Setting up MC Sampler")

s = Sampler(stacked, events_sc[:])
scanner_sim = core.multiplet_scanner_simulated_table

da_mc = np.array([])
dt_mc = np.array([])

print("Generating MC datsets and scanning")

for N_set in tqdm(range(0, N_sets)):
    mc_events = [s.sample_vectorized(len(event.table)) for event in events_sc]
    table = scanner_sim(
        position_src,
        mc_events,
        dt_threshold,
        r_deg=da_cut,
        r_area=r_area,
        Nmin=multi_N,
        Nmax=None,
    )
    if table is not None:
        dt_mc_, da_mc_ = table["dt"].data, table["da"].data
        dt_mc, da_mc = np.append(dt_mc, dt_mc_), np.append(da_mc, da_mc_)

print(f"MC datsets generated and scanned, {len(da_mc)} events recorded")

if len(da_mc) > 0:
    dump_path_lists.mkdir(parents=True, exist_ok=True)

    fname = Path(dump_path_lists / f"da_dt_{secrets.token_hex(6)}.npz")
    fname.touch()

    with open(fname, "wb") as file:
        np.savez_compressed(file, da=da_mc, dt=dt_mc)
    print(f"dumped results to {fname}")
else:
    print("No events found, terminating...")
