# Gamma Transients

Software for search and processing short-timescale transients using gammapy.

## Getting started
- install gammapy 1.0, download tutorial data folder if not done automatically

- fermi data from here: https://github.com/gammapy/gammapy-fermi-lat-data/tree/master/3fhl/galactic-center
### Preparation of virtual environment and package installation
⚠️ We recommend to install gammapy first using instructions from their webite
```bash
python -m venv venv
python -m pip install -e .
```

## Command-line interface
### Simulations
The python script provided in `scripts/mc_scanner_script.py` can be used to perform the simulations as described in the paper. The following is an example command to generate and scan 200 simulated datasets corresponding to the open H.E.S.S. data for mh15-52.

The first step is to download gammapy datasets

```bash
gammapy download notebooks
gammapy download datasets
export GAMMAPY_DATA=$PWD/gammapy-datasets/$(ls gammapy-datasets/)
```

At first step we generate and save the exposure maps and time data required for subsequent simulations
```bash
mkdir -p out/
python scripts/mc_scanner_script.py -n mh15-52 -i  ${GAMMAPY_DATA}/hess-dl3-dr1 -o out/ -l 0 -s 10 --RA 228.3198 --DEC -59.081 --da-cut 0.2 --multi_N 3 --r-area-cut 0.5 --dt-max 800000000
```

Now we perform 10 simulations for the extracted exposure:

```bash
python scripts/mc_scanner_script.py -n mh15-52 -i ${GAMMAPY_DATA}/hess-dl3-dr1 -o out/ -l 1 -s 10 --RA 228.3198 --DEC -59.081 --da-cut 0.2 --multi_N 3 --r-area-cut 0.5 --dt-max 800000000
```


## Python notebooks

Example Jupyter notebooks are provided to demonstrate the working of the tool and reproduce the scans of the Fermi-LAT and open H.E.S.S. data from the paper.
