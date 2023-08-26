from matplotlib import figure, gridspec
import numpy as np
import dill
import scanner_interpretation as scani

with open("testing/pkl_jugs/mplest.pkl", "rb") as f:
    mplets: scani.Multiplets = dill.load(f)

dt = mplets.table["dt"].data
Nmax = mplets.table["Nmax"].data
da = mplets.table["da"].data

fig = figure.Figure()
gs = gridspec.GridSpec(2, 2, figure=fig)

ax00 = fig.add_subplot(gs[0, 0])  # bottom left, 2dhist
ax10 = fig.add_subplot(gs[1, 0], sharex=ax00)  # top left, dt hist
ax01 = fig.add_subplot(
    gs[0, 1],
)  # bottom right
ax11 = fig.add_subplot(gs[1, 1])  # top right, Nmax hist
