###########Plots for the field test data


import matplotlib.pyplot as plt
import numpy as np

# =========================
# DEPTH (positive downward)
# =========================
depth = np.array([0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15])

# =========================
# DATA
# =========================
params = {
    "N (–)": {
        "Medium": [0,5,13,13,14,15,14,17,20,22,29],
        "Soft":   [0,7,9,8,12,8,9,32,31,29,12],
        "Hard":   [0,12,24,33,50,50,50,50,50,50,50]
    },
    "Vs (m/s)": {
        "Medium": [0,173,220,220,224,228,224,235,245,251,269],
        "Soft":   [0,188,201,195,216,195,201,276,274,269,216],
        "Hard":   [0,216,257,278,309,309,309,309,309,309,309]
    },
    "G (kN/m²)": {
        "Medium": [0,55,89,89,93,96,93,102,111,116,133],
        "Soft":   [0,65,74,70,86,70,74,140,138,133,86],
        "Hard":   [0,79,112,153,188,188,188,188,183,183,183]
    },
    "E (kN/m²)": {
        "Medium": [0,149,241,241,250,259,250,275,299,314,360],
        "Soft":   [0,176,200,189,231,189,200,378,372,360,231],
        "Hard":   [0,213,302,413,508,508,508,508,495,495,495]
    },
    "B (kN/m²)": {
        "Medium": [0,166,268,268,278,287,278,306,332,348,400],
        "Soft":   [0,196,222,210,257,210,222,420,414,400,257],
        "Hard":   [0,237,335,458,565,565,565,565,550,550,550]
    }
}

# =========================
# STYLES
# =========================
styles = {
    "Soft":   dict(color="green", marker="^"),
    "Medium": dict(color="blue",  marker="x"),
    "Hard":   dict(color="red",   marker="s")
}

# =========================
# FIGURE
# =========================
fig, axes = plt.subplots(
    1, 5,
    figsize=(13.5, 5),
    sharey=True,
    gridspec_kw={"wspace": 0.02}  # almost joined
)

# =========================
# PLOT
# =========================
for ax, (xlabel, pdata) in zip(axes, params.items()):
    for soil in ["Soft", "Medium", "Hard"]:
        ax.plot(
            pdata[soil], depth,
            linewidth=1.2,
            markersize=5,
            linestyle="-",
            **styles[soil],
            label=soil
        )
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4)

    # remove subplot borders
    for spine in ax.spines.values():
        spine.set_visible(False)

# =========================
# DEPTH AXIS
# =========================
axes[0].set_ylabel("Depth (m)", fontsize=9)
axes[0].set_ylim(15, 0)  # start from 0, increase downward

# ticks
for ax in axes:
    ax.tick_params(labelsize=8)

# =========================
# LEGEND (TOP)
# =========================
fig.legend(
    ["Soft", "Medium", "Hard"],
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=9
)

fig.tight_layout(rect=[0, 0, 1, 0.92])

# =========================
# SAVE
# =========================
plt.savefig(
    "Soil_Profile_Composite_Elsevier.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()



###########Plots for the Adopted data for the NUmerical modellings

import matplotlib.pyplot as plt

# =========================
# DEPTH INTERVALS
# =========================
layers = [(0, 1.5), (1.5, 6.5), (6.5, 11.5)]

# =========================
# DATA (Layer-wise constants)
# =========================
params = {
    "φ (°)": {
        "Soft":   [0, 0, 0],
        "Medium": [0, 0, 0],
        "Hard":   [24, 32, 32]
    },
    "C (kPa)": {
        "Soft":   [42, 42, 42],
        "Medium": [38, 38, 41],
        "Hard":   [17, 17, 17]
    },
    "G (kPa)": {
        "Soft":   [65352, 69883, 133396],
        "Medium": [55195, 95812, 133396],
        "Hard":   [78896, 152819, 188264]
    },
    "E (kPa)": {
        "Soft":   [176451, 188684, 360171],
        "Medium": [149028, 258692, 360171],
        "Hard":   [213018, 412611, 508312]
    },
    "B (kPa)": {
        "Soft":   [196056, 209649, 400189],
        "Medium": [165586, 287435, 400189],
        "Hard":   [236687, 458457, 564791]
    },
    "ρ (Mg/m³)": {
        "Soft":   [1.83, 1.83, 1.83],
        "Medium": [1.79, 1.81, 1.81],
        "Hard":   [1.69, 1.97, 1.92]
    }
}

# =========================
# STYLE
# =========================
styles = {
    "Soft":   dict(color="green", marker="^"),
    "Medium": dict(color="blue",  marker="x"),
    "Hard":   dict(color="red",   marker="s")
}

# =========================
# STEP PROFILE FUNCTION
# =========================
def step_profile(values):
    x, y = [], []
    for (d1, d2), v in zip(layers, values):
        x.extend([v, v])
        y.extend([d1, d2])
    return x, y

# =========================
# FIGURE
# =========================
fig, axes = plt.subplots(
    1, len(params),
    figsize=(15, 5),
    sharey=True,
    gridspec_kw={"wspace": 0.03}
)

# =========================
# PLOTTING
# =========================
for ax, (xlabel, pdata) in zip(axes, params.items()):
    for soil in ["Soft", "Medium", "Hard"]:
        x, y = step_profile(pdata[soil])
        ax.plot(
            x, y,
            linewidth=1.4,
            markersize=5,
            linestyle="-",
            **styles[soil],
            label=soil
        )
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4)

    # remove subplot borders
    for spine in ax.spines.values():
        spine.set_visible(False)

# =========================
# DEPTH AXIS
# =========================
axes[0].set_ylabel("Depth (m)", fontsize=9)
axes[0].set_ylim(11.5, 0)

for ax in axes:
    ax.tick_params(labelsize=8)

# =========================
# LEGEND
# =========================
fig.legend(
    ["Soft (Bishalnagar)", "Medium (Ravibhawan)", "Hard (Surkhet)"],
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=9
)

fig.tight_layout(rect=[0, 0, 1, 0.92])

# =========================
# SAVE
# =========================
plt.savefig(
    "Layerwise_Soil_Properties_Elsevier.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


