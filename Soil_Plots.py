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
    "SPT value (N)": {
        "Medium": [0,5,13,13,14,15,14,17,20,22,29],
        "Soft":   [0,7,9,8,12,8,9,32,31,29,12],
        "Hard":   [0,12,24,33,50,50,50,50,50,50,50]
    },
    "Shearwave velocity - Vs (m/s)": {
        "Medium": [0,173,220,220,224,228,224,235,245,251,269],
        "Soft":   [0,188,201,195,216,195,201,276,274,269,216],
        "Hard":   [0,216,257,278,309,309,309,309,309,309,309]
    },
    "Shear modulus - G (kN/m²)": {
        "Medium": [0,55,89,89,93,96,93,102,111,116,133],
        "Soft":   [0,65,74,70,86,70,74,140,138,133,86],
        "Hard":   [0,79,112,153,188,188,188,188,183,183,183]
    },
    "Elastic modulus - E (kN/m²)": {
        "Medium": [0,149,241,241,250,259,250,275,299,314,360],
        "Soft":   [0,176,200,189,231,189,200,378,372,360,231],
        "Hard":   [0,213,302,413,508,508,508,508,495,495,495]
    },
    "Bulk Modulus - B (kN/m²)": {
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
# SET FONT TO TIMES NEW ROMAN
# =========================
plt.rcParams.update({'font.family': 'Times New Roman'})

# =========================
# FIGURE
# =========================
fig, axes = plt.subplots(
    1, 5,
    figsize=(14, 5),
    sharey=True,
    gridspec_kw={"wspace": 0.03}  # almost joined
)

# =========================
# PLOT
# =========================
for ax, (xlabel, pdata) in zip(axes, params.items()):
    for soil in ["Soft", "Medium", "Hard"]:
        # Skip the first zero row
        ax.plot(
            pdata[soil][1:], depth[1:],
            linewidth=1.2,
            markersize=5,
            linestyle="-",
            **styles[soil],
            label=soil
        )
    
    # Axis labels bold
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    
    # Subtle grids BOTH directions
    ax.minorticks_on()
    ax.grid(which='major', axis='y', linestyle='--', linewidth=0.3, alpha=0.3)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.2, alpha=0.2)
    ax.grid(which='major', axis='x', linestyle='--', linewidth=0.3, alpha=0.3)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.2, alpha=0.2)
    
    # Remove internal spines for compact look
    for spine in ['top','bottom','left','right']:
        ax.spines[spine].set_visible(False)

# =========================
# DEPTH AXIS
# =========================
axes[0].set_ylabel("Depth (m)", fontsize=10, fontweight='bold')
axes[0].set_ylim(15, 0)  # increase downward

# Ticks
for ax in axes:
    ax.tick_params(labelsize=9)

# =========================
# LEGEND (TOP) WITH BOLD TEXT
# =========================
leg = fig.legend(
    ["Soft", "Medium", "Hard"],
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=10
)
for text in leg.get_texts():
    text.set_fontweight('bold')

# =========================
# OUTERMOST BOUNDING BOX (all plots)
# =========================
for i, ax in enumerate(axes):
    # left spine only for first plot
    if i == 0:
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(0.8)
    # right spine only for last plot
    if i == len(axes)-1:
        ax.spines['right'].set_visible(True)
        ax.spines['right'].set_linewidth(0.8)
    # top and bottom for all
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_linewidth(0.8)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(0.8)

fig.tight_layout(rect=[0, 0, 1, 0.92])

# =========================
# SAVE
# =========================
plt.savefig(
    "Soil_Profile_Composite_JournalReady_NoZero.png",
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
# DATA
# =========================
params = {
    "Frictional Angle (°)": {
        "Soft":   [0, 0, 0],
        "Medium": [0, 0, 0],
        "Hard":   [24, 32, 32]
    },
    "Cohesion (kPa)": {
        "Soft":   [42, 42, 42],
        "Medium": [38, 38, 41],
        "Hard":   [17, 17, 17]
    },
    "Shear modulus - G (kPa)": {
        "Soft":   [65352, 69883, 133396],
        "Medium": [55195, 95812, 133396],
        "Hard":   [78896, 152819, 188264]
    },
    "Elastic modulus - E (kPa)": {
        "Soft":   [176451, 188684, 360171],
        "Medium": [149028, 258692, 360171],
        "Hard":   [213018, 412611, 508312]
    },
    "Bulk modulus - B (kPa)": {
        "Soft":   [196056, 209649, 400189],
        "Medium": [165586, 287435, 400189],
        "Hard":   [236687, 458457, 564791]
    },
    "Mass density (Mg/m³)": {
        "Soft":   [1.83, 1.83, 1.83],
        "Medium": [1.79, 1.81, 1.81],
        "Hard":   [1.69, 1.97, 1.92]
    }
}

# =========================
# STYLES + OFFSET
# =========================
styles = {
    "Soft":   dict(color="green", marker="^", offset=0.0),
    "Medium": dict(color="blue",  marker="x", offset=0.5),
    "Hard":   dict(color="red",   marker="s", offset=-0.5)
}

# =========================
# STEP PROFILE FUNCTION
# =========================
def step_profile(values, offset=0):
    x, y = [], []
    for (d1, d2), v in zip(layers, values):
        x.extend([v + offset, v + offset])
        y.extend([d1, d2])
    return x, y

# =========================
# SET FONT TO TIMES NEW ROMAN
# =========================
plt.rcParams.update({'font.family': 'Times New Roman'})

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
        x, y = step_profile(pdata[soil], offset=styles[soil]["offset"])
        ax.plot(
            x, y,
            linewidth=1.4,
            markersize=6,
            linestyle="-",
            color=styles[soil]["color"],
            marker=styles[soil]["marker"],
            label=soil
        )
    
    # Axis labels bold
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    
    # =========================
    # Minor + Major grid BOTH HORIZONTAL & VERTICAL
    # =========================
    ax.minorticks_on()
    # Major grid
    ax.grid(which='major', axis='y', linestyle='--', linewidth=0.3, alpha=0.3)
    ax.grid(which='major', axis='x', linestyle='--', linewidth=0.3, alpha=0.3)
    # Minor grid
    ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.2, alpha=0.2)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.2, alpha=0.2)
    
    # Remove internal spines to merge plots visually
    for spine in ['top','bottom','left','right']:
        ax.spines[spine].set_visible(False)

# =========================
# DEPTH AXIS
# =========================
axes[0].set_ylabel("Depth (m)", fontsize=10, fontweight='bold')
axes[0].set_ylim(11.5, 0)

for ax in axes:
    ax.tick_params(labelsize=9)

# =========================
# LEGEND (TOP) WITH BOLD TEXT
# =========================
leg = fig.legend(
    ["Soft", "Medium", "Hard"],
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=10
)
for text in leg.get_texts():
    text.set_fontweight('bold')

# =========================
# OUTERMOST BOUNDING BOX
# =========================
for i, ax in enumerate(axes):
    # left spine only for first plot
    if i == 0:
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(0.8)
    # right spine only for last plot
    if i == len(axes)-1:
        ax.spines['right'].set_visible(True)
        ax.spines['right'].set_linewidth(0.8)
    # top and bottom for all
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_linewidth(0.8)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(0.8)

fig.tight_layout(rect=[0, 0, 1, 0.92])

# =========================
# SAVE FIGURE
# =========================
plt.savefig(
    "Layerwise_Soil_Properties_VerticalGrid.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()



