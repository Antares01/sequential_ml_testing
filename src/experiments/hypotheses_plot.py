import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
from scipy.interpolate import splprep, splev

# ============================================================
# Global font scale -- change this ONE number to resize all text
# ============================================================
FONT_SCALE = 1.5

BASE_NULLBOX_FONT = 26
BASE_SETLABEL_FONT = 25
BASE_POINT_FONT = 23
BASE_ARROW_FONT = 14
BASE_CAPTION_FONT = 13.5
BASE_PANEL_FONT = 16

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 13 * FONT_SCALE,
    }
)

# ============================================================
# Soft, conference-friendly palette
# ============================================================
BLUE_FILL = "#DCE9F7"
BLUE_EDGE = "#4472A8"
RED_FILL = "#FBE6DD"
RED_EDGE = "#C4633D"
GRAY_EDGE = "#4A4A4A"
TEXT_GRAY = "#2B2B2B"
POINT_BLUE = "#2F5C93"
POINT_RED = "#B4502C"
PANEL_TOP_FILL = "#FAFAF8"
PANEL_BOTTOM_FILL = "#F5F7FA"
PANEL_EDGE = "#BBBBBB"
PANEL_LABEL = "#3A3A3A"

# ============================================================
# Layout constants (generously spaced so larger fonts still fit)
# ============================================================
LEFT_X, RIGHT_X = 3.2, 10.3
CANVAS_W, CANVAS_H = 13.5, 10.0

NULLBOX_Y = 8.55
PANEL_SPLIT_Y = 7.55
ESTAR_Y = 6.75
BLOB_L_CENTER = (LEFT_X, 2.85)
BLOB_R_CENTER = (RIGHT_X, 3.5)
CAPTION_Y = 0.35

fig, ax = plt.subplots(figsize=(CANVAS_W * 13.5 / 13.5, CANVAS_H * 9.6 / 9.6))
ax.set_xlim(0, CANVAS_W)
ax.set_ylim(0, CANVAS_H)
ax.axis("off")


# ============================================================
# 0. Background panels distinguishing the two "spaces"
#    (drawn first so everything else sits on top of them)
# ============================================================
def add_panel(x0, y0, width, height, facecolor, edgecolor, label):
    rect = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2,
        linestyle=(0, (5, 4)),
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=0,
    )
    ax.add_patch(rect)
    ax.text(
        x0 + 0.3,
        y0 + height / 2,
        label,
        ha="center",
        va="center",
        rotation=90,
        fontsize=BASE_PANEL_FONT * FONT_SCALE,
        color=PANEL_LABEL,
        style="italic",
        zorder=0,
    )


PANEL_TOP_H = CANVAS_H - PANEL_SPLIT_Y - 0.25
PANEL_BOTTOM_H = PANEL_SPLIT_Y - 0.15

add_panel(
    0.25,
    PANEL_SPLIT_Y,
    CANVAS_W - 0.5,
    PANEL_TOP_H,
    PANEL_TOP_FILL,
    PANEL_EDGE,
    "null space",
)
add_panel(
    0.25,
    0.15,
    CANVAS_W - 0.5,
    PANEL_BOTTOM_H,
    PANEL_BOTTOM_FILL,
    PANEL_EDGE,
    "e-value space",
)


# ============================================================
# Helpers
# ============================================================


def add_null_box(x, y, text):
    box = FancyBboxPatch(
        (x - 1.5, y - 0.5),
        3.0,
        1.0,
        boxstyle="round,pad=0.08,rounding_size=0.16",
        linewidth=1.4,
        edgecolor=GRAY_EDGE,
        facecolor="#F7F7F5",
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=BASE_NULLBOX_FONT * FONT_SCALE,
        color=TEXT_GRAY,
        zorder=3,
    )


def irregular_blob(
    cx, cy, rx, ry, seed, wobble=0.10, rotation=0.0, n_anchor=9
):
    """A closed, hand-drawn-looking blob: random radial wobble + spline smoothing."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n_anchor, endpoint=False)
    radii = 1.0 + rng.uniform(-wobble, wobble, size=n_anchor)
    xs = cx + rx * radii * np.cos(theta + rotation)
    ys = cy + ry * radii * np.sin(theta + rotation)
    xs = np.append(xs, xs[0])
    ys = np.append(ys, ys[0])
    tck, _ = splprep([xs, ys], s=0, per=True)
    u_fine = np.linspace(0, 1, 300)
    x_fine, y_fine = splev(u_fine, tck)
    return x_fine, y_fine


def add_search_set(
    x, y, rx, ry, label, seed, facecolor, edgecolor, rotation=0.0
):
    xf, yf = irregular_blob(x, y, rx, ry, seed=seed, rotation=rotation)
    verts = list(zip(xf, yf))
    path = Path(verts)
    patch = PathPatch(
        path,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.6,
        alpha=0.9,
        joinstyle="round",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y - ry * 1.05,
        label,
        ha="center",
        va="top",
        fontsize=BASE_SETLABEL_FONT * FONT_SCALE,
        color=edgecolor,
        zorder=3,
    )
    return xf, yf


def closest_point_on_curve(xf, yf, target_x, target_y):
    """Nearest point on the boundary curve to (target_x, target_y) -- a projection."""
    d2 = (xf - target_x) ** 2 + (yf - target_y) ** 2
    i = np.argmin(d2)
    return xf[i], yf[i]


def add_point(x, y, label, color, fontsize=None, dx=0.14, dy=0.06, ha="left"):
    if fontsize is None:
        fontsize = BASE_POINT_FONT * FONT_SCALE
    ax.plot(
        x,
        y,
        "o",
        markersize=6,
        color=color,
        zorder=5,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )
    ax.text(
        x + dx,
        y + dy,
        label,
        fontsize=fontsize,
        va="bottom",
        ha=ha,
        color=TEXT_GRAY,
        zorder=5,
    )


def arrow(
    x1,
    y1,
    x2,
    y2,
    text=None,
    text_offset=(0, 0),
    linestyle="-",
    linewidth=1.3,
    mutation_scale=14,
    color="#555555",
    fontsize=None,
    alpha=1.0,
    arrowstyle="->",
):
    if fontsize is None:
        fontsize = BASE_ARROW_FONT * FONT_SCALE
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=arrowstyle,
            linewidth=linewidth,
            linestyle=linestyle,
            mutation_scale=mutation_scale,
            color=color,
            alpha=alpha,
        ),
        zorder=4,
    )
    if text is not None:
        xm = (x1 + x2) / 2 + text_offset[0]
        ym = (y1 + y2) / 2 + text_offset[1]
        ax.text(
            xm,
            ym,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontstyle="italic",
            color=TEXT_GRAY,
            zorder=4,
        )


# ============================================================
# 1. Null hypotheses + information-loss arrow between them
# ============================================================
add_null_box(LEFT_X, NULLBOX_Y, r"$\mathcal{P}^{\mathrm{X-CI}}$")
add_null_box(RIGHT_X, NULLBOX_Y, r"$\mathcal{P}^{\#}$")

arrow(
    LEFT_X + 1.65,
    NULLBOX_Y,
    RIGHT_X - 1.65,
    NULLBOX_Y,
    text="information loss",
    text_offset=(0, 0.32),
    linewidth=1.6,
    mutation_scale=16,
    color=GRAY_EDGE,
)

arrow(
    LEFT_X,
    NULLBOX_Y - 0.55,
    LEFT_X,
    ESTAR_Y + 0.3,
    color="#6C6C6C",
    linewidth=1.8,
    linestyle=(0, (6, 5)),
    mutation_scale=16,
    alpha=0.6,
    arrowstyle="-|>",
)
arrow(
    RIGHT_X,
    NULLBOX_Y - 0.55,
    RIGHT_X,
    ESTAR_Y + 0.3,
    color="#6C6C6C",
    linewidth=1.8,
    linestyle=(0, (6, 5)),
    mutation_scale=16,
    alpha=0.6,
    arrowstyle="-|>",
)

# single E* point per side (used both for the coarsening-error and
# approximation-error comparisons below)
Estar_L = (LEFT_X, ESTAR_Y)
Estar_R = (RIGHT_X, ESTAR_Y)
add_point(
    *Estar_L,
    r"$E^{\mathrm{X-CI}}$",
    POINT_BLUE,
    ha="right",
    dx=-0.2,
    dy=-0.05,
)
add_point(
    *Estar_R,
    r"$E^{\#}$",
    POINT_RED,
    ha="left",
    dx=0.2,
    dy=-0.05,
)

arrow(
    Estar_L[0] + 0.3,
    Estar_L[1],
    Estar_R[0] - 0.3,
    Estar_R[1],
    text="null enlargement error",
    text_offset=(0, 0.34),
    linestyle="--",
    linewidth=1.2,
    color="#888888",
)


# ============================================================
# 2. Search spaces (irregular blobs)
#    Left blob sits a bit farther below E*_X-CI than the right blob
#    sits below E*_Exch, so the *projected* approximation error is
#    visibly larger on the left without needing an artificial target.
#    The orange (Exch) blob is intentionally smaller than the blue one.
# ============================================================
xf_L, yf_L = add_search_set(
    *BLOB_L_CENTER,
    2.15,
    1.55,
    r"$\mathfrak{E}_{\mathrm{X-CI}}$",
    seed=7,
    facecolor=BLUE_FILL,
    edgecolor=BLUE_EDGE,
    rotation=0.3,
)
xf_R, yf_R = add_search_set(
    *BLOB_R_CENTER,
    1.55,
    0.85,
    r"$\mathfrak{E}_{\#}$",
    seed=12,
    facecolor=RED_FILL,
    edgecolor=RED_EDGE,
    rotation=1.1,
)


# ============================================================
# 3. Approximation error: E* projected onto the NEAREST point of
#    the search-space boundary (a true nearest-point projection).
# ============================================================
bx, by = closest_point_on_curve(xf_L, yf_L, Estar_L[0], Estar_L[1])
add_point(
    bx,
    by,
    r"$E^{\mathfrak{E}_{\mathrm{X-CI}}}$",
    POINT_BLUE,
    dx=0.18,
    dy=0.12,
)
arrow(
    Estar_L[0],
    Estar_L[1] - 0.2,
    bx,
    by + 0.10,
    text="approximation\nerror",
    text_offset=(-1.2, -0.05),
    color=POINT_BLUE,
)

ex_R, ey_R = closest_point_on_curve(xf_R, yf_R, Estar_R[0], Estar_R[1])
add_point(
    ex_R,
    ey_R,
    r"$E^{\mathfrak{E}_{\#}}$",
    POINT_RED,
    dx=-0.22,
    dy=0.04,
    ha="right",
)
arrow(
    Estar_R[0],
    Estar_R[1] - 0.2,
    ex_R,
    ey_R + 0.10,
    text="approximation\nerror",
    text_offset=(1.25, 0.05),
    color=POINT_RED,
)


# ============================================================
# 4. Estimation error: search-space optimum -> estimated e-value.
#    This point can land anywhere in the interior (it need not sit
#    below, or even near, the projection point).
# ============================================================
ehat_L = (BLOB_L_CENTER[0] + 0.75, BLOB_L_CENTER[1] - 1.0)
add_point(
    ehat_L[0],
    ehat_L[1],
    r"$\widehat{E}^{\mathrm{X-CI}}$",
    POINT_BLUE,
    dx=-0.18,
    dy=-0.10,
    ha="right",
)
arrow(
    bx - 0.01,
    by - 0.15,
    ehat_L[0],
    ehat_L[1] + 0.15,
    text="estimation\nerror",
    text_offset=(-1.3, -0.15),
    color=POINT_BLUE,
)

ehat_R = (BLOB_R_CENTER[0] - 0.55, BLOB_R_CENTER[1] - 0.65)
add_point(
    ehat_R[0],
    ehat_R[1],
    r"$\widehat{E}^{\#}$",
    POINT_RED,
    dx=0.18,
    dy=-0.10,
)
arrow(
    ex_R + 0.01,
    ey_R - 0.1,
    ehat_R[0],
    ehat_R[1] + 0.15,
    text="estimation\nerror",
    text_offset=(2.3, -1.15),
    color=POINT_RED,
)


plt.tight_layout()
plt.savefig("null_to_evalue_diagram.pdf", bbox_inches="tight")
print("done")
