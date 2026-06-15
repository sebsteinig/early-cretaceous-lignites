"""Publication Figure 1: orographic overview map.

An orthographic globe centred on the central-Asian study region showing the
HadCM3 model surface orography on the Scotese (PALEOMAP) Aptian palaeogeography.
Land elevation is drawn at the native model resolution (grid-cell style) with a
terrain colormap, palaeo-coastlines are outlined from the land/sea mask, and the
new brGDGT study sites are marked at their reconstructed palaeo-locations.

The figure is intended as the opening / scene-setting figure of the manuscript
and is exported as a fully editable (vector) PDF.

Run from the repository root with:

    python notebooks/publication_figures/figure_1_orography_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Keep text and vector geometry editable in the exported PDF.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# Helvetica house style for all publication figures (Arial / DejaVu Sans are
# metric-compatible fallbacks for machines without Helvetica).
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

import cmocean  # noqa: F401  (registers colormaps / kept for env parity)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

REPO_ROOT = Path(__file__).resolve().parents[2]
# The ``src`` package may live at the repo root or under ``notebooks``; add
# whichever directory actually contains it so imports work from either layout.
for _candidate in (REPO_ROOT, REPO_ROOT / "notebooks"):
    if (_candidate / "src").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.fonts import ensure_helvetica_bold
from src.helper import find_geo_coords, find_varname_from_keywords

# macOS only registers regular Helvetica; pull in a bold face so semibold/bold
# weights render (instead of silently falling back to regular).
ensure_helvetica_bold()


# --- paths -----------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data" / "v2"
MODEL_DIR = DATA_DIR / "raw" / "model_clims"
PROXY_LOCATIONS_FILE = (
    DATA_DIR / "processed" / "proxy_temps_and_reconstructed_locations.csv"
)
FIG_DIR = REPO_ROOT / "figures" / "publication_v1"
OUTPUT_FILE = FIG_DIR / "figure_1_orography_overview.pdf"

# HadCM3 + Scotese palaeogeography (texzx1/texpx2 share identical orography).
OROG_MODEL_ID = "texzx1"

# --- study sites -----------------------------------------------------------
# Reconstructed (Scotese / PALEOMAP) palaeo-locations of the new brGDGT sites.
STUDY_SITES = ["TSG", "SVO"]
SITE_ROTATION = "scotese"  # use scotese_lat / scotese_lon columns
# Long site names shown in the plot, keyed by site code.
SITE_LONG_NAMES = {
    "TSG": "Tevshiin Govi (TSG)",
    "SVO": "Shivee Ovoo (SVO)",
}
# Per-site label placement (short offset in points) and leader-line direction.
SITE_LABEL_OFFSETS = {
    "TSG": (-30, -18),
    "SVO": (26, 18),
}

# --- map framing -----------------------------------------------------------
CENTRAL_LON = 90.0
CENTRAL_LAT = 12.0

# --- styling ---------------------------------------------------------------
# Combined topography (land, >0 m) + bathymetry (ocean depth, <0 m) range.
ELEVATION_VMIN = -5200.0  # deepest ocean
ELEVATION_VMAX = 2500.0  # highest land
SEA_LEVEL = 0.0
# GEBCO-style hypsometric ramp: light blue bathymetry below sea level, then a
# hard break at the coast to green -> tan -> brown -> snow above. The duplicated
# stop at 0.5 keeps the land/ocean transition crisp, and a TwoSlopeNorm pins sea
# level to that midpoint.
TOPO_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "gebco_light",
    [
        (0.00, "#0a3d62"),  # deep ocean
        (0.20, "#2e86c1"),
        (0.38, "#7fb3d5"),
        (0.50, "#dcecf7"),  # shallow ocean (just below sea level)
        (0.50, "#3f7d44"),  # green lowland (just above sea level)
        (0.64, "#7cb342"),
        (0.76, "#cbbb66"),  # tan
        (0.88, "#9c6b3c"),  # brown
        (1.00, "#f5f3ee"),  # snow / white
    ],
)
OCEAN_COLOR = "#08233f"  # globe-disk fallback (deep ocean)
COASTLINE_COLOR = "#1a1a1a"
COASTLINE_WIDTH = 1.1
GRATICULE_COLOR = "lightgrey"
GRATICULE_LABEL_COLOR = "#3a3a3a"
# Graticule spacing and the latitude circles that get text labels.
LON_GRID_LOCS = np.arange(-180, 181, 30)
LAT_GRID_LOCS = np.arange(-90, 91, 30)
LAT_LABEL_LOCS = np.arange(-60, 61, 30)
SITE_COLOR = "#d7191c"
SITE_EDGE = "#1a1a1a"
SITE_MARKER = "*"
SITE_MARKER_SIZE = 26
FIGURE_FACECOLOR = "white"

# Ocean basin labels (text, palaeo-lon/lat, rotation in degrees). White text
# with a dark outline so they read over the dark cmocean ocean colours.
OCEAN_LABEL_COLOR = "white"
OCEAN_LABEL_STROKE = "#0a1a2b"
OCEAN_LABELS = [
    {"text": "Neotethys", "lon": 56.0, "lat": 10.0, "rotation": -40.0},
    {"text": "Panthalassa\nOcean", "lon": 146.0, "lat": 22.0, "rotation": 12.0},
]

ADD_TITLE = False
FIGURE_TITLE = "HadCM3 Aptian (~116 Ma) orography on the Scotese palaeogeography"


def model_file(model_id: str, suffix: str) -> Path:
    candidate = MODEL_DIR / f"{model_id}.{suffix}.nc"
    if not candidate.exists():
        raise FileNotFoundError(f"No {suffix} file for {model_id} in {MODEL_DIR}")
    return candidate


def load_topography_bathymetry() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Return the combined topography/bathymetry field and the land mask.

    Land elevation comes from the HadCM3 orography (``ht``); ocean depth comes
    from the ocean-model file (``depthdepth``, positive metres) and is stored as
    a negative elevation. The ocean-model land/sea mask (``lsm``: 1=land,
    0=ocean) is the single authority for the coastline so it stays consistent
    with the bathymetry. All fields are returned on a common longitude/latitude
    grid sorted with ascending latitude.

    Returns
    -------
    elevation : 2-D array (lat, lon) of metres, positive over land, negative
        (= -depth) over ocean.
    land_mask : 2-D boolean array, True over land.
    lon, lat : 1-D coordinate arrays.
    """
    orog_ds = (
        xr.open_dataset(model_file(OROG_MODEL_ID, "orog"), decode_times=False)
        .squeeze()
        .sortby("latitude")
    )
    omask_ds = (
        xr.open_dataset(
            MODEL_DIR / f"{OROG_MODEL_ID}.qrparm.omask.nc", decode_times=False
        )
        .squeeze()
        .sortby("latitude")
    )

    orog_name = find_varname_from_keywords(
        orog_ds, ["height", "orog", "surface_altitude"]
    )
    if orog_name is None:
        orog_name = "ht"
    lon_name, lat_name = find_geo_coords(orog_ds)

    lon = orog_ds[lon_name].values
    lat = orog_ds[lat_name].values
    orography = orog_ds[orog_name].values
    depth = omask_ds["depthdepth"].values  # positive metres, NaN over land
    land_mask = omask_ds["lsm"].values > 0.5

    elevation = np.where(land_mask, orography, -depth)
    return elevation, land_mask, lon, lat


def cyclic(data: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Append a wrap-around column so the globe has no seam at lon=0/360."""
    data_c, lon_c = add_cyclic_point(data, coord=lon)
    return data_c, lon_c


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges (length n+1) from monotonic cell centres (length n)."""
    centers = np.asarray(centers, dtype=float)
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (midpoints[0] - centers[0])
    last = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def mask_pixel_outline(
    land_mask: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the staircase land/ocean boundary that follows grid-cell edges.

    Returns concatenated x/y coordinate arrays (NaN-separated segments) tracing
    every grid-cell edge where a land cell meets an ocean cell, including the
    cyclic seam. Unlike a contour at 0.5 this does not interpolate: it follows
    the true pixel outline of the mask.
    """
    lon_edges = centers_to_edges(lon)
    lat_edges = np.clip(centers_to_edges(lat), -90.0, 90.0)

    def segments(x0, y0, x1, y1):
        """Pack equal-length endpoint arrays into NaN-separated polyline coords."""
        nan = np.full(x0.shape, np.nan)
        return (
            np.column_stack([x0, x1, nan]).ravel(),
            np.column_stack([y0, y1, nan]).ravel(),
        )

    # Vertical edges between east/west neighbours (interior + cyclic seam).
    interior = land_mask[:, :-1] != land_mask[:, 1:]
    wrap = (land_mask[:, -1] != land_mask[:, 0])[:, None]
    vdiff = np.concatenate([interior, wrap], axis=1)
    i_idx, j_idx = np.nonzero(vdiff)
    vx, vy = segments(
        lon_edges[j_idx + 1],
        lat_edges[i_idx],
        lon_edges[j_idx + 1],
        lat_edges[i_idx + 1],
    )

    # Horizontal edges between north/south neighbours.
    hdiff = land_mask[:-1, :] != land_mask[1:, :]
    i_idx, j_idx = np.nonzero(hdiff)
    hx, hy = segments(
        lon_edges[j_idx],
        lat_edges[i_idx + 1],
        lon_edges[j_idx + 1],
        lat_edges[i_idx + 1],
    )

    return np.concatenate([vx, hx]), np.concatenate([vy, hy])


def load_study_sites() -> pd.DataFrame:
    locations = pd.read_csv(PROXY_LOCATIONS_FILE)
    sites = locations[locations["location"].isin(STUDY_SITES)].copy()
    sites["plat"] = sites[f"{SITE_ROTATION}_lat"]
    sites["plon"] = sites[f"{SITE_ROTATION}_lon"]
    return sites


def add_study_sites(ax: plt.Axes, sites: pd.DataFrame) -> None:
    geo_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
    for _, site in sites.iterrows():
        ax.scatter(
            site["plon"],
            site["plat"],
            marker=SITE_MARKER,
            s=SITE_MARKER_SIZE**2 / 4,
            c=SITE_COLOR,
            edgecolors=SITE_EDGE,
            linewidths=0.9,
            transform=ccrs.PlateCarree(),
            zorder=30,
        )

        offset = SITE_LABEL_OFFSETS.get(site["location"], (40, 28))
        label_text = SITE_LONG_NAMES.get(site["location"], site["location"])
        ax.annotate(
            label_text,
            xy=(site["plon"], site["plat"]),
            xycoords=geo_transform,
            xytext=offset,
            textcoords="offset points",
            fontsize=10.5,
            fontweight="semibold",
            color="#111111",
            ha="left" if offset[0] >= 0 else "right",
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                color=SITE_EDGE,
                linewidth=0.9,
                connectionstyle="arc3,rad=0.12",
            ),
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#bcc4cc",
                linewidth=0.8,
                alpha=0.9,
            ),
            zorder=31,
        )


def add_ocean_labels(ax: plt.Axes) -> None:
    """Place ocean-basin name labels at palaeo-locations."""
    for label in OCEAN_LABELS:
        ax.text(
            label["lon"],
            label["lat"],
            label["text"],
            transform=ccrs.PlateCarree(),
            rotation=label["rotation"],
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=12,
            color=OCEAN_LABEL_COLOR,
            zorder=20,
            path_effects=[
                patheffects.withStroke(linewidth=2.2, foreground=OCEAN_LABEL_STROKE)
            ],
        )


def add_graticule(ax: plt.Axes) -> None:
    """Draw a 30° lat/lon graticule in front of the data.

    Drawn as plain PlateCarree polylines rather than via ``ax.gridlines`` because
    cartopy's Gridliner does not reliably honour ``zorder`` — in particular it
    ends up behind the rasterised high-res surface. ``ax.plot`` respects zorder,
    so the lines stay on top in both the vector and rasterised figures. Cartopy
    clips them to the visible hemisphere automatically.
    """
    lons = np.linspace(-180, 180, 361)
    lats = np.linspace(-90, 90, 181)
    line_kw = dict(
        transform=ccrs.PlateCarree(),
        color=GRATICULE_COLOR,
        alpha=0.4,
        linewidth=0.4,
        linestyle=(0, (12, 8)),  # longer dashes, fewer per line
        zorder=15,
    )
    for lat in LAT_GRID_LOCS:
        ax.plot(lons, np.full_like(lons, lat), **line_kw)
    # for lon in LON_GRID_LOCS:
    #     ax.plot(np.full_like(lats, lon), lats, **line_kw)


def _format_latitude(lat: float) -> str:
    if lat == 0:
        return "0°"
    return f"{abs(int(lat))}°{'N' if lat > 0 else 'S'}"


def add_latitude_labels(ax: plt.Axes) -> None:
    """Label each 30° latitude circle just outside the globe's left limb.

    For an orthographic view centred at ``(CENTRAL_LON, CENTRAL_LAT)`` a latitude
    circle meets the visible limb where the angular distance to the centre is
    90°, i.e. cos(Δlon) = -tan(lat0)·tan(lat). We place the label at that western
    crossing and nudge the text outward onto the white margin so it reads clearly
    regardless of the ocean colour beneath the limb.
    """
    lat0 = np.deg2rad(CENTRAL_LAT)
    geo = ccrs.PlateCarree()._as_mpl_transform(ax)
    for lat in LAT_LABEL_LOCS:
        cos_dlon = -np.tan(lat0) * np.tan(np.deg2rad(lat))
        if abs(cos_dlon) > 1:  # circle fully visible/hidden, no limb crossing
            continue
        dlon = np.rad2deg(np.arccos(cos_dlon))
        lon = CENTRAL_LON - dlon * 0.999  # western (left) limb crossing
        ax.annotate(
            _format_latitude(lat),
            xy=(lon, lat),
            xycoords=geo,
            xytext=(-9, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9.5,
            color=GRATICULE_LABEL_COLOR,
            zorder=25,
        )


def make_figure(output_file: Path) -> None:
    elevation, land_mask, lon, lat = load_topography_bathymetry()
    sites = load_study_sites()

    elev_c, lon_c = cyclic(elevation, lon)

    projection = ccrs.Orthographic(
        central_longitude=CENTRAL_LON, central_latitude=CENTRAL_LAT
    )

    fig = plt.figure(figsize=(8.5, 8.5), facecolor=FIGURE_FACECOLOR)
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_global()
    ax.set_facecolor(OCEAN_COLOR)  # fallback deep-ocean disk colour

    norm = mcolors.TwoSlopeNorm(
        vmin=ELEVATION_VMIN, vcenter=SEA_LEVEL, vmax=ELEVATION_VMAX
    )
    mappable = ax.pcolormesh(
        lon_c,
        lat,
        elev_c,
        transform=ccrs.PlateCarree(),
        cmap=TOPO_CMAP,
        norm=norm,
        shading="nearest",  # one flat-coloured quad per model grid point
        rasterized=False,  # keep cells as vector geometry in the PDF
        zorder=5,
    )

    # Palaeo-coastline following the exact pixel outline of the land/ocean mask.
    coast_x, coast_y = mask_pixel_outline(land_mask, lon, lat)
    ax.plot(
        coast_x,
        coast_y,
        transform=ccrs.PlateCarree(),
        color=COASTLINE_COLOR,
        linewidth=COASTLINE_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=10,
    )

    # Graticule (30° spacing) drawn in front of the data, with latitude labels.
    ax.spines["geo"].set_edgecolor("#3a3a3a")
    ax.spines["geo"].set_linewidth(1.1)

    add_graticule(ax)
    add_latitude_labels(ax)
    add_ocean_labels(ax)
    add_study_sites(ax, sites)

    # Vertical colorbar.
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="vertical",
        fraction=0.046,
        pad=0.03,
        shrink=0.78,
        extend="max",
    )
    cbar.set_label("surface elevation (m)", fontsize=12)
    cbar.set_ticks(
        np.concatenate(
            [
                np.arange(ELEVATION_VMIN + 200, 0, 1000),
                np.arange(0, ELEVATION_VMAX + 1, 1000),
            ]
        )
    )
    cbar.ax.tick_params(labelsize=10, length=3)
    cbar.outline.set_linewidth(0.8)

    if ADD_TITLE:
        ax.set_title(FIGURE_TITLE, fontsize=12.5, fontweight="semibold", pad=14)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_file,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        dpi=300,
    )
    plt.close(fig)
    print(f"Wrote {output_file}")


def main() -> None:
    make_figure(OUTPUT_FILE)


if __name__ == "__main__":
    main()
