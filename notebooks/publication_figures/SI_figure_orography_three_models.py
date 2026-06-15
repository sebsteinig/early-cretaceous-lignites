"""SI Figure: orography comparison of the three Aptian model palaeogeographies.

Three orthographic globes side by side, each centred on the central-Asian study
region, showing the land surface orography of the three model setups used in the
manuscript:

    a) KCM (Muller / Blakey palaeogeography)
    b) HadCM3 (Scotese / PALEOMAP palaeogeography)
    c) HadCM3 (Getech palaeogeography)

Land elevation is drawn at the native model resolution (grid-cell / pixel style)
with a land-only terrain colormap, palaeo-coastlines are outlined from each
model's land/sea mask, and the new brGDGT study sites are marked at their
model-appropriate reconstructed palaeo-locations.

The style mirrors ``figure_1_orography_overview.py`` but omits bathymetry and the
ocean-basin labels, and adds (a)/(b)/(c) panel labels. Exported as an editable
(vector) PDF.

Run from the repository root with:

    python notebooks/publication_figures/SI_figure_orography_three_models.py
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
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

REPO_ROOT = Path(__file__).resolve().parents[2]
# The ``src`` package may live at the repo root or under ``notebooks``; add
# whichever directory actually contains it so imports work from either layout.
for _candidate in (REPO_ROOT, REPO_ROOT / "notebooks"):
    if (_candidate / "src").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.fonts import ensure_helvetica_bold
from src.helper import (
    find_geo_coords,
    find_varname_from_attribute,
    find_varname_from_keywords,
)

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
OUTPUT_FILE = FIG_DIR / "SI_figure_orography_three_models.pdf"

# --- models ----------------------------------------------------------------
# (model_id, panel label, panel title, site-rotation column prefix). The
# rotation prefix selects the matching <prefix>_lat / <prefix>_lon columns in the
# reconstructed-locations CSV so the sites land in each model's own palaeo-frame.
MODELS = [
    ("teuyO", "a", "HadCM3 (Farnsworth et al., 2019)", "getech"),
    ("texzx1", "b", "HadCM3 (Valdes et al., 2021)", "scotese"),
    ("KCM_1200", "c", "KCM (Steinig et al., 2020)", "kcm"),
]

# --- study sites -----------------------------------------------------------
# Reconstructed palaeo-locations of the new brGDGT sites.
STUDY_SITES = ["TSG", "SVO"]

# --- map framing -----------------------------------------------------------
# Orthographic view centred near the study sites (~47 N, ~112 E across the three
# rotations) and cropped to a regional window so the sites and the surrounding
# Asian orography fill each panel. MAP_EXTENT is [west, east, south, north] in
# PlateCarree degrees.
CENTRAL_LON = 95.0
CENTRAL_LAT = 47.0
MAP_EXTENT = [50.0, 140.0, 8.0, 85.0]

# --- styling ---------------------------------------------------------------
# Land-only hypsometric ramp (the land portion of figure 1's GEBCO-style cmap):
# green lowland -> tan -> brown -> snow. Ocean is shown as a flat light disk.
ELEVATION_VMIN = 0.0
ELEVATION_VMAX = 2500.0
LAND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "land_terrain",
    [
        (0.00, "#3f7d44"),  # green lowland
        (0.28, "#7cb342"),
        (0.52, "#cbbb66"),  # tan
        (0.76, "#9c6b3c"),  # brown
        (1.00, "#f5f3ee"),  # snow / white
    ],
)
OCEAN_COLOR = "#dceaf6"  # flat light ocean disk (no bathymetry)
COASTLINE_COLOR = "#1a1a1a"
COASTLINE_WIDTH = 1.0
GRATICULE_COLOR = "#555555"
GRATICULE_LABEL_COLOR = "#3a3a3a"
# Latitude circles to draw and the subset that additionally gets a text label.
# Longitude meridians are drawn as well but never labelled.
LAT_GRID_LOCS = np.array([0, 20, 40, 60, 80])
LAT_LABEL_LOCS = np.array([20, 40])
LON_GRID_LOCS = np.arange(0, 181, 20)
# Only the central meridians get labels; edge meridians collapse into the bottom
# corners of the orthographic crop and would overlap.
LON_LABEL_LOCS = np.array([60, 80, 100, 120])
SITE_COLOR = "#d7191c"
SITE_EDGE = "#1a1a1a"
SITE_MARKER = "*"
SITE_MARKER_SIZE = 22
PANEL_LABEL_FONTSIZE = 15
FIGURE_FACECOLOR = "white"


def model_file(model_id: str, suffix: str) -> Path:
    """Return a model file path, allowing for the historical teuyO/teuyo typo."""
    for candidate in (
        MODEL_DIR / f"{model_id}.{suffix}.nc",
        MODEL_DIR / f"{model_id.replace('O', 'o')}.{suffix}.nc",
        MODEL_DIR / f"{model_id.replace('o', 'O')}.{suffix}.nc",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {suffix} file for {model_id} in {MODEL_DIR}")


def load_orography(
    model_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the land orography and land mask for ``model_id``.

    Ocean cells are set to NaN so they drop out of the pixel plot and reveal the
    flat ocean disk beneath. Fields are returned on a longitude/latitude grid
    sorted with ascending latitude.

    Returns
    -------
    elevation : 2-D array (lat, lon) of metres over land, NaN over ocean.
    land_mask : 2-D boolean array, True over land.
    lon, lat : 1-D coordinate arrays.
    """
    orog_ds = xr.open_dataset(
        model_file(model_id, "orog"), decode_times=False
    ).squeeze()
    mask_ds = xr.open_dataset(
        model_file(model_id, "mask"), decode_times=False
    ).squeeze()

    orog_name = find_varname_from_attribute(orog_ds, "units", "m")
    if orog_name is None:
        orog_name = find_varname_from_keywords(
            orog_ds, ["height", "orog", "surface_altitude"]
        )
    mask_name = find_varname_from_keywords(mask_ds, ["land sea mask", "land/sea mask"])
    lon_name, lat_name = find_geo_coords(orog_ds)

    orog_ds = orog_ds.sortby(lat_name)
    mask_ds = mask_ds.sortby(lat_name)

    lon = orog_ds[lon_name].values
    lat = orog_ds[lat_name].values
    orography = orog_ds[orog_name].values
    land_mask = mask_ds[mask_name].values >= 0.5

    elevation = np.where(land_mask, orography, np.nan)
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


def load_study_sites(rotation: str) -> pd.DataFrame:
    """Study-site palaeo-locations in the requested model rotation frame."""
    locations = pd.read_csv(PROXY_LOCATIONS_FILE)
    sites = locations[locations["location"].isin(STUDY_SITES)].copy()
    sites["plat"] = sites[f"{rotation}_lat"]
    sites["plon"] = sites[f"{rotation}_lon"]
    return sites


def add_study_sites(ax: plt.Axes, sites: pd.DataFrame) -> None:
    """Mark the study sites with star markers (no text labels, to keep the
    three small panels uncluttered)."""
    ax.scatter(
        sites["plon"],
        sites["plat"],
        marker=SITE_MARKER,
        s=SITE_MARKER_SIZE**2 / 4,
        c=SITE_COLOR,
        edgecolors=SITE_EDGE,
        linewidths=0.8,
        transform=ccrs.PlateCarree(),
        zorder=30,
    )


def _format_latitude(lat: float) -> str:
    if lat == 0:
        return "0°"
    return f"{abs(int(lat))}°{'N' if lat > 0 else 'S'}"


def _format_longitude(lon: float) -> str:
    lon = ((lon + 180) % 360) - 180  # wrap to [-180, 180)
    if lon in (0, 180):
        return f"{abs(int(lon))}°"
    return f"{abs(int(lon))}°{'E' if lon > 0 else 'W'}"


def add_graticule(ax: plt.Axes) -> None:
    """Draw the latitude circles and longitude meridians by hand, above the data.

    ``ax.gridlines`` does not reliably honour its ``zorder`` here, so the dashed
    graticule ends up hidden under the pixel orography. Plotting the lines
    directly keeps full control over the draw order. Longitude meridians are
    drawn but never labelled.
    """
    line_kw = dict(
        transform=ccrs.PlateCarree(),
        color=GRATICULE_COLOR,
        linewidth=0.5,
        alpha=0.7,
        linestyle=(0, (6, 8)),
        zorder=22,
    )
    lons = np.linspace(-180.0, 180.0, 1441)
    for lat in LAT_GRID_LOCS:
        ax.plot(lons, np.full_like(lons, float(lat)), **line_kw)

    lats = np.linspace(-90.0, 90.0, 721)
    for lon in LON_GRID_LOCS:
        ax.plot(np.full_like(lats, float(lon)), lats, **line_kw)


def add_latitude_labels(ax: plt.Axes) -> None:
    """Label a subset of latitude circles just outside the western frame.

    In the orthographic view each latitude circle is curved and crosses the
    rectangular crop boundary at more than one point, so the built-in gridliner
    stamps duplicate, oddly-placed labels. Here we sample each latitude line,
    find the height at which it reaches the western edge, and anchor a single
    label on the left spine nudged outward onto the white margin.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lons = np.linspace(-180.0, 180.0, 1441)
    for lat in LAT_LABEL_LOCS:
        pts = ax.projection.transform_points(
            ccrs.PlateCarree(), lons, np.full_like(lons, float(lat))
        )
        x, y = pts[:, 0], pts[:, 1]
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if not inside.any():
            continue
        idx = np.flatnonzero(inside)
        k = idx[np.argmin(x[idx])]  # leftmost visible point of this circle
        ax.annotate(
            _format_latitude(lat),
            xy=(x0, y[k]),  # anchor on the left spine at this latitude's height
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.5,
            color=GRATICULE_LABEL_COLOR,
            annotation_clip=False,  # allow the label to sit outside the axes
            zorder=35,
        )


def add_longitude_labels(ax: plt.Axes) -> None:
    """Label a subset of meridians just below the southern frame.

    Mirrors :func:`add_latitude_labels`: each meridian is sampled, clipped to the
    visible axes, and a single label is anchored on the bottom spine at the
    meridian's lowest visible point, nudged downward onto the white margin.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lats = np.linspace(-90.0, 90.0, 721)
    for lon in LON_LABEL_LOCS:
        pts = ax.projection.transform_points(
            ccrs.PlateCarree(), np.full_like(lats, float(lon)), lats
        )
        x, y = pts[:, 0], pts[:, 1]
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if not inside.any():
            continue
        idx = np.flatnonzero(inside)
        k = idx[np.argmin(y[idx])]  # lowest visible point of this meridian
        ax.annotate(
            _format_longitude(lon),
            xy=(x[k], y0),  # anchor on the bottom spine at this meridian's x
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            color=GRATICULE_LABEL_COLOR,
            annotation_clip=False,  # allow the label to sit outside the axes
            zorder=35,
        )


def draw_panel(
    ax: plt.Axes, model_id: str, panel_label: str, title: str, rotation: str
) -> matplotlib.cm.ScalarMappable:
    """Render one model's orography panel and return its mappable."""
    elevation, land_mask, lon, lat = load_orography(model_id)
    sites = load_study_sites(rotation)

    elev_c, lon_c = cyclic(elevation, lon)

    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor(OCEAN_COLOR)  # flat ocean disk (no bathymetry)

    norm = mcolors.Normalize(vmin=ELEVATION_VMIN, vmax=ELEVATION_VMAX)
    mappable = ax.pcolormesh(
        lon_c,
        lat,
        elev_c,
        transform=ccrs.PlateCarree(),
        cmap=LAND_CMAP,
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

    # Latitude circles only (no longitude lines), drawn above the data, with
    # labels just outside the western frame.
    add_graticule(ax)
    add_latitude_labels(ax)
    add_longitude_labels(ax)
    ax.spines["geo"].set_edgecolor("#3a3a3a")
    ax.spines["geo"].set_linewidth(1.0)

    add_study_sites(ax, sites)

    # Panel label and model name combined on a single, left-aligned title line.
    ax.set_title(
        f"({panel_label}) {title}",
        fontsize=12,
        fontweight="semibold",
        pad=8,
        loc="left",
    )
    return mappable


def make_figure(output_file: Path) -> None:
    projection = ccrs.Orthographic(
        central_longitude=CENTRAL_LON, central_latitude=CENTRAL_LAT
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 5.4),
        subplot_kw={"projection": projection},
        facecolor=FIGURE_FACECOLOR,
    )

    mappable = None
    for ax, (model_id, panel_label, title, rotation) in zip(axes, MODELS):
        mappable = draw_panel(ax, model_id, panel_label, title, rotation)

    # Single shared horizontal colorbar beneath the three panels.
    cbar = fig.colorbar(
        mappable,
        ax=axes.tolist(),
        orientation="horizontal",
        fraction=0.045,
        pad=0.06,
        shrink=0.55,
        extend="max",
    )
    cbar.set_label("surface elevation (m)", fontsize=12)
    cbar.set_ticks(np.arange(ELEVATION_VMIN, ELEVATION_VMAX + 1, 500))
    cbar.ax.tick_params(labelsize=10, length=3)
    cbar.outline.set_linewidth(0.8)

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
