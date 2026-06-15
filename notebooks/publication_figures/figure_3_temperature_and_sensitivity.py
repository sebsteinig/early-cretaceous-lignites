"""Publication Figure 3: high-CO2 temperatures and CO2 sensitivity by model.

A 3-column x 2-row figure comparing the three Aptian model setups (same column
order as the orography overview, Figure SI / 3):

    columns : (a/d) HadCM3 Getech, (b/e) HadCM3 Scotese, (c/f) KCM

    top row    : 1.5 m / 2 m surface air temperature of the HIGHER-CO2 simulation
    bottom row : warming between the high- and low-CO2 simulations of each model
                 (high - low), i.e. an indication of the CO2 sensitivity

Each panel uses the same orthographic projection, regional extent, graticule, and
study-site markers as ``SI_figure_orography_three_models.py``. A faint hint of the
underlying orography is overlaid on every panel as 500 / 1500 / 2000 m contour
lines of that model's own orography.

Run from the repository root with:

    python notebooks/publication_figures/figure_3_temperature_and_sensitivity.py
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

import cmocean
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
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
OUTPUT_FILE = FIG_DIR / "figure_3_temperature_and_sensitivity.pdf"

# --- models ----------------------------------------------------------------
# Same column order as the orography overview. For each column:
# (column title, site-rotation prefix, high-CO2 run, low-CO2 run, panel labels
#  (top, bottom)).
# (column title, rotation, high run, low run, panel labels, climatology suffix).
# KCM uses its native (original) grid climatology rather than the version
# interpolated onto the HadCM3 grid.
MODELS = [
    (
        "HadCM3 (Farnsworth et al., 2019)",
        "getech",
        "teuyo1",
        "teuyO",
        ("a", "d"),
        "clim",
    ),
    ("HadCM3 (Valdes et al., 2021)", "scotese", "texpx2", "texzx1", ("b", "e"), "clim"),
    (
        "KCM (Steinig et al., 2020)",
        "kcm",
        "KCM_1200",
        "KCM_600",
        ("c", "f"),
        "clim.original_grid",
    ),
]

# --- study sites -----------------------------------------------------------
STUDY_SITES = ["TSG", "SVO"]

# --- map framing (identical to the orography overview) ---------------------
CENTRAL_LON = 95.0
CENTRAL_LAT = 47.0
MAP_EXTENT = [50.0, 140.0, 8.0, 85.0]

# --- field rendering -------------------------------------------------------
# "smooth" -> interpolated filled contours (contourf, discrete colour bands);
# "pixels" -> one flat quad per grid cell (pcolormesh).
FIELD_RENDERING = "smooth"

# --- temperature / sensitivity colour scales -------------------------------
# Top row: absolute surface air temperature (sequential, perceptually uniform).
# Discrete 5 degC bands; colorbar ticks every 10 degC.
TEMP_CMAP = cmocean.cm.thermal
TEMP_VMIN = -5.0
TEMP_VMAX = 35.0
TEMP_STEP = 10.0
TEMP_LEVELS = np.arange(TEMP_VMIN, TEMP_VMAX + 0.001, 5.0)
# Bottom row: high - low CO2 warming. Discrete 1 degC bands starting at 3 degC
# (values below 3 use the under-extend colour) so more of the colour range is
# used; colorbar ticks every 2 degC.
DELTA_CMAP = plt.get_cmap("YlOrRd")
DELTA_VMIN = 4.0
DELTA_VMAX = 10.0
DELTA_STEP = 2.0
DELTA_LEVELS = np.arange(DELTA_VMIN, DELTA_VMAX + 0.001, 1.0)

# --- orography hint --------------------------------------------------------
ORO_CONTOUR_LEVELS = [1000.0, 2000.0]
ORO_CONTOUR_COLOR = "#202020"
ORO_CONTOUR_LINEWIDTHS = [0.4, 0.55]  # thicker for higher elevations
ORO_CONTOUR_ALPHA = 0.7
ORO_LABEL_FONTSIZE = 3.5
# "smooth" -> interpolated contour lines with inline labels; "pixel" -> staircase
# outlines following the orography grid cells (same style as the coastline).
ORO_CONTOUR_STYLE = "smooth"
# Draw the relief hint above the dashed graticule.
ORO_CONTOUR_ZORDER = 24
ORO_LABEL_ZORDER = 26

# --- styling ---------------------------------------------------------------
BACKGROUND_COLOR = "white"
# Mask the ocean so only the land surface air temperature is shown: an ocean
# polygon (from each model's land/sea mask) is drawn on top of the field.
MASK_OCEAN = True
OCEAN_MASK_COLOR = "#e8eaed"
COASTLINE_COLOR = "#1a1a1a"
COASTLINE_WIDTH = 0.7
GRATICULE_COLOR = "#555555"
GRATICULE_LABEL_COLOR = "#3a3a3a"
LAT_GRID_LOCS = np.array([0, 20, 40, 60, 80])
LAT_LABEL_LOCS = np.array([20, 40])
LON_LABEL_LOCS = np.array([60, 80, 100, 120])
SITE_COLOR = "#ffffff"
SITE_EDGE = "#101010"
SITE_MARKER = "*"
SITE_MARKER_SIZE = 18
FIGURE_FACECOLOR = "white"

# --- font sizes (tuned for a ~180 mm two-column manuscript figure) ---------
COLUMN_TITLE_FONTSIZE = 7.0
PANEL_LABEL_FONTSIZE = 7.5
GRATICULE_LABEL_FONTSIZE = 6.5
COLORBAR_LABEL_FONTSIZE = 6.0
COLORBAR_TICK_FONTSIZE = 6.5

# --- figure size -----------------------------------------------------------
# Full text-width (two-column) figure: ~180 mm wide. Height kept tight so the
# two rows of (landscape) panels sit close together.
FIGURE_SIZE = (7.1, 4.2)


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


def load_surface_temperature(
    model_id: str, clim_suffix: str = "clim"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Annual-mean surface (1.5 m / 2 m) air temperature in degrees C.

    ``clim_suffix`` selects the climatology file (e.g. ``clim.original_grid`` for
    KCM's native grid). Returns the field plus its longitude/latitude coordinate
    arrays, sorted with ascending latitude.
    """
    ds = xr.open_dataset(
        model_file(model_id, clim_suffix), decode_times=False
    ).squeeze()
    temp_name = find_varname_from_attribute(ds, "units", "K")
    time_name = find_varname_from_attribute(ds, "axis", "T")
    lon_name, lat_name = find_geo_coords(ds)
    ds = ds.sortby(lat_name)

    temp = ds[temp_name]
    if time_name is not None and time_name in temp.dims:
        temp = temp.mean(time_name)
    temp = temp - 273.15
    return temp.values, ds[lon_name].values, ds[lat_name].values


def load_orography(
    model_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the land orography (NaN over ocean) and land mask for ``model_id``."""
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
    land_mask = mask_ds[mask_name].values >= 0.5
    elevation = np.where(land_mask, orog_ds[orog_name].values, np.nan)
    return elevation, land_mask, lon, lat


def cyclic(data: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Append a wrap-around column so there is no seam at lon=0/360."""
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
    """Staircase land/ocean boundary that follows grid-cell edges."""
    lon_edges = centers_to_edges(lon)
    lat_edges = np.clip(centers_to_edges(lat), -90.0, 90.0)

    def segments(x0, y0, x1, y1):
        nan = np.full(x0.shape, np.nan)
        return (
            np.column_stack([x0, x1, nan]).ravel(),
            np.column_stack([y0, y1, nan]).ravel(),
        )

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
    """Mark the study sites with star markers (white fill for contrast)."""
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


def add_ocean_mask_and_coastline(
    ax: plt.Axes, land_mask: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> None:
    """Optionally mask the ocean, then draw the pixel-staircase coastline.

    When ``MASK_OCEAN`` is set, the ocean grid cells are covered with a flat
    colour drawn on top of the field (pixelated, so it lines up with the
    pixel-staircase coastline). The coastline itself always follows the exact
    land/ocean grid-cell outline of the model's mask.
    """
    if MASK_OCEAN:
        ocean = np.where(land_mask, np.nan, 1.0)  # 1 over ocean, NaN over land
        ocean_c, lon_c = cyclic(ocean, lon)
        ax.pcolormesh(
            lon_c,
            lat,
            ocean_c,
            transform=ccrs.PlateCarree(),
            cmap=mcolors.ListedColormap([OCEAN_MASK_COLOR]),
            shading="nearest",
            antialiased=False,  # no faint cell-edge seams over the ocean
            rasterized=True,  # flatten to a uniform grey raster
            zorder=8,
        )

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
    """Draw the latitude circles by hand (no longitude meridians), above the data."""
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


def add_latitude_labels(ax: plt.Axes) -> None:
    """Label LAT_LABEL_LOCS circles just outside the western frame."""
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
        k = idx[np.argmin(x[idx])]
        ax.annotate(
            _format_latitude(lat),
            xy=(x0, y[k]),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=GRATICULE_LABEL_FONTSIZE,
            color=GRATICULE_LABEL_COLOR,
            annotation_clip=False,
            zorder=35,
        )


def add_longitude_labels(ax: plt.Axes) -> None:
    """Label LON_LABEL_LOCS meridians just below the southern frame."""
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
        k = idx[np.argmin(y[idx])]
        ax.annotate(
            _format_longitude(lon),
            xy=(x[k], y0),
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=GRATICULE_LABEL_FONTSIZE,
            color=GRATICULE_LABEL_COLOR,
            annotation_clip=False,
            zorder=35,
        )


def add_orography_contours(
    ax: plt.Axes, elevation: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> None:
    """Overlay the 1000 / 2000 m orography as a faint relief hint.

    ``ORO_CONTOUR_STYLE == "pixel"`` traces the staircase grid-cell outline of
    each elevation threshold (matching the pixel coastline, so the model grid
    points are visible); ``"smooth"`` draws interpolated, inline-labelled
    contour lines.
    """
    halo = [patheffects.withStroke(linewidth=1.0, foreground="white", alpha=0.5)]

    if ORO_CONTOUR_STYLE == "pixel":
        # NaN (ocean) counts as below every threshold.
        elev = np.where(np.isnan(elevation), -1.0, elevation)
        for level, width in zip(ORO_CONTOUR_LEVELS, ORO_CONTOUR_LINEWIDTHS):
            above = elev >= level
            if not above.any():
                continue
            ox, oy = mask_pixel_outline(above, lon, lat)
            ax.plot(
                ox,
                oy,
                transform=ccrs.PlateCarree(),
                color=ORO_CONTOUR_COLOR,
                linewidth=width,
                alpha=ORO_CONTOUR_ALPHA,
                solid_capstyle="round",
                solid_joinstyle="round",
                path_effects=halo,
                zorder=ORO_CONTOUR_ZORDER,
            )
        return

    elev_c, lon_c = cyclic(elevation, lon)
    cs = ax.contour(
        lon_c,
        lat,
        elev_c,
        levels=ORO_CONTOUR_LEVELS,
        colors=ORO_CONTOUR_COLOR,
        linewidths=ORO_CONTOUR_LINEWIDTHS,
        alpha=ORO_CONTOUR_ALPHA,
        transform=ccrs.PlateCarree(),
        zorder=ORO_CONTOUR_ZORDER,
    )
    cs.set_path_effects(halo)
    # Inline elevation labels: break the line and print the value in the gap.
    labels = ax.clabel(
        cs,
        fmt="%d m",
        inline=True,
        inline_spacing=2,
        fontsize=ORO_LABEL_FONTSIZE,
        colors=ORO_CONTOUR_COLOR,
        zorder=ORO_LABEL_ZORDER,
    )
    for text in labels:
        text.set_path_effects(
            [patheffects.withStroke(linewidth=1.6, foreground="white", alpha=0.7)]
        )


def draw_field_panel(
    ax: plt.Axes,
    field: np.ndarray,
    field_lon: np.ndarray,
    field_lat: np.ndarray,
    *,
    cmap,
    norm,
    levels: np.ndarray,
    extend: str,
    orography: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    sites: pd.DataFrame,
    panel_label: str,
    column_title: str | None,
    show_lat_labels: bool,
    show_lon_labels: bool,
):
    """Render one field panel (temperature or warming) and return its mappable."""
    elevation, land_mask, oro_lon, oro_lat = orography

    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor(BACKGROUND_COLOR)

    field_c, lon_c = cyclic(field, field_lon)
    if FIELD_RENDERING == "smooth":
        # Interpolated filled contours with discrete colour bands.
        mappable = ax.contourf(
            lon_c,
            field_lat,
            field_c,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend=extend,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        mappable.set_edgecolor("face")  # avoid faint seams between bands
    else:
        mappable = ax.pcolormesh(
            lon_c,
            field_lat,
            field_c,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading="nearest",
            rasterized=False,
            zorder=5,
        )

    # Mask the ocean (so only land temperatures show) and draw the coastline.
    add_ocean_mask_and_coastline(ax, land_mask, oro_lon, oro_lat)

    # Faint orography relief hint.
    add_orography_contours(ax, elevation, oro_lon, oro_lat)

    add_graticule(ax)
    if show_lat_labels:
        add_latitude_labels(ax)
    if show_lon_labels:
        add_longitude_labels(ax)
    ax.spines["geo"].set_edgecolor("#3a3a3a")
    ax.spines["geo"].set_linewidth(1.0)

    add_study_sites(ax, sites)

    if column_title is not None:
        ax.set_title(
            column_title, fontsize=COLUMN_TITLE_FONTSIZE, fontweight="semibold", pad=6
        )

    ax.text(
        0.03,
        0.97,
        f"({panel_label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        color="#111111",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=1.0
        ),
        zorder=40,
    )
    return mappable


def _style_colorbar(cbar) -> None:
    """Light (but still visible) colorbar outline and tick marks."""
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("#808080")
    cbar.ax.tick_params(
        labelsize=COLORBAR_TICK_FONTSIZE, length=2.5, width=0.6, color="#808080"
    )


def make_figure(output_file: Path) -> None:
    projection = ccrs.Orthographic(
        central_longitude=CENTRAL_LON, central_latitude=CENTRAL_LAT
    )
    temp_norm = mcolors.Normalize(vmin=TEMP_VMIN, vmax=TEMP_VMAX)
    delta_norm = mcolors.Normalize(vmin=DELTA_VMIN, vmax=DELTA_VMAX)

    fig = plt.figure(figsize=FIGURE_SIZE, facecolor=FIGURE_FACECOLOR)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.08, hspace=0.0)
    axes = [
        [fig.add_subplot(gs[r, c], projection=projection) for c in range(3)]
        for r in range(2)
    ]
    cax_top = fig.add_subplot(gs[0, 3])
    cax_bot = fig.add_subplot(gs[1, 3])
    # Shorten each colorbar to ~80% of its row height (centred) so it does not
    # run taller than the map panels.
    for cax in (cax_top, cax_bot):
        pos = cax.get_position()
        cax.set_position(
            [pos.x0, pos.y0 + pos.height * 0.1, pos.width, pos.height * 0.8]
        )

    temp_mappable = None
    delta_mappable = None
    for col, (
        column_title,
        rotation,
        high_id,
        low_id,
        labels,
        clim_suffix,
    ) in enumerate(MODELS):
        orography = load_orography(high_id)
        sites = load_study_sites(rotation)

        temp_high, t_lon, t_lat = load_surface_temperature(high_id, clim_suffix)
        temp_low, _, _ = load_surface_temperature(low_id, clim_suffix)
        delta = temp_high - temp_low

        # Top row: higher-CO2 absolute surface air temperature.
        temp_mappable = draw_field_panel(
            axes[0][col],
            temp_high,
            t_lon,
            t_lat,
            cmap=TEMP_CMAP,
            norm=temp_norm,
            levels=TEMP_LEVELS,
            extend="both",
            orography=orography,
            sites=sites,
            panel_label=labels[0],
            column_title=column_title,
            show_lat_labels=(col == 0),
            show_lon_labels=False,
        )
        # Bottom row: high - low CO2 warming.
        delta_mappable = draw_field_panel(
            axes[1][col],
            delta,
            t_lon,
            t_lat,
            cmap=DELTA_CMAP,
            norm=delta_norm,
            levels=DELTA_LEVELS,
            extend="both",
            orography=orography,
            sites=sites,
            panel_label=labels[1],
            column_title=None,
            show_lat_labels=(col == 0),
            show_lon_labels=False,
        )

    # Per-row vertical colorbars.
    cbar_top = fig.colorbar(temp_mappable, cax=cax_top, extend="both")
    cbar_top.set_label("surface air temperature (°C)", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar_top.set_ticks([0, 10, 20, 30])
    _style_colorbar(cbar_top)

    cbar_bot = fig.colorbar(delta_mappable, cax=cax_bot, extend="both")
    cbar_bot.set_label(
        "surface air temperature change (°C)", fontsize=COLORBAR_LABEL_FONTSIZE
    )
    cbar_bot.set_ticks([4, 6, 8, 10])
    _style_colorbar(cbar_bot)

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
