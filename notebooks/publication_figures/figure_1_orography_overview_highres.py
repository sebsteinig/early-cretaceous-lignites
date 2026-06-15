"""Publication Figure 1 (high-resolution variant).

The same orthographic overview map as ``figure_1_orography_overview.py`` but
using the high-resolution Scotese & Wright (2018) PaleoMAP PaleoDEM for the
Early Cretaceous (115 Ma) instead of the coarse HadCM3 model grid.

The PaleoDEM ``z`` field is a single combined topography/bathymetry surface
(metres, sea level at 0) on a 0.1 degree grid, so no orography/ocean-depth merge
is needed and the coastline is the pixel outline of the land mask (elevation >=
0). The elevation surface is rasterised in the PDF (6.5 million cells); all
annotations stay vector.

View, colormap, study sites, and ocean labels are reused from the model figure
so the two panels match.

Run from the repository root with:

    python notebooks/publication_figures/figure_1_orography_overview_highres.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# Helvetica house style for all publication figures (Arial / DejaVu Sans are
# metric-compatible fallbacks for machines without Helvetica).
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

# Reuse the styling, framing, sites, and ocean labels from the model figure so
# the high-res panel is visually consistent with it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_1_orography_overview as base


REPO_ROOT = Path(__file__).resolve().parents[2]
DEM_FILE = (
    REPO_ROOT
    / "data"
    / "Scotese_Wright_2018_Maps_1-88_6minX6min_PaleoDEMS_nc"
    / "Map26_PALEOMAP_6min_Early_Cretaceous_115Ma.nc"
)
OUTPUT_FILE = base.FIG_DIR / "figure_1_orography_overview_highres.pdf"
FIGURE_TITLE = "Scotese & Wright (2018) PaleoDEM — Early Cretaceous (~115 Ma)"
# The high-res shoreline is finely detailed, so keep it thin (independent of the
# thicker low-res model coastline).
COASTLINE_WIDTH = 0.5


def load_dem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the high-res elevation field (metres) and its lon/lat axes."""
    ds = xr.open_dataset(DEM_FILE, decode_times=False)
    elevation = ds["z"].values
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    return elevation, lon, lat


def make_figure(output_file: Path) -> None:
    elevation, lon, lat = load_dem()
    sites = base.load_study_sites()

    projection = ccrs.Orthographic(
        central_longitude=base.CENTRAL_LON, central_latitude=base.CENTRAL_LAT
    )

    fig = plt.figure(figsize=(8.5, 8.5), facecolor=base.FIGURE_FACECOLOR)
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_global()
    ax.set_facecolor(base.OCEAN_COLOR)

    norm = mcolors.TwoSlopeNorm(
        vmin=base.ELEVATION_VMIN, vcenter=base.SEA_LEVEL, vmax=base.ELEVATION_VMAX
    )

    # Rasterised elevation surface. pcolormesh projects vertices directly (no
    # scipy/pykdtree image-warp dependency); rasterized=True keeps the field
    # a single embedded image in the PDF rather than millions of vectors.
    mappable = ax.pcolormesh(
        lon,
        lat,
        elevation,
        transform=ccrs.PlateCarree(),
        cmap=base.TOPO_CMAP,
        norm=norm,
        shading="nearest",
        rasterized=True,
        zorder=5,
    )
    # Coastline as the pixel outline of the land mask (elevation >= sea level):
    # it follows the exact grid-cell edges the pcolormesh fill uses, so it sits
    # precisely on the green/blue colour transition.
    land_mask = elevation >= base.SEA_LEVEL
    coast_x, coast_y = base.mask_pixel_outline(land_mask, lon, lat)
    ax.plot(
        coast_x,
        coast_y,
        transform=ccrs.PlateCarree(),
        color=base.COASTLINE_COLOR,
        linewidth=COASTLINE_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=10,
    )

    ax.spines["geo"].set_edgecolor("#3a3a3a")
    ax.spines["geo"].set_linewidth(1.1)

    base.add_graticule(ax)
    base.add_latitude_labels(ax)
    base.add_ocean_labels(ax)
    base.add_study_sites(ax, sites)

    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="vertical",
        fraction=0.046,
        pad=0.03,
        shrink=0.78,
        extend="both",
    )
    cbar.set_label("surface elevation (m)", fontsize=12)
    cbar.set_ticks(
        np.concatenate(
            [
                np.arange(base.ELEVATION_VMIN + 200, 0, 1000),
                np.arange(0, base.ELEVATION_VMAX + 1, 1000),
            ]
        )
    )
    cbar.ax.tick_params(labelsize=10, length=3)
    cbar.outline.set_linewidth(0.8)

    if base.ADD_TITLE:
        ax.set_title(FIGURE_TITLE, fontsize=12.5, fontweight="semibold", pad=14)

    base.FIG_DIR.mkdir(parents=True, exist_ok=True)
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
