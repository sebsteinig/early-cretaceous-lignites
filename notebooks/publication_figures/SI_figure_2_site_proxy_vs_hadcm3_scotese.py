"""Supplementary Figure 2: site proxy vs HadCM3 Scotese model comparison.

Run from the repository root with:

    python notebooks/publication_figures/SI_figure_2_site_proxy_vs_hadcm3_scotese.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Keep text editable (TrueType, not Type 3) in the exported PDF.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# Helvetica house style for all publication figures (Arial / DejaVu Sans are
# metric-compatible fallbacks for machines without Helvetica).
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
# Put the repo root on the path for ``notebooks.publication_figures.*`` imports,
# and the ``notebooks`` directory for the ``src`` package (which now lives under
# ``notebooks/src``).
for _candidate in (REPO_ROOT, REPO_ROOT / "notebooks"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from notebooks.publication_figures.figure_2_meridional_temp_gradients import (
    FIGURE_STYLE,
    FIG_DIR,
    MODEL_PAIRS,
    PROXY_LOCATIONS_FILE,
    additional_proxy_legend_label,
    load_additional_proxy_data,
    model_file,
)
from src.fonts import ensure_helvetica_bold
from src.helper import find_geo_coords, find_varname_from_attribute

# macOS only registers regular Helvetica; pull in a bold face so semibold/bold
# weights render (instead of silently falling back to regular).
ensure_helvetica_bold()


OUTPUT_FILE = FIG_DIR / "SI_figure_2_site_proxy_vs_hadcm3_scotese.pdf"
LOW_CO2_MODEL, HIGH_CO2_MODEL = MODEL_PAIRS["HadCM3+Scotese"]
MODEL_COLOR = FIGURE_STYLE["model_colors"]["HadCM3+Scotese"]
PROXY_COLOR = "black"
GRID_COLOR = "#D9D9D9"
MAP_EXTENT = [85, 130, 32, 56]
MAP_CENTER_LON = 108
MAP_CENTER_LAT = 44
OROGRAPHY_LEVELS = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
MONGOLIA_SITE_ORDER = ["TSG", "SVO", "SSO"]


def load_temperature_dataset(model_id: str) -> tuple[xr.Dataset, str, str, str, str]:
    ds_clim = xr.open_dataset(
        model_file(model_id, "clim"), decode_times=False
    ).squeeze()
    temp_name = find_varname_from_attribute(ds_clim, "units", "K")
    time_name = find_varname_from_attribute(ds_clim, "axis", "T")
    lon_name, lat_name = find_geo_coords(ds_clim)
    ds_clim[temp_name] -= 273.15
    return ds_clim, temp_name, time_name, lat_name, lon_name


def load_scotese_orography() -> tuple[xr.Dataset, xr.Dataset, str, str, str, str]:
    ds_orog = xr.open_dataset(
        model_file(LOW_CO2_MODEL, "orog"), decode_times=False
    ).squeeze()
    ds_mask = xr.open_dataset(
        model_file(LOW_CO2_MODEL, "mask"), decode_times=False
    ).squeeze()
    orog_name = find_varname_from_attribute(ds_orog, "unit", "m")
    mask_name = find_varname_from_attribute(ds_mask, "units", "1")
    lon_name, lat_name = find_geo_coords(ds_orog)
    ds_orog = ds_orog.where(ds_mask[mask_name] >= 0.5, np.nan)
    return ds_orog, ds_mask, orog_name, mask_name, lon_name, lat_name


def load_site_reconstructions() -> pd.DataFrame:
    mongolia_data = pd.read_csv(PROXY_LOCATIONS_FILE)
    mongolia_sites = (
        mongolia_data[mongolia_data["location"].isin(MONGOLIA_SITE_ORDER)]
        .set_index("location")
        .loc[MONGOLIA_SITE_ORDER]
        .reset_index()
    )
    mongolia_records = []
    for _, row in mongolia_sites.iterrows():
        mongolia_records.append(
            {
                "site": row["location"],
                "label": f"{row['location']} (this study)",
                "material": "brGDGT",
                "modern_lat": row["modern_lat"],
                "modern_lon": row["modern_lon"],
                "scotese_lat": row["scotese_lat"],
                "scotese_lon": row["scotese_lon"],
                "maat_mean": row["maat_mean"],
                "maat_min": row["maat_min"],
                "maat_max": row["maat_max"],
            }
        )

    additional_data = load_additional_proxy_data().copy()
    additional_data["label"] = additional_data.apply(
        additional_proxy_legend_label, axis=1
    )

    return pd.concat(
        [
            pd.DataFrame(mongolia_records),
            additional_data[
                [
                    "site",
                    "label",
                    "material",
                    "modern_lat",
                    "modern_lon",
                    "scotese_lat",
                    "scotese_lon",
                    "maat_mean",
                    "maat_min",
                    "maat_max",
                ]
            ],
        ],
        ignore_index=True,
    )


def sample_site_temperature(
    ds: xr.Dataset,
    temp_name: str,
    time_name: str,
    lat_name: str,
    lon_name: str,
    latitude: float,
    longitude: float,
) -> float:
    return float(
        ds[temp_name]
        .sel({lat_name: latitude, lon_name: longitude}, method="nearest")
        .mean(time_name)
        .values
    )


def add_model_temperatures(site_data: pd.DataFrame) -> pd.DataFrame:
    low_ds, temp_name, time_name, lat_name, lon_name = load_temperature_dataset(
        LOW_CO2_MODEL
    )
    high_ds, _, _, _, _ = load_temperature_dataset(HIGH_CO2_MODEL)
    site_data = site_data.copy()
    site_data["model_low"] = [
        sample_site_temperature(
            low_ds,
            temp_name,
            time_name,
            lat_name,
            lon_name,
            row.scotese_lat,
            row.scotese_lon,
        )
        for row in site_data.itertuples()
    ]
    site_data["model_high"] = [
        sample_site_temperature(
            high_ds,
            temp_name,
            time_name,
            lat_name,
            lon_name,
            row.scotese_lat,
            row.scotese_lon,
        )
        for row in site_data.itertuples()
    ]
    site_data["model_mean"] = (site_data["model_low"] + site_data["model_high"]) / 2
    return site_data


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.1)
    ax.tick_params(axis="both", colors="black")


def add_paleogeography_inset(fig: plt.Figure, site_data: pd.DataFrame) -> None:
    inset_ax = fig.add_axes(
        [0.76, 0.74, 0.22, 0.22],
        projection=ccrs.Orthographic(
            central_longitude=MAP_CENTER_LON,
            central_latitude=MAP_CENTER_LAT,
        ),
    )
    ds_orog, ds_mask, orog_name, mask_name, lon_name, lat_name = (
        load_scotese_orography()
    )
    inset_ax.set_facecolor("#D7E8F7")
    inset_ax.pcolormesh(
        ds_mask[lon_name],
        ds_mask[lat_name],
        ds_mask[mask_name],
        cmap=matplotlib.colors.ListedColormap(["#D7E8F7", "#EFEFE8"]),
        vmin=0,
        vmax=1,
        transform=ccrs.PlateCarree(),
        zorder=0,
    )
    inset_ax.contour(
        ds_mask[lon_name],
        ds_mask[lat_name],
        ds_mask[mask_name],
        levels=[0.5],
        colors="#4A4A4A",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    inset_ax.contourf(
        ds_orog[lon_name],
        ds_orog[lat_name],
        ds_orog[orog_name],
        levels=OROGRAPHY_LEVELS,
        cmap="Greys",
        alpha=0.75,
        extend="max",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    inset_ax.scatter(
        site_data["scotese_lon"],
        site_data["scotese_lat"],
        s=35,
        color=PROXY_COLOR,
        edgecolor="white",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    for site_number, row in enumerate(site_data.itertuples(), start=1):
        inset_ax.text(
            row.scotese_lon + 0.8,
            row.scotese_lat + 0.4,
            str(site_number),
            fontsize=7,
            fontweight="bold",
            color="black",
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
    inset_ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    gridlines = inset_ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="#7A7A7A",
        alpha=0.6,
        linestyle="--",
        x_inline=False,
        y_inline=False,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 7, "color": "black"}
    gridlines.ylabel_style = {"size": 7, "color": "black"}
    inset_ax.set_title("HadCM3 Scotese paleogeography", fontsize=9, pad=4)


def plot_site_comparison(site_data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(15, 8), facecolor="white")
    style_axis(ax)

    x = np.arange(len(site_data))
    proxy_x = x - 0.08
    model_x = x + 0.08

    proxy_yerr = np.vstack(
        [
            site_data["maat_mean"] - site_data["maat_min"],
            site_data["maat_max"] - site_data["maat_mean"],
        ]
    )
    ax.errorbar(
        proxy_x,
        site_data["maat_mean"],
        yerr=proxy_yerr,
        fmt="o",
        color=PROXY_COLOR,
        markerfacecolor=PROXY_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=12,
        capsize=5,
        linestyle="none",
        label="Proxy reconstruction",
        zorder=10,
    )

    model_lower = site_data[["model_low", "model_high"]].min(axis=1)
    model_upper = site_data[["model_low", "model_high"]].max(axis=1)
    ax.vlines(
        model_x,
        model_lower,
        model_upper,
        color=MODEL_COLOR,
        linewidth=5,
        label="HadCM3 (Scotese), low-high CO2",
        zorder=8,
    )
    ax.scatter(
        model_x,
        site_data["model_mean"],
        color=MODEL_COLOR,
        edgecolor="white",
        linewidth=0.8,
        s=95,
        zorder=9,
    )

    ax.set_xticks(x)
    site_labels = [
        f"{site_number}. {label}"
        for site_number, label in enumerate(site_data["label"], start=1)
    ]
    ax.set_xticklabels(site_labels, rotation=35, ha="right")
    ax.set_ylabel("Surface temperature (°C)", fontsize=13)
    ax.set_title(
        "model-data comparison for HadCM3 (Scotese)",
        fontsize=18,
        fontweight="semibold",
        pad=16,
    )

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=11,
        title="Temperature estimates",
        title_fontsize=12,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(GRID_COLOR)
    add_paleogeography_inset(fig, site_data)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.34, top=0.9)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_FILE}")


def main() -> None:
    site_data = add_model_temperatures(load_site_reconstructions())
    plot_site_comparison(site_data)


if __name__ == "__main__":
    main()
