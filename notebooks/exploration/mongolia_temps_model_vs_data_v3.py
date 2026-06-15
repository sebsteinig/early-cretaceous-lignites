"""Aptian Mongolia model-data comparison.

Run from the repository root with:

    python notebooks/exploration/mongolia_temps_model_vs_data_v3.py

This is a plain Python script, not a Jupytext/notebook companion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import add_proxy_location_markers
from src.helper import (
    find_geo_coords,
    find_varname_from_attribute,
    find_varname_from_keywords,
)
from src.plotting import plot_contours, plot_filled_map, split_cmap
from src.reconstruction import get_scotese_paleolocation


DATA_DIR = REPO_ROOT / "data" / "v2"
MODEL_DIR = DATA_DIR / "raw" / "model_clims"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = REPO_ROOT / "figures" / "v3"

SAVE_FIGURES = True
RECONSTRUCTION_AGE_MA = 116

MODELS = [
    ("KCM_600", "KCM 600"),
    ("KCM_1200", "KCM 1200"),
    ("texzx1", "HadCM3+Scotese 560"),
    ("texpx2", "HadCM3+Scotese 1103"),
    ("tfksx", "HadCM3+Scotese+phys 780"),
    ("tfkex", "HadCM3+Scotese+phys 1103"),
    ("teuyO", "HadCM3+Getech 560"),
    ("teuyo1", "HadCM3+Getech 1120"),
]
MODEL_IDS = [model_id for model_id, _ in MODELS]
MODEL_LABELS = [label for _, label in MODELS]

MODEL_PAIRS = {
    "KCM": ("KCM_600", "KCM_1200"),
    "HadCM3+Getech": ("teuyO", "teuyo1"),
    "HadCM3+Scotese": ("texzx1", "texpx2"),
    "HadCM3+Scotese+phys": ("tfksx", "tfkex"),
}
PAIR_COLORS = {
    "KCM": "tab:red",
    "HadCM3+Getech": "tab:blue",
    "HadCM3+Scotese": "tab:green",
    "HadCM3+Scotese+phys": "tab:purple",
}
MODEL_COLORS = ["red", "red", "green", "green", "purple", "purple", "blue", "blue"]

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
PLOT_SITES = ["TSG", "SVO"]


def model_file(model_id: str, suffix: str) -> Path:
    """Return a model file path, allowing for the historical teuyO/teuyo typo."""
    candidates = [
        MODEL_DIR / f"{model_id}.{suffix}.nc",
        MODEL_DIR / f"{model_id.replace('O', 'o')}.{suffix}.nc",
        MODEL_DIR / f"{model_id.replace('o', 'O')}.{suffix}.nc",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {suffix} file found for {model_id} in {MODEL_DIR}")


def model_rotation(model_id: str) -> str:
    if model_id in {"KCM_600", "KCM_1200"}:
        return "kcm"
    if model_id in {"texzx1", "texpx2", "tfksx", "tfkex"}:
        return "scotese"
    return "getech"


def save_figure(fig: plt.Figure, filename: str) -> None:
    if SAVE_FIGURES:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def getech_recon(lat: float, lon: float) -> tuple[float, float]:
    return lat + 2.0, lon + 14.0


def kcm_recon(lat: float, lon: float) -> tuple[float, float]:
    return lat - 1.0, lon + 14.0


def reconstruct_proxy_locations() -> pd.DataFrame:
    data_csv = pd.read_csv(DATA_DIR / "raw" / "proxy_temps_and_locations.csv")

    data_csv["scotese_lat"], data_csv["scotese_lon"] = zip(
        *data_csv.apply(
            lambda row: get_scotese_paleolocation(
                row["modern_lat"], row["modern_lon"], RECONSTRUCTION_AGE_MA
            ),
            axis=1,
        )
    )
    data_csv["kcm_lat"], data_csv["kcm_lon"] = zip(
        *data_csv.apply(
            lambda row: kcm_recon(row["scotese_lat"], row["scotese_lon"]), axis=1
        )
    )
    data_csv["getech_lat"], data_csv["getech_lon"] = zip(
        *data_csv.apply(
            lambda row: getech_recon(row["scotese_lat"], row["scotese_lon"]), axis=1
        )
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data_csv.to_csv(
        PROCESSED_DIR / "proxy_temps_and_reconstructed_locations.csv", index=False
    )
    return data_csv


def load_orography(model_id: str) -> tuple[xr.Dataset, xr.Dataset, str, str]:
    ds_orog = xr.open_dataset(model_file(model_id, "orog")).squeeze()
    ds_mask = xr.open_dataset(model_file(model_id, "mask")).squeeze()
    orog_name = find_varname_from_attribute(ds_orog, "unit", "m")
    mask_name = find_varname_from_keywords(ds_mask, ["land sea mask", "land/sea mask"])
    ds_orog = ds_orog.where(ds_mask[mask_name] >= 0.5, np.nan)
    return ds_orog, ds_mask, orog_name, mask_name


def load_annual_temperature(model_id: str) -> tuple[xr.Dataset, xr.Dataset, str, str]:
    ds_clim = xr.open_dataset(model_file(model_id, "clim"), decode_times=False).squeeze()
    ds_mask = xr.open_dataset(model_file(model_id, "mask")).squeeze()

    temp_name = find_varname_from_attribute(ds_clim, "units", "K")
    time_name = find_varname_from_attribute(ds_clim, "axis", "T")
    mask_name = find_varname_from_keywords(ds_mask, ["land sea mask", "land/sea mask"])

    ds_temp = ds_clim.mean(time_name) - 273.15
    return ds_temp, ds_mask, temp_name, mask_name


def load_annual_precipitation(model_id: str) -> tuple[xr.Dataset, xr.Dataset, str, str]:
    ds_clim = xr.open_dataset(model_file(model_id, "clim"), decode_times=False).squeeze()
    ds_mask = xr.open_dataset(model_file(model_id, "mask")).squeeze()

    pr_name = find_varname_from_keywords(ds_clim, ["precipitation", "PRECIPITATION"])
    time_name = find_varname_from_attribute(ds_clim, "axis", "T")
    mask_name = find_varname_from_keywords(ds_mask, ["land sea mask", "land/sea mask"])

    ds_pr = ds_clim.mean(time_name) * 86400.0
    return ds_pr, ds_mask, pr_name, mask_name


def plot_geographies(data_csv: pd.DataFrame) -> None:
    exp_ids = ["KCM_1200", "texzx1", "teuyO"]
    exp_labels = ["KCM (Muller/Blakey)", "HadCM3 (Scotese)", "HadCM3 (Getech)"]
    orog_levels = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    datasets = [load_orography(exp_id) for exp_id in exp_ids]
    cmap_terrain = split_cmap(plt.cm.terrain, 0.3, 1.0)

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 5), subplot_kw={"projection": ccrs.Robinson()}
    )
    fig.suptitle("Aptian Model Paleogeographies", fontsize=16, fontweight="bold", y=0.7)
    for idx, (ds_orog, ds_mask, orog_name, mask_name) in enumerate(datasets):
        p = plot_filled_map(
            axes[idx],
            ds_orog[orog_name],
            type="pcolormesh",
            cmap=cmap_terrain,
            levels=orog_levels,
            right_labels=True,
            title=exp_labels[idx],
        )
        plot_contours(axes[idx], ds_mask[mask_name], levels=[0.8], linewidths=[1])
        add_proxy_location_markers(axes[idx], exp_ids[idx], data_csv, size=8)
    cbar = fig.colorbar(
        p, ax=axes.ravel().tolist(), orientation="horizontal", pad=0.08, shrink=0.6
    )
    cbar.set_label("Model elevation (m)", fontsize=14)
    save_figure(fig, "global_aptian_geographies.pdf")

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    fig.suptitle(
        "Aptian Model Paleogeographies", fontsize=16, fontweight="bold", y=0.82
    )
    for idx, (ds_orog, ds_mask, orog_name, mask_name) in enumerate(datasets):
        p = plot_filled_map(
            axes[idx],
            ds_orog[orog_name],
            type="pcolormesh",
            cmap=cmap_terrain,
            levels=orog_levels,
            extent=[50, 160, 0, 85],
            title=exp_labels[idx],
        )
        plot_contours(axes[idx], ds_mask[mask_name], levels=[0.8], linewidths=[3])
        add_proxy_location_markers(axes[idx], exp_ids[idx], data_csv, size=10)
    cbar = fig.colorbar(
        p, ax=axes.ravel().tolist(), orientation="horizontal", pad=0.08, shrink=0.6
    )
    cbar.set_label("Model elevation (m)", fontsize=14)
    save_figure(fig, "regional_aptian_geographies.pdf")


def plot_regional_maps(
    datasets: list[tuple[xr.Dataset, xr.Dataset, str, str]],
    data_csv: pd.DataFrame,
    *,
    title: str,
    filename: str,
    cmap,
    levels: np.ndarray,
    colorbar_label: str,
) -> None:
    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)

    for idx, (ds_field, ds_mask, field_name, mask_name) in enumerate(datasets):
        row, column = idx % 2, idx // 2
        p = plot_filled_map(
            axes[row, column],
            ds_field[field_name],
            type="contourf",
            cmap=cmap,
            levels=levels,
            extent=[50, 160, 0, 85],
            title=MODEL_LABELS[idx],
        )
        plot_contours(axes[row, column], ds_mask[mask_name], levels=[0.8], linewidths=[3])
        add_proxy_location_markers(axes[row, column], MODEL_IDS[idx], data_csv, size=10)

    cbar = fig.colorbar(
        p, ax=axes.ravel().tolist(), orientation="horizontal", pad=0.08, shrink=0.6
    )
    cbar.set_label(colorbar_label, fontsize=14)
    save_figure(fig, filename)


def plot_anomaly_maps(
    datasets: list[tuple[xr.Dataset, xr.Dataset, str, str]],
    data_csv: pd.DataFrame,
    *,
    title: str,
    filename: str,
    normal_cmap,
    normal_levels: np.ndarray,
    anomaly_levels: np.ndarray,
    anomaly_cmap: str,
    colorbar_label: str,
    skip_model_ids: set[str] | None = None,
) -> None:
    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)
    skip_model_ids = skip_model_ids or set()

    plot_idx = 0
    for idx, (ds_field, ds_mask, field_name, mask_name) in enumerate(datasets):
        if MODEL_IDS[idx] in skip_model_ids:
            continue

        if MODEL_IDS[idx] in {"KCM_600", "tfksx", "teuyO"}:
            ref = datasets[2]
            field_to_plot = ds_field[field_name] - ref[0][ref[2]]
            levels = anomaly_levels
            colormap = anomaly_cmap
            panel_title = f"{MODEL_LABELS[idx]} minus ref"
        elif MODEL_IDS[idx] in {"KCM_1200", "tfkex", "teuyo1"}:
            ref = datasets[3]
            field_to_plot = ds_field[field_name] - ref[0][ref[2]]
            levels = anomaly_levels
            colormap = anomaly_cmap
            panel_title = f"{MODEL_LABELS[idx]} minus ref"
        else:
            field_to_plot = ds_field[field_name]
            levels = normal_levels
            colormap = normal_cmap
            panel_title = "ref"

        row, column = plot_idx % 2, plot_idx // 2
        p = plot_filled_map(
            axes[row, column],
            field_to_plot,
            type="contourf",
            cmap=colormap,
            levels=levels,
            extent=[50, 160, 0, 85],
            title=panel_title,
        )
        plot_contours(axes[row, column], ds_mask[mask_name], levels=[0.8], linewidths=[3])
        add_proxy_location_markers(axes[row, column], MODEL_IDS[idx], data_csv, size=10)
        plot_idx += 1

    for empty_idx in range(plot_idx, 8):
        axes[empty_idx % 2, empty_idx // 2].set_axis_off()

    cbar = fig.colorbar(
        p, ax=axes.ravel().tolist(), orientation="horizontal", pad=0.08, shrink=0.6
    )
    cbar.set_label(colorbar_label, fontsize=14)
    save_figure(fig, filename)


def extract_location_data(data_csv: pd.DataFrame) -> pd.DataFrame:
    results = []

    for model_idx, (model_id, label) in enumerate(MODELS):
        ds_clim = xr.open_dataset(model_file(model_id, "clim"), decode_times=False).squeeze()
        ds_orog = xr.open_dataset(model_file(model_id, "orog"), decode_times=False).squeeze()

        temp_name = find_varname_from_attribute(ds_clim, "units", "K")
        pr_name = find_varname_from_keywords(ds_clim, ["precipitation", "PRECIPITATION"])
        orog_name = find_varname_from_attribute(ds_orog, "unit", "m")
        lon_name, lat_name = find_geo_coords(ds_clim)
        lon_name_orog, lat_name_orog = find_geo_coords(ds_orog)

        ds_clim[temp_name] -= 273.15
        ds_clim[pr_name] *= 86400.0

        rotation = model_rotation(model_id)
        plat = data_csv[f"{rotation}_lat"]
        plon = data_csv[f"{rotation}_lon"]

        for site_idx, row in data_csv.iterrows():
            site_info = {
                "site": row["location"],
                "model": MODEL_IDS[model_idx],
                "label": label,
                "plat": plat[site_idx],
                "plon": plon[site_idx],
                "pheight": ds_orog[orog_name]
                .sel(
                    {lat_name_orog: plat[site_idx], lon_name_orog: plon[site_idx]},
                    method="nearest",
                )
                .values.round(0),
            }

            for var_name, var_label in [
                (temp_name, "temperature"),
                (pr_name, "precipitation"),
            ]:
                values = (
                    ds_clim[var_name]
                    .sel({lat_name: plat[site_idx], lon_name: plon[site_idx]}, method="nearest")
                    .values
                )
                values_zonal = (
                    ds_clim[var_name].sel({lat_name: plat[site_idx]}, method="nearest").values
                )
                results.append(
                    {
                        **site_info,
                        "variable": var_label,
                        "unit": "degC" if var_label == "temperature" else "mm/day",
                        **{month: value for month, value in zip(MONTHS, values)},
                        "annual_mean_location": np.mean(values),
                        "annual_mean_zonal_mean": np.mean(values_zonal),
                    }
                )

    output_df = pd.DataFrame(results).round(2)
    output_df.to_csv(
        PROCESSED_DIR / "simulated_temperature_precipitation_data_at_locations.csv",
        index=False,
    )
    return output_df


def plot_mean_and_shading(ax, site_data: pd.DataFrame, label: str, color: str) -> None:
    monthly_mean = site_data.mean(axis=1)
    monthly_min = site_data.min(axis=1)
    monthly_max = site_data.max(axis=1)
    annual_mean = site_data.mean().mean()
    months = monthly_mean.index.str[:3].tolist()

    ax.plot(months, monthly_mean.values, label=label, color=color, linewidth=2)
    ax.fill_between(months, monthly_min, monthly_max, color=color, alpha=0.2)
    ax.axhline(y=annual_mean, color=color, linestyle="--", linewidth=2)
    ax.text(
        1.02,
        annual_mean,
        f"{annual_mean:.1f}",
        va="center",
        ha="left",
        color=color,
        transform=ax.get_yaxis_transform(),
        fontsize=14,
    )


def plot_proxy_mean_and_shading(ax, proxy_row: pd.Series, months: list[str]) -> None:
    ax.plot(
        months,
        [proxy_row["maat_mean"]] * 12,
        label="Weijers et al. (2007)",
        color="black",
        linewidth=2,
    )
    ax.fill_between(
        months,
        [proxy_row["maat_min"]] * 12,
        [proxy_row["maat_max"]] * 12,
        color="black",
        alpha=0.2,
    )


def plot_annual_cycle(output_df: pd.DataFrame, data_csv: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey="col")
    variables = ["temperature", "precipitation"]

    for i, site in enumerate(PLOT_SITES):
        for j, variable in enumerate(variables):
            ax = axs[i][j]
            ax.set_title(f"{site} {variable.capitalize()}", fontsize=16)

            for pair_name, (model1, model2) in MODEL_PAIRS.items():
                site_data = output_df[
                    (output_df["site"] == site)
                    & (output_df["variable"] == variable)
                    & (output_df["model"].isin([model1, model2]))
                ]
                site_data = site_data.set_index("model").loc[:, "Jan":"Dec"].T
                plot_mean_and_shading(ax, site_data, pair_name, PAIR_COLORS[pair_name])

            if variable == "temperature":
                proxy_row = data_csv.loc[data_csv["location"] == site].iloc[0]
                plot_proxy_mean_and_shading(ax, proxy_row, MONTHS)

    for i, ax in enumerate(axs.flat):
        ax.set_xlabel("Month")
        ax.set_ylabel("Temperature ($^\\circ$C)" if i % 2 == 0 else "Precipitation (mm/day)")

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODEL_PAIRS) + 1, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_figure(fig, "T-P_annual_cycle.pdf")


def plot_zonal_mean_and_shading(ax, data1, data2, lat_name: str, label: str, color: str) -> None:
    mean = (data1 + data2) / 2
    minimum = xr.apply_ufunc(np.minimum, data1, data2)
    maximum = xr.apply_ufunc(np.maximum, data1, data2)

    ax.plot(data1[lat_name], mean, label=label, color=color, linewidth=2)
    ax.fill_between(data1[lat_name], minimum, maximum, color=color, alpha=0.2)


def plot_meridional_temperature_gradient(data_csv: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    for pair_name, (low_model, high_model) in MODEL_PAIRS.items():
        ds_clim_1 = xr.open_dataset(model_file(low_model, "clim"), decode_times=False).squeeze()
        ds_clim_2 = xr.open_dataset(model_file(high_model, "clim"), decode_times=False).squeeze()

        temp_name = find_varname_from_attribute(ds_clim_1, "units", "K")
        time_name = find_varname_from_attribute(ds_clim_1, "axis", "T")
        lon_name, lat_name = find_geo_coords(ds_clim_1)

        ds_clim_1[temp_name] -= 273.15
        ds_clim_2[temp_name] -= 273.15
        ds_zm_1 = ds_clim_1.mean([time_name, lon_name])
        ds_zm_2 = ds_clim_2.mean([time_name, lon_name])

        plot_zonal_mean_and_shading(
            ax,
            ds_zm_1[temp_name],
            ds_zm_2[temp_name],
            lat_name,
            pair_name,
            PAIR_COLORS[pair_name],
        )

        rotation = model_rotation(low_model)
        plat = data_csv[f"{rotation}_lat"]
        plon = data_csv[f"{rotation}_lon"]

        temp_low_mean = np.mean(
            [
                ds_clim_1[temp_name]
                .sel({lat_name: plat.iloc[idx], lon_name: plon.iloc[idx]}, method="nearest")
                .mean(time_name)
                .values
                for idx in range(2)
            ]
        )
        temp_high_mean = np.mean(
            [
                ds_clim_2[temp_name]
                .sel({lat_name: plat.iloc[idx], lon_name: plon.iloc[idx]}, method="nearest")
                .mean(time_name)
                .values
                for idx in range(2)
            ]
        )
        model_mean = (temp_low_mean + temp_high_mean) / 2
        model_min = np.minimum(temp_low_mean, temp_high_mean)
        model_max = np.maximum(temp_low_mean, temp_high_mean)

        ax.errorbar(
            np.mean([plat.iloc[0], plat.iloc[1]]),
            model_mean,
            yerr=np.array([[model_mean - model_min], [model_max - model_mean]]),
            fmt="o",
            color=PAIR_COLORS[pair_name],
            capsize=5,
            zorder=5,
        )

    plot_sites = data_csv[data_csv["location"].isin(PLOT_SITES)]
    proxy_mean = plot_sites["maat_mean"].mean()
    ax.errorbar(
        plot_sites["scotese_lat"].mean(),
        proxy_mean,
        yerr=np.array(
            [[proxy_mean - plot_sites["maat_min"].mean()], [plot_sites["maat_max"].mean() - proxy_mean]]
        ),
        fmt="o",
        color="black",
        label="Mongolia (brGDGT)",
        capsize=5,
        zorder=5,
    )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=2, zorder=0)
    ax.set_xlim(min(ds_clim_1[lat_name]), max(ds_clim_1[lat_name]))
    ax.set_xlabel("Latitude ($^\\circ$N)")
    ax.set_ylabel("Zonal Mean Temperature ($^\\circ$C)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODEL_PAIRS) + 1, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_figure(fig, "meridional_temp_gradients.pdf")


def plot_difference_to_zonal_mean(output_df: pd.DataFrame) -> None:
    filtered_df = output_df[output_df["model"].isin(MODEL_IDS)]
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    for idx, model_id in enumerate(MODEL_IDS):
        alpha = 0.3 if model_id in {"KCM_600", "texzx1", "tfksx", "teuyO"} else 1.0
        model_data_temp = filtered_df[
            (filtered_df["model"] == model_id) & (filtered_df["variable"] == "temperature")
        ]
        model_data_prec = filtered_df[
            (filtered_df["model"] == model_id) & (filtered_df["variable"] == "precipitation")
        ]

        for site, marker in [("TSG", "o"), ("SVO", "s")]:
            temp_site = model_data_temp[model_data_temp["site"] == site]
            prec_site = model_data_prec[model_data_prec["site"] == site]
            axs[0].scatter(
                temp_site["pheight"],
                temp_site["annual_mean_location"] - temp_site["annual_mean_zonal_mean"],
                label=f"{MODEL_LABELS[idx]} - {site}",
                marker=marker,
                color=MODEL_COLORS[idx],
                s=150,
                alpha=alpha,
            )
            axs[1].scatter(
                prec_site["pheight"],
                prec_site["annual_mean_location"] - prec_site["annual_mean_zonal_mean"],
                label=f"{MODEL_LABELS[idx]} - {site}",
                marker=marker,
                color=MODEL_COLORS[idx],
                s=150,
                alpha=alpha,
            )

    temp_diff = (
        filtered_df[filtered_df["variable"] == "temperature"]["annual_mean_location"]
        - filtered_df[filtered_df["variable"] == "temperature"]["annual_mean_zonal_mean"]
    )
    prec_diff = (
        filtered_df[filtered_df["variable"] == "precipitation"]["annual_mean_location"]
        - filtered_df[filtered_df["variable"] == "precipitation"]["annual_mean_zonal_mean"]
    )

    axs[0].axhline(y=temp_diff.mean(), color="black", linestyle="-", linewidth=3, label="Ensemble Mean")
    axs[1].axhline(y=prec_diff.mean(), color="black", linestyle="-", linewidth=3, label="Ensemble Mean")
    axs[0].axhline(y=0, color="gray", linestyle="--", linewidth=2)
    axs[1].axhline(y=0, color="gray", linestyle="--", linewidth=2)

    axs[0].set_title("Local vs. Zonal Mean Temperature Difference")
    axs[0].set_xlabel("Local Model Height (m)")
    axs[0].set_ylabel("Temperature Difference (degC)")
    axs[0].set_ylim(-6, 1)

    axs[1].set_title("Local vs. Zonal Mean Precipitation Difference")
    axs[1].set_xlabel("Local Model Height (m)")
    axs[1].set_ylabel("Annual Mean Precipitation Difference (mm/day)")
    axs[1].set_ylim(-2.4, 0.4)
    axs[1].legend()

    plt.tight_layout()
    save_figure(fig, "diff_to_zonal_mean.pdf")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data_csv = reconstruct_proxy_locations()
    plot_geographies(data_csv)

    temperature_data = [load_annual_temperature(model_id) for model_id in MODEL_IDS]
    temp_levels = np.arange(-15, 40, 5)
    temp_anomaly_levels = np.arange(-10, 10, 2)
    plot_regional_maps(
        temperature_data,
        data_csv,
        title="Aptian Surface Air Temperatures",
        filename="regional_aptian_temperature_maps.pdf",
        cmap=cmocean.cm.thermal,
        levels=temp_levels,
        colorbar_label="annual mean surface air temperature ($^\\circ$C)",
    )
    plot_anomaly_maps(
        temperature_data,
        data_csv,
        title="Aptian Surface Air Temperature Anomalies",
        filename="regional_aptian_temperature_anomaly_maps.pdf",
        normal_cmap=cmocean.cm.thermal,
        normal_levels=temp_levels,
        anomaly_levels=temp_anomaly_levels,
        anomaly_cmap="RdBu_r",
        colorbar_label="annual mean surface air temperature ($^\\circ$C)",
    )

    precipitation_data = [load_annual_precipitation(model_id) for model_id in MODEL_IDS]
    pr_levels = np.arange(1, 11, 1)
    pr_anomaly_levels = np.arange(-5, 6, 1)
    plot_regional_maps(
        precipitation_data,
        data_csv,
        title="Aptian Precipitation",
        filename="regional_aptian_precipitation_maps.pdf",
        cmap=cmocean.cm.rain,
        levels=pr_levels,
        colorbar_label="annual mean precipitation (mm/day)",
    )
    plot_anomaly_maps(
        precipitation_data,
        data_csv,
        title="Aptian Precipitation Anomalies",
        filename="regional_aptian_precipitation_anomaly_maps.pdf",
        normal_cmap=cmocean.cm.rain,
        normal_levels=pr_levels,
        anomaly_levels=pr_anomaly_levels,
        anomaly_cmap="BrBG",
        colorbar_label="annual mean precipitation (mm/day)",
        skip_model_ids={"teuyO"},
    )

    output_df = extract_location_data(data_csv)
    plot_annual_cycle(output_df, data_csv)
    plot_meridional_temperature_gradient(data_csv)
    plot_difference_to_zonal_mean(output_df)

    print(f"Wrote processed data to {PROCESSED_DIR}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
