"""Publication Figure 2: meridional temperature gradient.

Run from the repository root with:

    python notebooks/publication_figures/figure_2_meridional_temp_gradients.py
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

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
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
# The ``src`` package may live at the repo root or under ``notebooks``; add
# whichever directory actually contains it so imports work from either layout.
for _candidate in (REPO_ROOT, REPO_ROOT / "notebooks"):
    if (_candidate / "src").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.fonts import ensure_helvetica_bold
from src.helper import find_geo_coords, find_varname_from_attribute
from src.reconstruction import get_scotese_paleolocation

# macOS only registers regular Helvetica; pull in a bold face so semibold/bold
# weights render (instead of silently falling back to regular).
ensure_helvetica_bold()


DATA_DIR = REPO_ROOT / "data" / "v2"
MODEL_DIR = DATA_DIR / "raw" / "model_clims"
PROXY_LOCATIONS_FILE = (
    DATA_DIR / "processed" / "proxy_temps_and_reconstructed_locations.csv"
)
ADDITIONAL_PROXY_FILE = REPO_ROOT / "additional_data" / "MAAT data existing sites.xlsx"
FIG_DIR = REPO_ROOT / "figures" / "publication_v1"
OUTPUT_FILE = FIG_DIR / "figure_2_meridional_temp_gradients.pdf"
RECONSTRUCTION_AGE_MA = 116
XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

# Order matches the other figures: HadCM3 (Farnsworth) -> HadCM3 (Valdes) -> KCM.
MODEL_PAIRS = {
    "HadCM3+Getech": ("teuyO", "teuyo1"),
    "HadCM3+Scotese": ("texzx1", "texpx2"),
    "KCM": ("KCM_600", "KCM_1200"),
}
PLOT_SITES = ["TSG", "SVO"]
ADDITIONAL_PROXY_MARKERS = ["^", "s", "D", "P", "X", "v", "<", ">", "*", "h"]
MONGOLIA_PROXY_MEAN = 15.1
MONGOLIA_PROXY_UNCERTAINTY = 2.6
# Match the model names used in the orography panels (Figure 1 / SI), which
# cite the actual palaeogeography references rather than the internal tags.
MODEL_REFERENCE_NAMES = {
    "KCM": "KCM (Steinig et al., 2020)",
    "HadCM3+Getech": "HadCM3 (Farnsworth et al., 2019)",
    "HadCM3+Scotese": "HadCM3 (Valdes et al., 2021)",
}
MODEL_LEGEND_LABELS = {
    key: f"{name} zonal mean" for key, name in MODEL_REFERENCE_NAMES.items()
}
MODEL_SITE_LEGEND_LABELS = {
    key: f"{name} mean at study sites" for key, name in MODEL_REFERENCE_NAMES.items()
}
MONGOLIA_PROXY_LEGEND_LABEL = "this study (mean across sites)"
ADDITIONAL_PROXY_LEGEND_LABELS = {
    (
        "Changma Grotto",
        "Suarez",
        "2017",
    ): "Changma Grotto D47 lake sed. (Suarez et al. 2017)",
    (
        "Yujingzi Basin",
        "Harper",
        "2021",
    ): "Yujingzi Basin D47 soil carb. (Harper et al. 2021)",
    ("Mazongshan", "Amiot", "2010"): "Mazongshan d18O teeth (Amiot et al. 2010)",
    ("Sihetun", "Zhang", "2021"): "Sihetun D47 soil carb. (Zhang et al. 2021)",
    ("Fuxin", "Amiot", "2010"): "Fuxin d18O teeth (Amiot et al. 2010)",
    ("Sihetun", "Amiot", "2010"): "Sihetun d18O teeth (Amiot et al. 2010)",
}
PROXY_MATERIAL_ABBREVIATIONS = {
    "D47 of dolomitic lake sediment": "D47 lake sed.",
    "D47 soil carbonate": "D47 soil carb.",
    "d18O teeth": "d18O teeth",
}
FIGURE_STYLE = {
    "file": OUTPUT_FILE,
    # Condensed double-column footprint for the manuscript (was 15 x 10).
    "figsize": (9, 6),
    # Font sizes scaled ~0.72x from the original 15 x 10 layout so text stays
    # proportional on the smaller figure.
    "font_sizes": {
        "title": 14.4,
        "axis_label": 13.5,
        "ticks": 10.5,
        "legend": 7.6,
        "legend_title": 8.6,
    },
    "model_colors": {
        "KCM": "#D55E00",
        "HadCM3+Getech": "#0072B2",
        "HadCM3+Scotese": "#009E73",
    },
    "proxy_color": "#3F3F46",
    "additional_proxy_color": "#3F3F46",
    "additional_marker_face": "white",
    "figure_facecolor": "white",
    "axes_facecolor": "white",
    "grid_color": "#D9D9D9",
    "spine_color": "black",
    "line_width": 2.0,
    "shade_alpha": 0.18,
    "proxy_marker_size": 9.0,
    "proxy_legend_marker_size": 8.5,
    "proxy_error_linewidth": 1.8,
    # Thinner whiskers / error bars for the older literature reconstructions so
    # they recede behind the simulations.
    "additional_proxy_error_linewidth": 0.9,
    # Drawing order (back -> front): old literature reconstructions, then the
    # simulation shading / lines / site markers, then this study on top.
    "zorder": {
        "additional_whisker": 2,
        "additional_marker": 3,
        "model_shade": 4,
        "model_line": 6,
        "model_marker": 8,
        "this_study_whisker": 30,
        "this_study_marker": 31,
    },
    # Group the two legends around the figure centre, where the white space
    # beneath the temperature peak keeps them clear of the data. They butt up
    # to either side of the midline so the longer reference labels can't collide.
    "model_legend_kwargs": {
        "loc": "upper right",
        "bbox_to_anchor": (0.49, 0.34),
        "ncol": 1,
        "frameon": True,
    },
    "proxy_legend_kwargs": {
        "loc": "upper left",
        "bbox_to_anchor": (0.51, 0.34),
        "ncol": 1,
        "frameon": True,
    },
    "tight_rect": [0, 0, 1, 0.95],
}


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


def xlsx_column_index(cell_reference: str) -> int:
    column_letters = "".join(char for char in cell_reference if char.isalpha())
    index = 0
    for char in column_letters:
        index = index * 26 + ord(char.upper()) - ord("A") + 1
    return index - 1


def read_first_xlsx_sheet(path: Path) -> pd.DataFrame:
    """Read a simple .xlsx sheet without requiring optional pandas Excel engines."""
    with ZipFile(path) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_strings_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in shared_strings_root.findall("a:si", XLSX_NS):
                shared_strings.append(
                    "".join(text.text or "" for text in item.findall(".//a:t", XLSX_NS))
                )

        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        rels_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        rels = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_root}
        first_sheet = workbook_root.find("a:sheets/a:sheet", XLSX_NS)
        if first_sheet is None:
            raise ValueError(f"No sheets found in {path}")
        rel_id = first_sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        sheet_root = ET.fromstring(workbook.read(f"xl/{rels[rel_id]}"))

    rows = []
    for row in sheet_root.findall(".//a:sheetData/a:row", XLSX_NS):
        values = []
        for cell in row.findall("a:c", XLSX_NS):
            index = xlsx_column_index(cell.attrib["r"])
            while len(values) <= index:
                values.append("")

            cell_type = cell.attrib.get("t")
            value_element = cell.find("a:v", XLSX_NS)
            if cell_type == "inlineStr":
                value = "".join(
                    text.text or "" for text in cell.findall(".//a:t", XLSX_NS)
                )
            elif value_element is None:
                value = ""
            elif cell_type == "s":
                value = shared_strings[int(value_element.text)]
            else:
                value = value_element.text
            values[index] = value
        rows.append(values)

    if not rows:
        return pd.DataFrame()

    header = rows[0]
    records = [
        {
            header[index]: row[index] if index < len(row) else ""
            for index in range(len(header))
        }
        for row in rows[1:]
    ]
    return pd.DataFrame(records)


def parse_dms_coordinate(value: str) -> float:
    normalized = str(value).replace(",", ".")
    parts = [float(part) for part in re.findall(r"\d+(?:\.\d+)?", normalized)]
    if not parts:
        raise ValueError(f"Could not parse coordinate: {value}")
    degrees = parts[0]
    minutes = parts[1] if len(parts) > 1 else 0
    seconds = parts[2] if len(parts) > 2 else 0
    sign = (
        -1
        if "-" in normalized or any(hemi in normalized.upper() for hemi in ["S", "W"])
        else 1
    )
    return sign * (degrees + minutes / 60 + seconds / 3600)


def parse_temperature_estimate(value: str) -> tuple[float, float, float]:
    normalized = (
        str(value)
        .replace(",", ".")
        .replace("±", "+/-")
        .replace("–", "-")
        .replace("−", "-")
    )
    numbers = [float(number) for number in re.findall(r"-?\d+(?:\.\d+)?", normalized)]
    if not numbers:
        raise ValueError(f"Could not parse MAAT value: {value}")

    mean = numbers[0]
    if "+/-" in normalized and "to" in normalized and len(numbers) >= 5:
        minimum = numbers[1] - abs(numbers[2])
        maximum = numbers[3] + abs(numbers[4])
    elif "+/-" in normalized and len(numbers) >= 2:
        uncertainty = abs(numbers[1])
        minimum = mean - uncertainty
        maximum = mean + uncertainty
    elif len(numbers) >= 3:
        minimum = numbers[1]
        maximum = numbers[2]
    else:
        minimum = mean
        maximum = mean

    return mean, minimum, maximum


def should_skip_additional_proxy_record(row: pd.Series) -> bool:
    return (
        str(row.get("author", "")).strip() == "Amiot"
        and str(row.get("year", "")).strip() == "2010"
        and str(row.get("formation", "")).strip() == "Shahai Fm."
        and str(row.get("Site", "")).strip() == "Fuxin"
    )


def additional_proxy_legend_label(proxy: pd.Series) -> str:
    key = (
        str(proxy["site"]).strip(),
        str(proxy["author"]).strip(),
        str(proxy["year"]).strip(),
    )
    material = str(proxy.get("material", "")).strip()
    material_label = PROXY_MATERIAL_ABBREVIATIONS.get(material, material)
    return ADDITIONAL_PROXY_LEGEND_LABELS.get(
        key,
        f"{key[0]} ({key[1]} {key[2]}, {material_label})",
    )


def load_additional_proxy_data() -> pd.DataFrame:
    raw_data = read_first_xlsx_sheet(ADDITIONAL_PROXY_FILE)
    records = []
    for _, row in raw_data.iterrows():
        if not row.get("Site") or not row.get("MAAT"):
            continue
        if should_skip_additional_proxy_record(row):
            continue

        modern_lat = parse_dms_coordinate(row["latitude"])
        modern_lon = parse_dms_coordinate(row["longitude"])
        scotese_lat, scotese_lon = get_scotese_paleolocation(
            modern_lat,
            modern_lon,
            age=RECONSTRUCTION_AGE_MA,
        )
        maat_mean, maat_min, maat_max = parse_temperature_estimate(row["MAAT"])
        records.append(
            {
                "author": row["author"],
                "year": row["year"],
                "formation": row["formation"],
                "material": row["material"],
                "site": row["Site"],
                "modern_lat": modern_lat,
                "modern_lon": modern_lon,
                "scotese_lat": scotese_lat,
                "scotese_lon": scotese_lon,
                "maat_mean": maat_mean,
                "maat_min": maat_min,
                "maat_max": maat_max,
            }
        )

    return pd.DataFrame(records)


def proxy_latitude_uncertainty(proxy_locations: pd.DataFrame) -> float:
    paleolat_columns = [
        column for column in proxy_locations.columns if column.endswith("_lat")
    ]
    ranges = proxy_locations[paleolat_columns].max(axis=1) - proxy_locations[
        paleolat_columns
    ].min(axis=1)
    return float(ranges[proxy_locations["location"].isin(PLOT_SITES)].mean())


def plot_zonal_mean_and_shading(
    ax: plt.Axes,
    low_co2_data: xr.DataArray,
    high_co2_data: xr.DataArray,
    lat_name: str,
    label: str,
    color: str,
    line_width: float,
    shade_alpha: float,
    line_zorder: int,
    shade_zorder: int,
) -> None:
    mean = (low_co2_data + high_co2_data) / 2
    minimum = xr.apply_ufunc(np.minimum, low_co2_data, high_co2_data)
    maximum = xr.apply_ufunc(np.maximum, low_co2_data, high_co2_data)

    ax.plot(
        low_co2_data[lat_name],
        mean,
        label=label,
        color=color,
        linewidth=line_width,
        zorder=line_zorder,
    )
    ax.fill_between(
        low_co2_data[lat_name],
        minimum,
        maximum,
        color=color,
        alpha=shade_alpha,
        linewidth=0,
        zorder=shade_zorder,
    )


def load_temperature_dataset(model_id: str) -> tuple[xr.Dataset, str, str, str, str]:
    ds_clim = xr.open_dataset(
        model_file(model_id, "clim"), decode_times=False
    ).squeeze()
    temp_name = find_varname_from_attribute(ds_clim, "units", "K")
    time_name = find_varname_from_attribute(ds_clim, "axis", "T")
    lon_name, lat_name = find_geo_coords(ds_clim)
    ds_clim[temp_name] -= 273.15
    return ds_clim, temp_name, time_name, lat_name, lon_name


def plot_model_pair(
    ax: plt.Axes,
    pair_name: str,
    low_model: str,
    high_model: str,
    proxy_locations: pd.DataFrame,
    style: dict,
) -> tuple[xr.Dataset, str]:
    low_ds, temp_name, time_name, lat_name, lon_name = load_temperature_dataset(
        low_model
    )
    high_ds, _, _, _, _ = load_temperature_dataset(high_model)

    low_zonal_mean = low_ds.mean([time_name, lon_name])
    high_zonal_mean = high_ds.mean([time_name, lon_name])
    plot_zonal_mean_and_shading(
        ax,
        low_zonal_mean[temp_name],
        high_zonal_mean[temp_name],
        lat_name,
        pair_name,
        style["model_colors"][pair_name],
        style["line_width"],
        style["shade_alpha"],
        style["zorder"]["model_line"],
        style["zorder"]["model_shade"],
    )

    rotation = model_rotation(low_model)
    site_locations = proxy_locations.head(2)
    paleolats = site_locations[f"{rotation}_lat"]
    paleolons = site_locations[f"{rotation}_lon"]

    low_site_mean = np.mean(
        [
            low_ds[temp_name]
            .sel(
                {lat_name: paleolats.iloc[idx], lon_name: paleolons.iloc[idx]},
                method="nearest",
            )
            .mean(time_name)
            .values
            for idx in range(len(site_locations))
        ]
    )
    high_site_mean = np.mean(
        [
            high_ds[temp_name]
            .sel(
                {lat_name: paleolats.iloc[idx], lon_name: paleolons.iloc[idx]},
                method="nearest",
            )
            .mean(time_name)
            .values
            for idx in range(len(site_locations))
        ]
    )
    model_mean = (low_site_mean + high_site_mean) / 2
    model_min = np.minimum(low_site_mean, high_site_mean)
    model_max = np.maximum(low_site_mean, high_site_mean)

    ax.errorbar(
        paleolats.mean(),
        model_mean,
        yerr=np.array([[model_mean - model_min], [model_max - model_mean]]),
        fmt="o",
        color=style["model_colors"][pair_name],
        capsize=5,
        markersize=style["proxy_marker_size"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=style["zorder"]["model_marker"],
    )

    return low_ds, lat_name


def plot_horizontal_uncertainty(
    ax: plt.Axes,
    latitudes: float | pd.Series,
    temperatures: float | pd.Series,
    latitude_uncertainty: float,
    color: str,
    line_width: float,
    zorder: int = 4,
) -> None:
    latitudes = np.atleast_1d(latitudes)
    temperatures = np.atleast_1d(temperatures)
    for latitude, temperature in zip(latitudes, temperatures):
        ax.hlines(
            temperature,
            latitude - latitude_uncertainty,
            latitude + latitude_uncertainty,
            color=color,
            linestyle="-",
            linewidth=line_width,
            zorder=zorder,
        )


def plot_proxy_estimate(
    ax: plt.Axes,
    proxy_locations: pd.DataFrame,
    additional_proxy_data: pd.DataFrame,
    latitude_uncertainty: float,
    style: dict,
) -> None:
    proxy_sites = proxy_locations[proxy_locations["location"].isin(PLOT_SITES)]
    proxy_mean = MONGOLIA_PROXY_MEAN
    proxy_lower_error = MONGOLIA_PROXY_UNCERTAINTY
    proxy_upper_error = MONGOLIA_PROXY_UNCERTAINTY
    proxy_latitude = proxy_sites["scotese_lat"].mean()

    # "this study" sits in front of everything else: whiskers, error bars and a
    # filled black circle on top.
    plot_horizontal_uncertainty(
        ax,
        proxy_latitude,
        proxy_mean,
        latitude_uncertainty,
        color=style["proxy_color"],
        line_width=style["proxy_error_linewidth"],
        zorder=style["zorder"]["this_study_whisker"],
    )
    ax.errorbar(
        proxy_latitude,
        proxy_mean,
        yerr=np.array([[proxy_lower_error], [proxy_upper_error]]),
        fmt="o",
        color=style["proxy_color"],
        label=MONGOLIA_PROXY_LEGEND_LABEL,
        capsize=5,
        elinewidth=style["proxy_error_linewidth"],
        markersize=style["proxy_marker_size"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=style["zorder"]["this_study_marker"],
    )

    # Older literature reconstructions recede into the background with thinner
    # whiskers / error bars so the simulations read in front of them.
    for idx, (_, proxy) in enumerate(additional_proxy_data.iterrows()):
        marker = ADDITIONAL_PROXY_MARKERS[idx % len(ADDITIONAL_PROXY_MARKERS)]
        label = additional_proxy_legend_label(proxy)
        plot_horizontal_uncertainty(
            ax,
            proxy["scotese_lat"],
            proxy["maat_mean"],
            latitude_uncertainty,
            color=style["additional_proxy_color"],
            line_width=style["additional_proxy_error_linewidth"],
            zorder=style["zorder"]["additional_whisker"],
        )
        ax.errorbar(
            proxy["scotese_lat"],
            proxy["maat_mean"],
            yerr=np.array(
                [
                    [proxy["maat_mean"] - proxy["maat_min"]],
                    [proxy["maat_max"] - proxy["maat_mean"]],
                ]
            ),
            fmt=marker,
            color=style["additional_proxy_color"],
            markerfacecolor=style["additional_marker_face"],
            markeredgecolor=style["additional_proxy_color"],
            markeredgewidth=1.2,
            label=label,
            capsize=4,
            elinewidth=style["additional_proxy_error_linewidth"],
            capthick=style["additional_proxy_error_linewidth"],
            linestyle="none",
            markersize=style["proxy_marker_size"],
            zorder=style["zorder"]["additional_marker"],
        )


def latitude_tick_label(value: float, _position: int | None = None) -> str:
    """Format a latitude tick as e.g. ``60°S`` / ``0°`` / ``30°N``.

    Carries the hemisphere on the tick labels themselves so the x-axis title
    can be dropped entirely.
    """
    degrees = int(round(value))
    if degrees > 0:
        return f"{degrees}°N"
    if degrees < 0:
        return f"{abs(degrees)}°S"
    return "0°"


def style_legend_frame(legend: matplotlib.legend.Legend, style: dict) -> None:
    if legend.get_frame_on():
        legend.get_frame().set_facecolor(style["figure_facecolor"])
        legend.get_frame().set_edgecolor(style["grid_color"])


def style_axis(fig: plt.Figure, ax: plt.Axes, style: dict) -> None:
    fig.patch.set_facecolor(style["figure_facecolor"])
    ax.set_facecolor(style["axes_facecolor"])
    ax.grid(
        True,
        which="major",
        axis="y",
        color=style["grid_color"],
        linewidth=0.8,
        alpha=0.9,
    )
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        labelsize=style.get("font_sizes", {}).get("ticks", 12),
        colors=style["spine_color"],
    )
    for spine in ax.spines.values():
        spine.set_color(style["spine_color"])
        spine.set_linewidth(1.1)


def make_figure(
    style: dict,
    proxy_locations: pd.DataFrame,
    additional_proxy_data: pd.DataFrame,
    latitude_uncertainty: float,
) -> None:
    fig, ax = plt.subplots(
        1, 1, figsize=style.get("figsize", (15, 10)), facecolor=style["figure_facecolor"]
    )
    style_axis(fig, ax, style)
    last_ds = None
    last_lat_name = None

    for pair_name, (low_model, high_model) in MODEL_PAIRS.items():
        last_ds, last_lat_name = plot_model_pair(
            ax, pair_name, low_model, high_model, proxy_locations, style
        )

    plot_proxy_estimate(
        ax,
        proxy_locations,
        additional_proxy_data,
        latitude_uncertainty,
        style,
    )

    ax.axhline(
        y=0,
        color=style["spine_color"],
        linestyle="--",
        linewidth=1.8,
        zorder=0,
    )
    if last_ds is not None and last_lat_name is not None:
        ax.set_xlim(min(last_ds[last_lat_name]), max(last_ds[last_lat_name]))
    fonts = style.get("font_sizes", {})
    # Latitude is carried on the tick labels (e.g. "60°S"), so the x-axis title
    # is dropped entirely.
    ax.xaxis.set_major_formatter(FuncFormatter(latitude_tick_label))
    ax.set_ylabel(
        "surface temperature (°C)",
        fontsize=fonts.get("axis_label", 14),
        color=style["spine_color"],
    )

    handles, labels = ax.get_legend_handles_labels()
    model_keys = [label for label in labels if label in MODEL_PAIRS]
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            color=style["model_colors"][label],
            linewidth=style["line_width"],
        )
        for label in model_keys
    ]
    model_site_handles = [
        plt.Line2D(
            [0],
            [0],
            color=style["model_colors"][label],
            marker="o",
            linestyle="none",
            markersize=style["proxy_legend_marker_size"],
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        for label in model_keys
    ]
    model_handles = model_handles + model_site_handles
    model_labels = [MODEL_LEGEND_LABELS.get(label, label) for label in model_keys] + [
        MODEL_SITE_LEGEND_LABELS.get(label, f"{label} Mongolia site")
        for label in model_keys
    ]
    proxy_labels = [label for label in labels if label not in MODEL_PAIRS]
    proxy_markers = ["o"] + ADDITIONAL_PROXY_MARKERS[: len(proxy_labels) - 1]
    proxy_handles = [
        plt.Line2D(
            [0],
            [0],
            color=style["proxy_color"],
            marker=marker,
            linestyle="none",
            markersize=style["proxy_legend_marker_size"],
            markerfacecolor=(
                style["proxy_color"] if idx == 0 else style["additional_marker_face"]
            ),
            markeredgecolor=style["proxy_color"],
            markeredgewidth=1.2,
        )
        for idx, marker in enumerate(proxy_markers)
    ]

    model_legend = ax.legend(
        model_handles,
        model_labels,
        fontsize=fonts.get("legend", 10.5),
        title="Simulations",
        title_fontsize=fonts.get("legend_title", 12),
        **style["model_legend_kwargs"],
    )
    style_legend_frame(model_legend, style)
    ax.add_artist(model_legend)

    proxy_legend = ax.legend(
        proxy_handles,
        proxy_labels,
        fontsize=fonts.get("legend", 10.5),
        title="Reconstructions",
        title_fontsize=fonts.get("legend_title", 12),
        **style["proxy_legend_kwargs"],
    )
    style_legend_frame(proxy_legend, style)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=style["tight_rect"])
    fig.savefig(style["file"], bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {style['file']}")


def main() -> None:
    proxy_locations = pd.read_csv(PROXY_LOCATIONS_FILE)
    additional_proxy_data = load_additional_proxy_data()
    latitude_uncertainty = proxy_latitude_uncertainty(proxy_locations)
    make_figure(
        FIGURE_STYLE,
        proxy_locations,
        additional_proxy_data,
        latitude_uncertainty,
    )


if __name__ == "__main__":
    main()
