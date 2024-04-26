import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from src.helper import find_geo_coords


def split_cmap(cmap, start, end):
    """
    Splits a given colormap into a truncated colormap.

    Parameters:
    - cmap (matplotlib.colors.Colormap): The original colormap to be split.
    - start (float): The starting point of the truncated colormap (between 0 and 1).
    - end (float): The ending point of the truncated colormap (between 0 and 1).

    Returns:
    - matplotlib.colors.LinearSegmentedColormap: The truncated colormap.

    """
    return mcolors.LinearSegmentedColormap.from_list(
        'truncated_colormap', cmap(np.linspace(start, end, 256))
        )

def plot_filled_map(ax, data, type="contourf", cmap="viridis", levels=None, extent=None, right_labels=False, title='', **kwargs):
    """
    Plot a filled map of a variable in an xarray dataset using standard matplotlib plotting.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to plot the map on.
    - data (xarray.DataArray): The data to plot.
    - type (str, optional): The type of plot to create. Default is "contourf".
    - cmap (matplotlib.colors.Colormap, optional): The colormap to use. Default is "viridis".
    - levels (array-like, optional): The explicit levels for contourf or pcolormesh plot. Default is None.
    - **kwargs: Additional keyword arguments to pass to the plot function.

    Returns:
    - p (matplotlib.contour.QuadContourSet or matplotlib.collections.QuadMesh): The plot object.

    """

    # Find the variable name for latitude and longitude
    lon_name, lat_name = find_geo_coords(data)

    if type == "contourf":
        p = ax.contourf(
            data[lon_name],
            data[lat_name],
            data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            levels=levels,
            **kwargs
        )
    elif type == "pcolormesh":
        # Create a colormap based on levels for discrete color mapping
        if levels is not None:
            norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        p = ax.pcolormesh(
            data[lon_name],
            data[lat_name],
            data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            **kwargs
        )

    # Set the geographic extent if specified
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add gridlines and labels
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--', x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = right_labels
    gl.xlabel_style = {'size': 6, 'color': 'black'}
    gl.ylabel_style = {'size': 6, 'color': 'black'}

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)

    return p

def plot_contours(ax, data, levels, colors=['black'], linewidths=[1]):

    # Find the variable name for latitude and longitude
    lon_name, lat_name = find_geo_coords(data)

    p = ax.contour(
        data[lon_name],
        data[lat_name],
        data,
        transform=ccrs.PlateCarree(),
        levels=levels,
        colors=colors,
        linewidths=linewidths
    )

    return p
