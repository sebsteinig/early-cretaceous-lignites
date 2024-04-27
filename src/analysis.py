# module for defining functions to help with the analysis    # get reconstructed paleo locations and add them as red circle

import cartopy.crs as ccrs

def add_proxy_location_markers(ax, exp, data_csv, size=10):
    """
    Add proxy locations as red circles to a set of axes.

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        List of axes to add the proxy locations to.
    data_csv : pandas.DataFrame
        Dataframe containing the proxy locations.
    """
    if exp in ["KCM_600", "KCM_1200"]:
        rotation = 'kcm'
    elif exp in ["texzx1", "texpx2", "tfkex", "tfksx"]:
        rotation = 'scotese'
    elif exp in ["teuyO", "teuyo1"]:
        rotation = 'getech'
    plat = data_csv.loc[:, f'{rotation}_lat']
    plon = data_csv.loc[:, f'{rotation}_lon']

    ax.plot(plon, plat, 'ro', markersize=size, markeredgecolor='black', transform=ccrs.PlateCarree())
