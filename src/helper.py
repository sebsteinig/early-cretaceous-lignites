def find_varname_from_unit(ds, unit):
    """
    Find the variable name in a xarray dataset that has a specific unit.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to search for variable name.
    unit : str
        Unit to search for.

    Returns
    -------
    str
        Variable name with the specific unit.
    """
    var_name = None
    for var_name, var_data in ds.variables.items():
        if var_data.attrs.get('units') == unit:
            return var_name
    return var_name

def find_varname_from_keywords(ds, keywords):
    """
    Find the variable name in a xarray dataset that contains specific keywords in its long name.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to search for variable name.
    keywords : list of str
        Keywords to search for in the variable's long name.

    Returns
    -------
    str
        Variable name where the long name contains all the specified keywords, or None if no such variable exists.
    """
    for var_name, var_data in ds.variables.items():
        # Retrieve the long name attribute of the variable
        long_name = var_data.attrs.get('long_name', '').lower()
        
        # Check if all keywords are present in the long name
        if any(keyword.lower() in long_name for keyword in keywords):
            return var_name

    return None  # Return None if no variable matches all keywords


def find_geo_coords(ds):
    """
    Find the latitude and longitude variable names in a xarray dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to search for latitude and longitude variable names.

    Returns
    -------
    tuple
        A tuple containing the longitude and latitude variable names.
    
    Raises
    ------
    ValueError
        If the latitude or longitude variable names cannot be determined automatically.
    """
    # Common names for latitude and longitude variables
    possible_lat_names = ['latitude', 'lat', 'LAT', 'Latitude']
    possible_lon_names = ['longitude', 'lon', 'LON', 'Longitude']

    lat_name = None
    lon_name = None

    # Find latitude and longitude in dataset
    for var in ds.coords:
        if var in possible_lat_names:
            lat_name = var
        if var in possible_lon_names:
            lon_name = var

    if lat_name is None or lon_name is None:
        raise ValueError("Could not automatically determine the latitude or longitude variable names.")
    
    return lon_name, lat_name
