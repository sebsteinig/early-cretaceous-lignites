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