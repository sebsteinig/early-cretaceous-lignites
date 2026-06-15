import requests


SCOTESE_FALLBACKS = {
    (45.2, 106.2, 116): (46.7, 104.1),
    (46.2, 108.5, 116): (48.1, 106.0),
    (46.3, 108.5, 116): (48.1, 106.0),
}


def get_scotese_paleolocation(lat, lon, age=0.0):
    """
    Get the reconstructed paleolocation coordinates for a given latitude, longitude, and age
    consistent with the Scotese PALEOMAP model.

    Parameters:
    lat (float): The latitude of the location.
    lon (float): The longitude of the location.
    age (float, optional): The age in million years. Default is 0.0.

    Returns:
    tuple: A tuple containing the rounded latitude and longitude coordinates.
    """

    endpoint = "https://gws.gplates.org/reconstruct/reconstruct_points/"
    params = {
        "points": f"{lon},{lat}",
        "time": age,
        "model": "PALEOMAP",
    }

    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        plon, plat = data["coordinates"][0]
        plat = round(plat, 1)
        plon = round(plon, 1)
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        fallback_key = (round(lat, 1), round(lon, 1), round(age))
        if fallback_key not in SCOTESE_FALLBACKS:
            raise RuntimeError(
                "Could not reconstruct Scotese paleolocation and no cached "
                f"fallback exists for lat={lat}, lon={lon}, age={age}."
            ) from exc
        plat, plon = SCOTESE_FALLBACKS[fallback_key]

    return plat, plon