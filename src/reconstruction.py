import requests
import json

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

    # Define the endpoint from teh GPlates Web Service
    endpoint = "https://gws.gplates.org/reconstruct/reconstruct_points/"

    # Define the parameters
    params = {
        "points": f"{lon},{lat}",
        "time": age,
        "model": "PALEOMAP"
    }

    # Send the request
    response = requests.get(endpoint, params=params)

    # Check the status of the request
    if response.status_code == 200:
        # Parse the response
        data = json.loads(response.text)
        # Extract the reconstructed coordinates
        plon, plat = data["coordinates"][0]
        # Round the coordinates to one decimal place
        plat = round(plat, 1)
        plon = round(plon, 1)
    else:
        print(f"Error: {response.status_code}")
        plat, plon = None, None

    return plat, plon