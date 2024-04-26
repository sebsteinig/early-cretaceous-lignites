# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Aptian land temperatures model-data comparison
# Comapre new lignite-based land surface temperature reconstructions (brGDGT) with a suite of available model simulations to see whether absolute temperatures are in broad agreement with existing model results. Simulation differences include different models, paleogeographies and CO2 levels.

# %% [markdown]
# ## User input

# %%
work_dir       = '/Users/wb19586/Documents/coding_github/early-cretaceous-lignites' # location of cloned repository
data_dir       = work_dir + '/data' # location of data files
fig_dir        = work_dir + '/figures' # location of figure files

save_figures   = True # flag whether to save figures to disk or not

# %% [markdown]
# ## model overview
# | model ID | model    | CO2 (ppm) | geography     | physics                    |
# | -------- | -------- | --------- | ------------- | -------------------------- |
# | KCM_600  | KCM      | 600       | Müller/Blakey | ECHAM5 default             |
# | KCM_1200 | KCM      | 1200      | Müller/Blakey | ECHAM5 default             |
# | teuyO    | HadCM3BL | 560       | Getech        | Valdes et al. (2017)       |
# | teuyo    | HadCM3BL | 1120      | Getech        | Valdes et al. (2017)       |
# | texzx1   | HadCM3BL | 560       | Scotese       | Valdes et al. (2017)       |
# | texpx2   | HadCM3BL | 1103      | Scotese       | Valdes et al. (2017)       |
# | tfksx    | HadCM3BL | 780       | Scotese       | new (improved polar. amp.) |
# | tfkex    | HadCM3BL | 1103      | Scotese       | new (improved polar. amp.) |

# %% [markdown]
# ## Main code

# %%
# laod packages
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cmocean
import csv

# adjust Python path to import local modules
import sys
sys.path.append(work_dir)

from src.helper import find_varname_from_unit
from src.plotting import split_cmap

# %% [markdown]
# ### paleogeographic differences
# We have model simulations with three different Aptian paleogeographies (see table above). Start by plotting them side-by-side for the study region.

# %%
exp_list = ['KCM_1200', 'texzx1', 'teuyO']
exp_labels = ['KCM (Müller/Blakey)', 'HadCM3 (Scotese)', 'HadCM3 (Getech)']

# new multi-panel figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': ccrs.Robinson()})

for idx,exp in enumerate(exp_list):
    # load data
    ds_orog = xr.open_dataset(f"{data_dir}/raw/model_clims/{exp}.orog.nc").squeeze()
    ds_mask = xr.open_dataset(f"{data_dir}/raw/model_clims/{exp}.mask.nc").squeeze()

    # find the variable names
    orog_name = find_varname_from_unit(ds_orog, "m")
    mask_name = find_varname_from_unit(ds_mask, "m")

    # use land colors only from colormap
    cmap_topo = split_cmap(cmocean.cm.topo, 0.5, 1.0)

    ds_orog = ds_orog.where(ds_mask[mask_name] >= 0.5, np.nan)  # Filter out bad values

    # map plot with cartopy
    p = ds_orog[orog_name].plot.pcolormesh(
        ax=axes[idx], 
        transform=ccrs.PlateCarree(),
        # vmin=-1, vmax=2500, 
        levels = (0, 250, 500,750,1000,1250,1500,1750, 2000,2250,2500),
        cmap=cmap_orog,
        add_colorbar=False)
    
# add colorbar
fig.colorbar(p, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.05)


plt.show()


# %%

# %%

# %%

# %% [markdown]
# #### appendix: some code snippets I regularly use

# %% colab={"base_uri": "https://localhost:8080/", "height": 713} id="055047f3" outputId="8b79a52d-0d53-4631-91b9-e4fb17cfc984"
#### loop analysis over data sets   
# for expCount, exp in enumerate(exp_list):

#### load netcdf data set
# ds = xr.open_dataset(work_dir + '/data/file_name.nc')

#### new multi-panel figure
# fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=(width, height) ) # figsize in inches

#### map plot with cartopy
# ax = fig.add_subplot(nrows, ncols, index, projection=ccrs.Robinson()) # or e.g. ccrs.PlateCarree()
# ax.set_extent([minlon,maxlon, minlat,maxlat], ccrs.PlateCarree()) # or ax.set_global()
# ax.coastlines()
# ax.contourf(ds['variable_name'], transform=ccrs.PlateCarree(), levels=21, 
#             vmin=..., vmax=..., cmap=cmocean.cm.topo, add_colorbar=False)

#### add cyclic longitude to field and coordinate (from cartopy.util import add_cyclic_point)
# variable_cyclic, longitude_cyclic = add_cyclic_point(variable, coord=longitude)

#### save figure
# if save_figures:
#      plt.savefig(work_dir + '/figures/figure_name.pdf')  
#      plt.savefig(work_dir + '/figures/figure_name.png', dpi=200)  

