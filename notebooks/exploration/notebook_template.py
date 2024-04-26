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
# Enable the autoreload of modules
# %load_ext autoreload
# %autoreload 2

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

from src.helper import *
from src.plotting import *

# %% [markdown]
# ### paleogeographic differences
# We have model simulations with three different Aptian paleogeographies (see table above). Start by plotting them side-by-side for the study region.

# %%
exp_list = ['KCM_1200', 'texzx1', 'teuyO']
exp_labels = ['KCM (Müller/Blakey)', 'HadCM3 (Scotese)', 'HadCM3 (Getech)']
orog_levels = [0,250,500,750,1000,1250,1500,1750,2000,2250,2500]

# global comparison of paleogeographies

# Load all data first
data = []
for idx,exp in enumerate(exp_list):
    # load data
    ds_orog = xr.open_dataset(f"{data_dir}/raw/model_clims/{exp}.orog.nc").squeeze()
    ds_mask = xr.open_dataset(f"{data_dir}/raw/model_clims/{exp}.mask.nc").squeeze()

    # find the variable names
    orog_name = find_varname_from_unit(ds_orog, "m")
    mask_name = find_varname_from_keywords(ds_mask, ["land sea mask", "land/sea mask"])

    ds_orog = ds_orog.where(ds_mask[mask_name] >= 0.5, np.nan)  # Filter out bad values

    data.append((ds_orog, ds_mask, orog_name, mask_name))

# Plot the data
fig, axes = plt.subplots(1, 3, figsize=(12, 5), subplot_kw={'projection': ccrs.Robinson()})
fig.suptitle('Aptian Model Paleogeographies', fontsize=16, fontweight='bold', y = 0.7)

# use land colors only from colormap
cmap_topo = split_cmap(cmocean.cm.topo, 0.5, 1.0)

for idx, (ds_orog, ds_mask, orog_name, mask_name) in enumerate(data):
    # use land colors only from colormap
    cmap_topo = split_cmap(cmocean.cm.topo, 0.5, 1.0)

    # global map plot with cartopy
    p = plot_filled_map(
        ax=axes[idx],
        data=ds_orog[orog_name],
        type='pcolormesh',
        cmap=cmap_topo, 
        levels=orog_levels,
        right_labels=True,
        title=exp_labels[idx])
    
    # add coastlines
    plot_contours(ax=axes[idx], data=ds_mask[mask_name], levels=[0.5], colors=['black'], linewidths=[1])
    
# add common colorbar
cbar = fig.colorbar(p, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.08, aspect=40, shrink = 0.6, extend='max')
cbar.set_label('Model elevation (m)', fontsize=14)  
cbar.ax.tick_params(labelsize=12)

#### save figure
if save_figures:
     plt.savefig(fig_dir + '/global_aptian_geographies.pdf', bbox_inches='tight')  
     
plt.show()


# %%
# plot regional comparison of paleogeographies
fig, axes = plt.subplots(1, 3, figsize=(12, 5), subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle('Aptian Model Paleogeographies', fontsize=16, fontweight='bold', y = 0.82)

for idx, (ds_orog, ds_mask, orog_name, mask_name) in enumerate(data):
    # global map plot with cartopy
    p = plot_filled_map(
        ax=axes[idx],
        data=ds_orog[orog_name],
        type='pcolormesh',
        cmap=cmap_topo, 
        levels=orog_levels,
        extent=[50, 160, 0, 85],
        title=exp_labels[idx])
    
    # add coastlines
    plot_contours(ax=axes[idx], data=ds_mask[mask_name], levels=[0.5], colors=['black'], linewidths=[3])
    
# add common colorbar
cbar = fig.colorbar(p, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.08, aspect=40, shrink = 0.6, extend='max')
cbar.set_label('Model elevation (m)', fontsize=14)  
cbar.ax.tick_params(labelsize=12)

#### save figure
if save_figures:
     plt.savefig(fig_dir + '/regional_aptian_geographies.pdf', bbox_inches='tight')  
     
plt.show()
