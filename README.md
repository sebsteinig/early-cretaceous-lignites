# Early Cretaceous land temperatures in Mongolia
This repo should contain all documentation, scripts and data to recreate the model-data comparison of Early Cretaceous land temperatures in Mongolia.

## purpose
Comapre new lignite-based land surface temperature reconstructions (brGDGT) with a suite of available model simulations to see whether absolute temperatures are in broad agreement with existing model results. Simulation differences include different models, paleogeographies and CO2 levels.

## source data
- `data/raw/proxy_temps_and_locations.csv`: CSV file of annual mean air temperature reconstructions at two sites
- `data/raw/model_clims`: gridded monthly mean climatologies of global simulation output + orography and land-sea-mask boundary conditions (netCDF)

## models considered
The following models were analysed to assess the modelled range in temperatures for different models, CO2 levels, paleogeographies and model physics: 

| model ID | model    | CO2 (ppm) | geography     | physics                    |
| -------- | -------- | --------- | ------------- | -------------------------- |
| KCM_600  | KCM      | 600       | Müller/Blakey | ECHAM5 default             |
| KCM_1200 | KCM      | 1200      | Müller/Blakey | ECHAM5 default             |
| teuyO    | HadCM3BL | 560       | Getech        | Valdes et al. (2017)       |
| teuyo    | HadCM3BL | 1120      | Getech        | Valdes et al. (2017)       |
| texzx1   | HadCM3BL | 560       | Scotese       | Valdes et al. (2017)       |
| texpx2   | HadCM3BL | 1103      | Scotese       | Valdes et al. (2017)       |
| tfksx    | HadCM3BL | 780       | Scotese       | new (improved polar. amp.) |
| tfkex    | HadCM3BL | 1103      | Scotese       | new (improved polar. amp.) |

## running the notebooks
Easiest way to run locally is to first download the repo with

```
git clone https://github.com/USERNAME/REPOSITORY
``` 

and then install [conda](https://conda.io/projects/conda/en/latest/index.html) (if not installed already). Then create an environment `env_name` with 

```
conda env create --name env_name --file=environment.yml
``` 

using the `environment.yml` file from this repository to install all necessary python packages. The notebooks can then be run interactively by typing

```
jupyter lab
```

I use the [jupytext](https://jupytext.readthedocs.io/en/latest/index.html) package to develop the notebooks as text notebooks in the `py:percent` [format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format). This is very helpful for clean diffs in the version control and allows you to run the analysis in your local terminal with:

```
python notebook_name.py
```
The python file can also be shared with others to work on the code together using all the version control benefits (branches, pull requests, ...). You can edit it with any tex editor/IDE and it can also be converted back to a jupyter notebook (with no output) via
```
jupytext --to notebook notebook_name.py
```
or by opening them as a Notebook with `jupytext`. Final Notebooks in the `notebooks/publication` directory are always available as `.ipynb` Notebooks for convenience.

