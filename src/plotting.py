import numpy as np
import matplotlib.colors as mcolors

def split_cmap(cmap, start, end):
    return mcolors.LinearSegmentedColormap.from_list(
        'truncated_colormap', cmap(np.linspace(start, end, 256))
        )

