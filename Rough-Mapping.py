# %% [markdown]
# # Analysis of Electrophysiology Data and Optogenetic Manipulation

# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

sys.path += ['physion/src'] # add src code directory for physion
import physion.utils.plot_tools as pt

from physion.analysis.read_NWB import Data,\
    scan_folder_for_NWBfiles

dataset = scan_folder_for_NWBfiles(\
        os.path.join(os.path.expanduser('~'), 'DATA'),
            # for_protocol='flashed-stimuli')
            for_protocol='rough-mapping')
            # )

# %%
pt.set_style('dark')#'manuscript')
from scipy.ndimage import gaussian_filter1d
from physion.analysis.episodes.build import EpisodeData
from physion.dataviz.episodes.trial_average import plot

for f in dataset['files']:

    data = Data(f)
    data.build_photodiode()
    data.build_MUA()

    ep = EpisodeData(data, 
                    prestim_duration=2, 
                    quantities=['MUA', 'photodiode'],
                    protocol_id=0)

    fig, AX = plot(ep, 
                   smoothing=40,
                   quantity='MUA',
                   with_screen_inset=True,
                   column_key='x-center',
                   row_key='y-center')
    fig.suptitle(f)
# %%
