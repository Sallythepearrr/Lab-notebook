# %% [markdown]
# # Analysis of Electrophysiology Data and Optogenetic Manipulation

# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

sys.path += ['./physion/src'] # add src code directory for physion
import physion.utils.plot_tools as pt

from physion.analysis.read_NWB import Data,\
    scan_folder_for_NWBfiles

dataset = scan_folder_for_NWBfiles(\
        os.path.join(os.path.expanduser('~'), 
            'DATA', '2026_07_31'),
            for_protocols=['detailed-mapping', 'precise-mapping'])
            # )

# %%
pt.set_style('dark')#'manuscript')
from scipy.ndimage import gaussian_filter1d
from physion.analysis.episodes.build import EpisodeData
from physion.dataviz.episodes.trial_average import plot

for f in dataset['files']:

    data = Data(f)
    data.build_MUA()
    data.build_LFP()
    data.build_spikes()

    ep = EpisodeData(data, 
                    prestim_duration=1, 
                    quantities=['MUA','spikes','LFP'],
                    protocol_id=0)

    fig, AX = plot(ep, 
                   smoothing=40,
                   quantity='MUA',
                   with_screen_inset=True,
                   column_key='x-center',
                   row_key='y-center')

    fig, AX = plot(ep, 
                       smoothing=40,
                       quantity='spikes',
                       with_screen_inset=True,
                       column_key='x-center',
                       row_key='y-center')

    fig, AX = plot(ep, smoothing=30,
                        quantity='LFP',
                        with_screen_inset=True,
                        column_key='x-center',
                        row_key='y-center'
                        )
    fig.suptitle(f)


# %%

fig, AX = plot(ep, smoothing=50,
                    with_std=False,
                    quantity='spikes',
                    with_screen_inset=True,
                    with_annotation=True,
                    column_key='x-center',
                    row_key='y-center',
                    # color_key='angle'
                    )

fig, AX = plot(ep, smoothing=50,
                    with_std=False,
                    quantity='LFP',
                    with_screen_inset=True,
                    with_annotation=True,
                    column_key='x-center',
                    row_key='y-center',
                    # color_key='angle'
                    )

# %%
fig, AX = plot(ep, smoothing=50,
                    with_std=False,
                    quantity='MUA',
                    with_screen_inset=True,
                    with_annotation=True,
                    column_key='x-center',
                    row_key='y-center',
                    # color_key='angle'
                    )
# %%
