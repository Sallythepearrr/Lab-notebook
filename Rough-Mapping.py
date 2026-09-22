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
        # os.path.join(os.path.expanduser('~'), 'DATA'),
        os.path.join(os.path.expanduser('~'), 'DATA', '2026_08_04'),
            for_protocol='rough-mapping')
            # )

# %%
pt.set_style('manuscript')
from scipy.ndimage import gaussian_filter1d
from physion.analysis.episodes.build import EpisodeData
from physion.dataviz.episodes.trial_average import plot

# %%
for f in dataset['files']:

    # load the data
    data = Data(f)
    data.build_MUA()
    data.build_spikes()
    data.build_LFP()

    # restructure the data into episodes of visual-stimulation
    ep = EpisodeData(data, 
                    prestim_duration=1, 
                    quantities=['MUA', 
                                'spikes', 
                                'LFP'],
                    protocol_id=0)

    fig, AX = plot(ep, 
                   smoothing=40,
                   quantity='MUA',
                   with_screen_inset=True,
                   column_key='x-center',
                   row_key='y-center')
    fig.suptitle(f)
    fig, AX = plot(ep, 
                   smoothing=40,
                   quantity='spikes',
                   with_screen_inset=True,
                   column_key='x-center',
                   row_key='y-center')
    fig.suptitle(f)

    fig, AX = plot(ep, smoothing=100,
                    with_std=False,
                    quantity='spikes',
                    with_screen_inset=True,
                    column_key='x-center',
                    row_key='y-center'
                    )
    fig.suptitle(f)

    fig, AX = plot(ep, 
                       smoothing=40,
                       quantity='LFP',
                       with_screen_inset=True,
                       column_key='x-center',
                       row_key='y-center')
    fig.suptitle(f)



# %%
# Single protocol
data = Data(dataset['files'][0])
ep = EpisodeData(data, 
                    prestim_duration=1, 
                    quantities=['MUA', 
                                'spikes', 
                                'LFP'],
                    protocol_id=0)

# %%

fig, AX = plot(ep, smoothing=50,
                    with_std=False,
                    quantity='spikes',
                    with_screen_inset=True,
                    column_key='x-center',
                    row_key='y-center'
                    )


fig, AX = plot(ep, smoothing=30,
                    with_std=False,
                    quantity='MUA',
                    with_screen_inset=True,
                    column_key='x-center',
                    row_key='y-center')
#%%
# Plot heatmap PSTH


# %%
# fig, AX = pt.figure(axes=(3,1))

# for i, f in enumerate(dataset['files']):

#     color = pt.tab10(i) # colormap

#     # load the data
#     data = Data(f)

#     # restructure the data into episodes of visual-stimulation
#     ep = EpisodeData(data, 
#                     prestim_duration=2, 
#                     quantities=['MUA', 
#                                 'spikes', 
#                                 'LFP'],
#                     protocol_id=0)

#     for ax, quantity in zip(AX, ['LFP', 'MUA', 'spikes']):
#         ax.plot(ep.t, getattr(ep, quantity).mean(axis=(0,1)), color=color)

#     pt.annotate(AX[-1], os.path.basename(f)+i*'\n', (1, 0), color=color)    


# for ax, title in zip(AX, ['LFP', 'MUA', 'spikes']):
#     ax.set_title(title)


from physion.dataviz.episodes.trial_average import plot as plot_ta

plot_ta(ep, quantity='spikes',
        column_key='x-center')