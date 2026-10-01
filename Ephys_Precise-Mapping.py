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

# dataset = scan_folder_for_NWBfiles(\
#         os.path.join(os.path.expanduser('~'), 'DATA', '2026_08_04'),
#             for_protocol='detailed-mapping')
            # )

dataset = scan_folder_for_NWBfiles(\
        os.path.join(
            os.path.expanduser('/'), 
            'mnt','data', 'NWB_npx'),
            # for_protocol='rough-mapping'
            for_protocols=['precise-mapping', 
                           'detailed-mapping',
                           'precise-mapping-FIX'])

for i, filename in enumerate(dataset['files']):
    data = Data(filename, metadata_only=True)
    print(i, filename, '->', data.protocols)


# %%
pt.set_style('manuscript')
from scipy.ndimage import gaussian_filter1d
from physion.analysis.episodes.build import EpisodeData
from physion.dataviz.episodes.trial_average import plot


for f in dataset['files']:

    # load the data
    data = Data(f)

    # restructure the data into episodes of visual-stimulation
    ep = EpisodeData(data, 
                    prestim_duration=1, 
                    quantities=[
                        'spikes', 
                        ],
                    protocol_id=0)

    fig, AX = plot(ep, smoothing=30,
                    with_std=False,
                    quantity='spikes',
                    with_screen_inset=True,
                    column_key='x-center',
                    row_key='y-center'
                    )
    fig.suptitle(f)



# %%
