# %% [markdown]
# # Analysis of Electrophysiology Data and Optogenetic Manipulation

# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

sys.path += ['physion/src'] # add src code directory for physion
import physion.utils.plot_tools as pt
pt.set_style('dark')

from physion.analysis.read_NWB import Data,\
    scan_folder_for_NWBfiles

dataset = scan_folder_for_NWBfiles(\
        os.path.join(os.path.expanduser('/'), 
            'mnt','data','2026_08_25'),
            for_protocol='flashed-stimuli')


# %%
# dataset = scan_folder_for_NWBfiles(\
#         os.path.join(
#             os.path.expanduser('/'), 
#             'mnt','data', 'NWB_npx'),
#             for_protocol='flashed-stimuli')


# %%

from scipy.ndimage import gaussian_filter1d
from physion.analysis.episodes.build import EpisodeData
from physion.dataviz.episodes.trial_average import plot

for f in dataset['files']:

    data = Data(f)
    data.build_photodiode()
    data.build_MUA()

    ep = EpisodeData(data, 
                    prestim_duration=1, 
                    quantities=[
                        'MUA', 
                        'photodiode',
                        'spikes',
                        'LFP'],
                    protocol_id=0)

    fig, AX = plot(ep, 
                   smoothing=40,
                   quantity='MUA')
    fig.suptitle(f)

    
    fig, AX = plot(ep,
                    smoothing=50,
                    quantity='spikes')
    fig.suptitle(f)

    fig, AX = plot(ep,
                    smoothing=50,
                    with_std=False,
                    quantity='spikes')
    fig.suptitle(f)

    fig, AX = plot(ep, 
                       smoothing=40,
                       quantity='LFP')
    fig.suptitle(f)

# %%
fig, AX = pt.figure(ax_scale=(1,1), reshape_axes=False)
fig, AX = plot(ep, fig=fig, AX=AX,
                    smoothing=30,
                    with_std=False,
                    quantity='spikes')







# %%
#Test

# %%
from physion.analysis.episodes.build import EpisodeData
from scipy.ndimage import gaussian_filter1d
for f in dataset['files']:
    data = Data(f)
    data.build_spikes()
    data.build_MUA()
    data.build_LFP()

    ep = EpisodeData(data, 
                        prestim_duration=1, 
                        quantities=[
                            'MUA', 
                            'spikes',
                            'LFP'],
                        protocol_id=0)
# %%
fig, ax = pt.figure()

# Plot visually evoked activity across trial across units
ax.plot(ep.t, gaussian_filter1d(ep.spikes.mean(axis=(0,1)), 100))

# %%

unit_mean = ep.spikes.mean(axis=0)  # shape: (n_units, n_timepoints)

fig, ax = pt.figure()

for unit_idx in range(unit_mean.shape[0]):
# for unit_idx in range(75,140):
    fig, ax = pt.figure()
    ax.plot(ep.t, gaussian_filter1d(unit_mean[unit_idx],100), label= f"Unit{unit_idx}")

# %%
# Plot heatmap PSTH
dt=1
bin_ms = 50 #set bin size as 50ms
samples_per_bin = bin_ms//dt  # because dt = 1 ms

psth = (ep.spikes[:, :, :(ep.spikes.shape[2] // samples_per_bin) * samples_per_bin]
    .reshape(ep.spikes.shape[0], ep.spikes.shape[1], -1, samples_per_bin)
    .sum(axis=-1) #number of spikes in each 50ms window
    .mean(axis=0) ) / (bin_ms / 1000) # convert to Hz

#reshape to [n_ep, n_unit, n_bin, bins_per_psth_bin]


t_psth = ep.t[::samples_per_bin][:-1] # ep.t[:n_bins * bins_per_psth_bin]
# t_psth = t_psth.reshape(n_bins, bins_per_psth_bin).mean(axis=1)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))

baseline_cond = t_psth<0
plt.imshow(
    np.transpose(psth.T/(
                        1+
                        psth[:,baseline_cond].mean(axis=1))),
    # psth,
    aspect='auto',
    origin='lower',
    #extent=[t_psth[0], t_psth[-1], 0, 80]
)

plt.colorbar(label='Firing rate (Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Unit')
plt.title('50 ms PSTH')
plt.show()

# %%
