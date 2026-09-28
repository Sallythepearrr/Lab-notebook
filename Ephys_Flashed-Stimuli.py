# %% [markdown]
# # Analysis of Flashed Stimuli


# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

sys.path += ['physion/src'] # add src code directory for physion

from scipy.ndimage import gaussian_filter1d
from physion.analysis.episodes.build import EpisodeData
from physion.dataviz.episodes.trial_average import plot

import physion.utils.plot_tools as pt
pt.set_style('Light')

from physion.analysis.read_NWB import Data,\
    scan_folder_for_NWBfiles

dataset = scan_folder_for_NWBfiles(\
        os.path.join(
            os.path.expanduser('/'), 
            'mnt','data', 'NWB_npx'),
            for_protocol='flashed-stimuli')


# %%
# Loop over all files

for f in dataset['files']:

    data = Data(f)
    ep = EpisodeData(data, 
                    prestim_duration=1, 
                    quantities=[
                        'spikes'],
                    protocol_id=0,
                    )

    fig, AX = plot(ep,
                    smoothing=20,
                    with_std=False,
                    quantity='spikes',
                    with_annotation=True,
                    # with_stat_test=True
                    )
    fig.suptitle(f)


# %%
# Single Protocol

import matplotlib.pyplot as plt

data = Data(dataset['files'][9])

ep = EpisodeData(data, 
                    prestim_duration=1, 
                    quantities=[ 'spikes'],
                    protocol_id=0)

# %%
# plot heatmap
cond = np.ones(ep.spikes.shape[0], dtype=bool)

psth = ep.get_response2D(
    quantity='spikes',
    episode_cond=cond,
    index=np.arange(ep.spikes.shape[1]),
    averaging_dimension='episodes'
)

# convert probability of spike / bin -> spikes/s = Hz
dt = np.median(np.diff(ep.t))
psth_hz = psth / dt

print(psth_hz.shape)
# (n_units, n_timepoints)

smooth_ms = 20

sigma_samples = (smooth_ms / 1000) / dt

psth_hz_smooth = gaussian_filter1d(
    psth_hz,
    sigma=sigma_samples,
    axis=1
)

fig, ax = plt.subplots(figsize=(8, 8))

im = ax.imshow(
    psth_hz_smooth,
    aspect='auto',
    interpolation='nearest',
    extent=[
        ep.t[0],
        ep.t[-1],
        psth_hz_smooth.shape[0] - 0.5,
        -0.5
    ]
)

ax.axvline(0, color='k', linestyle='--', linewidth=1)

ax.set_xlabel('Time from stimulus onset (s)')
ax.set_ylabel('Unit')
ax.set_title('Trial-averaged PSTH')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Firing rate (Hz)')

plt.show()

# %%
# subtract baseline
baseline = (-0.5, 0)

baseline_cond = (
    (ep.t >= baseline[0]) &
    (ep.t < baseline[1])
)

baseline_rate = psth_hz_smooth[:, baseline_cond].mean(
    axis=1,
    keepdims=True
)

delta_fr = psth_hz_smooth - baseline_rate


fig, ax = plt.subplots(figsize=(9, 9))

lim = np.percentile(np.abs(delta_fr), 99)

im = ax.imshow(
    delta_fr,
    aspect='auto',
    interpolation='nearest',
    extent=[
        ep.t[0],
        ep.t[-1],
        psth_hz_smooth.shape[0] - 0.5,
        -0.5
    ],
    cmap='RdBu_r',
    vmin=-lim,
    vmax=lim
)

ax.axvline(0, color='k', linestyle='--', linewidth=1)

ax.set_xlabel('Time from stimulus onset (s)')
ax.set_ylabel('Unit')
ax.set_title('Normalized Trial-averaged PSTH')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$\Delta$ firing rate (Hz)')

plt.show()

# %%
# statisical test of evoked responses:
stat_test_props = dict(interval_pre=[-0.5,0],                                   
                        interval_post=[0.,0.5],                                   
                        test='wilcoxon',                                            
                        sign='positive')

summary_evoked = ep.pre_post_statistics(\
                    stat_test_props=stat_test_props,
                    response_args=dict(quantity='spikes',
                                    ),
                    loop_over_cells=True,
                    response_significance_threshold=0.05,
                    multiple_comparison_correction=False,
                    verbose=True)

# for key in summary_evoked:
#         print('- %s : %s' % (key, summary_evoked[key]))

# Test Reliability
summary_rel = ep.reliability(\
                    stat_test_props=dict(
                          seed=1,
                          n_samples=500
                          ),
                    response_args=dict(quantity='spikes',
                                    ),
                    loop_over_cells=True,
                    response_significance_threshold=0.05,
                    multiple_comparison_correction=False,
                    verbose=True)

# for key in summary_rel:
#         print('- %s : %s' % (key, summary_rel[key]))


# Summary of responsive and reliable units
responsive_units = np.asarray(summary_evoked['significant'], dtype=bool)
reliable_units = np.asarray(summary_rel['significant'], dtype=bool)

n_responsive_units = np.sum(responsive_units)
n_reliable_units = np.sum(reliable_units)
n_total_unit = responsive_units.size
# percentage_significant = 100 * n_responsive_units / n_total_unit

print('Stimulus-evoked cells: %i/%i (%.1f%%)' % (
    np.sum(responsive_units),
    n_total_unit,
    100 * n_responsive_units / n_total_unit,
))

print('Reliable cells: %i/%i (%.1f%%)' % (
    np.sum(reliable_units),
    n_total_unit,
    100 * n_reliable_units / n_total_unit,
))

overlap = responsive_units & reliable_units
n_overlap_units = np.sum(overlap)

print('Overlaping cells: %i/%i (%.1f%%)' % (
    np.sum(overlap),
    n_total_unit,
    100 * n_overlap_units / n_total_unit,
))

# %%
# Spareness (fraction of activated neurons)

fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(
    [n_responsive_units, n_total_unit - n_responsive_units],
    labels=['Significant', 'Not significant'],
    autopct='%1.1f%%',
    startangle=90,
)
ax.set_title('Stimulus-evoked cells')
ax.axis('equal')
plt.show()






# %% [markdown]
# # Extra analysis


# %%
# Compare Reliable and Reliable

counts = {
    'Neither': np.sum(~responsive_units & ~reliable_units),
    'Responsive only': np.sum(responsive_units & ~reliable_units),
    'Reliable only': np.sum(~responsive_units & reliable_units),
    'Both': np.sum(responsive_units & reliable_units),
}

labels = list(counts.keys())
values = list(counts.values())

plt.figure(figsize=(6,4))
plt.bar(labels, values)
plt.ylabel('Number of units')
plt.title('Unit categories')
plt.xticks(rotation=20, ha='right')

for i, v in enumerate(values):
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()




# %%
# Activity levels for significant cells
post_cond = (
    (ep.t >= stat_test_props['interval_post'][0]) &
    (ep.t < stat_test_props['interval_post'][1])
)

baseline_rate_per_cell = psth_hz[:, baseline_cond].mean(axis=1)
evoked_rate_per_cell = psth_hz[:, post_cond].mean(axis=1)
delta_rate_per_cell = evoked_rate_per_cell - baseline_rate_per_cell

activated_indices = np.flatnonzero(significant_cells)
activated_activity = np.column_stack((
    activated_indices,
    baseline_rate_per_cell[activated_indices],
    evoked_rate_per_cell[activated_indices],
    delta_rate_per_cell[activated_indices],
))

print('\nActivity of stimulus-evoked cells:')
print('unit\tbaseline_Hz\tevoked_Hz\tdelta_Hz')
for unit, baseline_hz, evoked_hz, delta_hz in activated_activity:
    print('%i\t%.2f\t\t%.2f\t\t%.2f' % (
        unit, baseline_hz, evoked_hz, delta_hz,
    ))

if activated_indices.size > 0:
    order = activated_indices[np.argsort(
        delta_rate_per_cell[activated_indices]
    )]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        np.arange(order.size),
        delta_rate_per_cell[order],
        color='tab:red',
    )
    ax.set_xticks(np.arange(order.size))
    ax.set_xticklabels(order)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Evoked - baseline firing rate (Hz)')
    ax.set_title('Activity increase in stimulus-evoked neurons')
    plt.tight_layout()
    plt.show()


# %%
#Test

# unit_mean = ep.spikes.mean(axis=0)  # shape: (n_units, n_timepoints)

# fig, ax = pt.figure()

# # for unit_idx in range(unit_mean.shape[0]):
# for unit_idx in range(150,250):
#     ax.plot(ep.t, gaussian_filter1d(unit_mean[unit_idx],50)*100, label= unit_idx)

