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
#             for_protocol='rough-mapping')
            # )

dataset = scan_folder_for_NWBfiles(\
        os.path.join(
            os.path.expanduser('/'), 
            'mnt','data', 'NWB_npx'),
            for_protocol='rough-mapping')

for i, filename in enumerate(dataset['files']):
    print(i, filename, '->', dataset['protocols'][i])

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
# Single protocol
data = Data(dataset['files'][0])
print('\nLoading:', data)

ep = EpisodeData(data, 
                    prestim_duration=1, 
                    quantities=[ 'spikes'],
                    protocol_id=0)

print('ep.spikes shape:', ep.spikes.shape)  # trials, units, time


# %% [Markdown]
# # Heatmap for each location

# %%
import matplotlib.pyplot as plt

# Shared PSTH settings
n_trials, n_units, _ = ep.spikes.shape
dt = np.median(np.diff(ep.t))

SMOOTH_MS = 20
sigma_samples = (SMOOTH_MS / 1000.0) / dt

BASELINE = (-0.5, 0.0)
baseline_cond = (ep.t >= BASELINE[0]) & (ep.t < BASELINE[1])


def make_psth(episode_cond):
    """Return smoothed trial-averaged firing rate and baseline-subtracted rate.

    Shapes returned: (n_units, n_timepoints)
    """
    psth = ep.get_response2D(
        quantity='spikes',
        episode_cond=episode_cond,
        index=np.arange(n_units),
        averaging_dimension='episodes',
    )

    # ep.spikes is a binary spike train sampled at dt, so convert to Hz.
    psth_hz = psth / dt

    psth_hz_smooth = gaussian_filter1d(
        psth_hz,
        sigma=sigma_samples,
        axis=1,
    )

    baseline_rate = psth_hz_smooth[:, baseline_cond].mean(
        axis=1,
        keepdims=True,
    )
    delta_hz = psth_hz_smooth - baseline_rate

    return psth_hz_smooth, delta_hz


# Sort x left->right and y top->bottom so the subplot arrangement resembles
# the spatial arrangement of the rough-mapping stimuli.
x_values = np.sort(np.asarray(ep.varied_parameters['x-center']))
y_values = np.sort(np.asarray(ep.varied_parameters['y-center']))[::-1]

rate_heatmaps = {}
delta_heatmaps = {}
trial_counts = {}

for iy, y_value in enumerate(y_values):
    for ix, x_value in enumerate(x_values):
        location_cond = (
            (getattr(ep, 'x-center') == x_value) &
            (getattr(ep, 'y-center') == y_value)
        )

        n_location_trials = int(location_cond.sum())
        trial_counts[(iy, ix)] = n_location_trials

        if n_location_trials == 0:
            continue

        rate_heatmaps[(iy, ix)], delta_heatmaps[(iy, ix)] = make_psth(
            location_cond
        )

        print(
            'x=%s, y=%s: %i trials' %
            (x_value, y_value, n_location_trials)
        )


def plot_location_heatmaps(
    heatmaps,
    title,
    colorbar_label,
    cmap,
    symmetric=False,
):
    """Plot one unit x time heatmap per visual-field location."""

    fig, axes = plt.subplots(
        len(y_values),
        len(x_values),
        figsize=(3.1 * len(x_values), 2.6 * len(y_values)),
        # figsize=(4 * len(x_values), 3 * len(y_values)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    available = list(heatmaps.values())

    # IMPORTANT: use one common color scale across all locations.
    # Otherwise each subplot autoscales independently and the common
    # colorbar would be misleading.
    if available:
        flattened = np.concatenate([values.ravel() for values in available])

        if symmetric:
            limit = max(
                np.percentile(np.abs(flattened), 99),
                np.finfo(float).eps,
            )
            vmin, vmax = -limit, limit
        else:
            vmin = 0
            vmax = max(np.percentile(flattened, 99), np.finfo(float).eps)
    else:
        vmin, vmax = None, None

    image = None

    for iy, y_value in enumerate(y_values):
        for ix, x_value in enumerate(x_values):
            ax = axes[iy, ix]
            values = heatmaps.get((iy, ix))
            n_here = trial_counts.get((iy, ix), 0)

            if values is not None:
                image = ax.imshow(
                    values,
                    aspect='auto',
                    interpolation='nearest',
                    extent=[ep.t[0], ep.t[-1], n_units - 0.5, -0.5],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
            else:
                ax.text(
                    0.5, 0.5, 'no trials',
                    transform=ax.transAxes,
                    ha='center', va='center',
                )

            ax.set_title('x=%s, y=%s (n=%i)' % (x_value, y_value, n_here))

            if iy == len(y_values) - 1:
                ax.set_xlabel('Time (s)')
            if ix == 0:
                ax.set_ylabel('Unit')

    # Reserve space on the far right for a dedicated colorbar axis.
    # This prevents the colorbar from overlapping the last heatmap column.
    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.10,
        top=0.90,
        wspace=0.15,
        hspace=0.30,
    )

    if image is not None:
        cbar_ax = fig.add_axes([0.91, 0.12, 0.015, 0.74])
        fig.colorbar(
            image,
            cax=cbar_ax,
            label=colorbar_label,
        )

    fig.suptitle(title, y=0.97)
    return fig, axes


# %%
plot_location_heatmaps(
    rate_heatmaps,
    'Rough mapping: PSTH by stimulus location',
    'Firing rate (Hz)',
    'viridis',
)
plt.show()


# %%
# Baseline subtraction
plot_location_heatmaps(
    delta_heatmaps,
    'Rough mapping: baseline-subtracted PSTH by stimulus location',
    r'$\Delta$ firing rate (Hz)',
    'RdBu_r',
    symmetric=True,
)
plt.show()



# %%
# location-specific responsive/reliable-unit statistics

RUN_CELL_STATS = True

if RUN_CELL_STATS:
    minimum_trials = 5

    stat_test_props = dict(
        interval_pre=[-0.5, 0],
        interval_post=[0, 0.5],
        test='wilcoxon',
        sign='positive',
    )

    # Same setting as your original code.
    # For a quicker test run, change n_samples=500 to 100.
    reliability_props = dict(seed=1, n_samples=500)

    # x-center and y-center are already fixed explicitly by location_cond,
    # so treat all varied parameters as repetition keys to avoid splitting
    # the selected location into further stimulus configurations.
    repetition_keys = list(ep.varied_parameters.keys())

    location_results = {}

    for iy, y_value in enumerate(y_values):
        for ix, x_value in enumerate(x_values):
            location_cond = (
                (getattr(ep, 'x-center') == x_value) &
                (getattr(ep, 'y-center') == y_value)
            )

            n_location_trials = int(location_cond.sum())
            location_key = (iy, ix)

            result = {'n_trials': n_location_trials}

            if n_location_trials >= minimum_trials:
                summary_evoked = ep.pre_post_statistics(
                    episode_cond=location_cond,
                    stat_test_props=stat_test_props,
                    response_args=dict(quantity='spikes'),
                    loop_over_cells=True,
                    response_significance_threshold=0.05,
                    multiple_comparison_correction=False,
                    repetition_keys=repetition_keys,
                    nMin_episodes=minimum_trials,
                    verbose=False,
                )

                summary_rel = ep.reliability(
                    episode_cond=location_cond,
                    stat_test_props=reliability_props,
                    response_args=dict(quantity='spikes'),
                    loop_over_cells=True,
                    response_significance_threshold=0.05,
                    multiple_comparison_correction=False,
                    repetition_keys=repetition_keys,
                    nMin_episodes=minimum_trials,
                    verbose=False,
                )

                if (
                    'significant' in summary_evoked and
                    'significant' in summary_rel
                ):
                    responsive = np.asarray(
                        summary_evoked['significant'],
                        dtype=bool,
                    ).reshape(n_units, -1).any(axis=1)

                    reliable = np.asarray(
                        summary_rel['significant'],
                        dtype=bool,
                    ).reshape(n_units, -1).any(axis=1)

                    result.update(
                        responsive=responsive,
                        reliable=reliable,
                        overlap=responsive & reliable,
                    )
                else:
                    result.update(
                        responsive=None,
                        reliable=None,
                        overlap=None,
                    )
            else:
                result.update(
                    responsive=None,
                    reliable=None,
                    overlap=None,
                )

            location_results[location_key] = result

            # Same print format as your original code.
            print(
                'Location x=%s, y=%s: %i trials' %
                (x_value, y_value, n_location_trials),
                end='',
            )

            if result['responsive'] is None:
                print(' (insufficient trials for cell statistics)')
            else:
                print(
                    '; responsive %i/%i, reliable %i/%i, both %i/%i' %
                    (
                        result['responsive'].sum(), n_units,
                        result['reliable'].sum(), n_units,
                        result['overlap'].sum(), n_units,
                    )
                )
# %%
