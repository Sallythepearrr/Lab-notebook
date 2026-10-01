# %% [markdown]
# # Ephys Plasticity analysis
#
# Population firing-rate comparisons for the Plasticity protocol.
# Plot: population mean +/- SD across trials.
# Statistics: two-sided Mann-Whitney U on trial-wise evoked change
#             (mean post-stim FR - mean pre-stim FR).

# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

sys.path += ['physion/src'] # add src code directory for physion

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from physion.analysis.episodes.build import EpisodeData
from physion.dataviz.episodes.trial_average import plot

import physion.utils.plot_tools as pt
pt.set_style('Light')

from scipy.stats import mannwhitneyu

from physion.analysis.read_NWB import Data,\
    scan_folder_for_NWBfiles

FIGURE_FOLDER = '/home/pan.zhang/OneDrive/Lab-Notebook/4D_Project_Analysis/Figs/Plasticity'
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Current physion build_firing default is 10 ms; set explicitly for reproducibility.
FIRING_DT = 1e-2          # seconds = 10 ms
EPISODE_DT_MS = 10        # EpisodeData expects ms
PRESTIM_DURATION = 1.0    # seconds
SMOOTHING_SAMPLES = 5

BASELINE = (-0.5, 0.0)
POST = (0.0, 0.5)
N_EDGE_TRIALS = 30

BLUE = '#1f77b4'
ORANGE = '#ff7f0e'

# Requested dates
DAY1 = '2026-08-25'
DAY2 = '2026-08-26'


# %%
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

dataset = scan_folder_for_NWBfiles(\
        os.path.join(
            os.path.expanduser('/'), 
            'mnt','data', 'NWB_npx'),
            for_protocol='Plasticity')

for i, filename in enumerate(dataset['files']):
    print(i, filename)


# %%
# -----------------------------------------------------------------------------
# Load each session once and reduce to trial x time population firing traces
# -----------------------------------------------------------------------------
def load_session(filename):
    data = Data(filename)

    # Build firing explicitly: units x continuous-time, values already in Hz.
    data.build_firing(dt=FIRING_DT, verbose=False)

    ep = EpisodeData(
        data,
        protocol_name='Plasticity-Im1-mid-c0.7-#1',
        quantities=['firing'],
        prestim_duration=PRESTIM_DURATION,
        dt_sampling=EPISODE_DT_MS,
        interpolation='nearest',
        verbose=False,
    )

    # ep.firing: trials x units x time
    # Average units first -> one population firing trace per trial.
    trial_traces = ep.firing.mean(axis=1)

    start_time = data.nwbfile.session_start_time

    out = {
        'filename': filename,
        'start_time': start_time,
        'date': start_time.strftime('%Y-%m-%d'),
        'clock': start_time.strftime('%H:%M:%S'),
        'ep': ep,
        't': np.asarray(ep.t).copy(),
        'traces': np.asarray(trial_traces).copy(),
        'n_trials': trial_traces.shape[0],
        'n_units': ep.firing.shape[1],
    }

    data.close()
    return out


sessions = [load_session(f) for f in dataset['files']]
sessions.sort(key=lambda x: x['start_time'])

print('\nLoaded sessions:')
for i, s in enumerate(sessions):
    print(
        i,
        s['date'], s['clock'],
        '| trials =', s['n_trials'],
        '| units =', s['n_units'],
        '|', os.path.basename(s['filename'])
    )

sessions_by_day = {}
for s in sessions:
    sessions_by_day.setdefault(s['date'], []).append(s)

for day in sessions_by_day:
    sessions_by_day[day].sort(key=lambda x: x['start_time'])

if DAY1 not in sessions_by_day or DAY2 not in sessions_by_day:
    raise ValueError(
        f'Expected Plasticity sessions on {DAY1} and {DAY2}. '
        f'Found: {list(sessions_by_day.keys())}'
    )

if len(sessions_by_day[DAY1]) < 2:
    raise ValueError(f'Expected >=2 valid sessions on {DAY1}, found {len(sessions_by_day[DAY1])}')
if len(sessions_by_day[DAY2]) < 2:
    raise ValueError(f'Expected >=2 valid sessions on {DAY2}, found {len(sessions_by_day[DAY2])}')


# %%
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def interpolate_traces(traces, old_t, new_t):
    """Interpolate trial x time traces onto a common time axis if necessary."""
    old_t = np.asarray(old_t)
    new_t = np.asarray(new_t)

    if old_t.shape == new_t.shape and np.allclose(old_t, new_t):
        return traces

    return np.asarray([
        np.interp(new_t, old_t, trace)
        for trace in traces
    ])


def evoked_change_per_trial(traces, t, baseline=BASELINE, post=POST):
    """Return post - pre firing rate (Hz) for every trial."""
    t = np.asarray(t)
    pre_cond = (t >= baseline[0]) & (t < baseline[1])
    post_cond = (t >= post[0]) & (t < post[1])

    if not np.any(pre_cond) or not np.any(post_cond):
        raise ValueError('Baseline or post-stimulus interval does not overlap ep.t')

    pre = traces[:, pre_cond].mean(axis=1)
    post_fr = traces[:, post_cond].mean(axis=1)
    return post_fr - pre


def compare_groups(traces_a, traces_b, t):
    """Independent comparison of trial-wise evoked changes."""
    delta_a = evoked_change_per_trial(traces_a, t)
    delta_b = evoked_change_per_trial(traces_b, t)

    result = mannwhitneyu(delta_a, delta_b, alternative='two-sided')

    return {
        'U': result.statistic,
        'p': result.pvalue,
        'delta_a_mean': np.mean(delta_a),
        'delta_b_mean': np.mean(delta_b),
        'difference': np.mean(delta_b) - np.mean(delta_a),
        'n_a': len(delta_a),
        'n_b': len(delta_b),
    }


def format_p(p):
    if p < 1e-4:
        return f'{p:.2e}'
    return f'{p:.4f}'


def plot_comparison(
    ax,
    t,
    traces_a,
    traces_b,
    label_a,
    label_b,
    title,
    stim_duration=None,
):
    """
    Plot unsmoothed population mean across trials for two independent groups.

    traces_a / traces_b: trial x time, already averaged across units.
    Statistics are calculated on the same unsmoothed trial traces.
    """
    mean_a = traces_a.mean(axis=0)
    mean_b = traces_b.mean(axis=0)

    ax.plot(t, mean_a, color=BLUE, lw=1.8, label=f'{label_a} (n={len(traces_a)})')
    ax.plot(t, mean_b, color=ORANGE, lw=1.8, label=f'{label_b} (n={len(traces_b)})')

    ax.axvline(0, color='black', linestyle='--', lw=0.8)
    if stim_duration is not None:
        ax.axvspan(0, stim_duration, color='grey', alpha=0.20, linewidth=0)

    stats = compare_groups(traces_a, traces_b, t)

    ax.set_title(
        title + '\n' +
        r'$\Delta$(orange-blue) = %.2f Hz, Mann-Whitney p=%s' %
        (stats['difference'], format_p(stats['p'])),
        fontsize=9,
    )
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Population firing rate (Hz)')
    ax.legend(frameon=False, fontsize=8)

    return stats


def session_traces_on_t(session, t_ref):
    return interpolate_traces(session['traces'], session['t'], t_ref)


def pool_sessions(session_list, t_ref):
    """Pool all trial-level population traces from a list of sessions."""
    return np.concatenate([
        session_traces_on_t(s, t_ref)
        for s in session_list
    ], axis=0)

def save_figure(fig, name, dpi=300):
    """Save a figure to FIGURE_FOLDER using a filesystem-safe name."""
    safe_name = ''.join(
        c if c.isalnum() or c in ('-', '_') else '_'
        for c in name.strip()
    )
    safe_name = '_'.join(filter(None, safe_name.split('_')))
    path = os.path.join(FIGURE_FOLDER, safe_name + '.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print('Saved:', path)
    return path

# Use a single common time axis for all cross-session plots.
t_ref = sessions[0]['t']



# %% [markdown]
# ## 1. Within each session: first 30 vs last 30 trials
#
# One figure per session.
# Blue = first 30 trials, orange = last 30 trials, shaded area = +/- SD.

# %%
within_session_stats = []

for day in [DAY1, DAY2]:
    day_sessions = sessions_by_day[day]

    for session_idx, s in enumerate(day_sessions, start=1):
        ep = s['ep']
        n_trials = ep.firing.shape[0]

        if n_trials < 2 * N_EDGE_TRIALS:
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            ax.text(
                0.5, 0.5,
                f'Only {n_trials} trials\nneed >= {2*N_EDGE_TRIALS}',
                ha='center', va='center', transform=ax.transAxes,
            )
            ax.set_title(
                f'Plasticity {day} - Session {session_idx} ({s["clock"]})'
            )
            ax.set_axis_off()
            fig.tight_layout()
            plt.show()
            continue

        # First 30 and last 30 trial-level population traces.
        # Use the same plotting/statistics helper as Parts 2 and 3.
        traces = np.asarray(s['traces'])
        first = traces[:N_EDGE_TRIALS]
        last = traces[-N_EDGE_TRIALS:]

        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        stats = plot_comparison(
            ax,
            s['t'],
            first,
            last,
            'First 30 trials',
            'Last 30 trials',
            f'Plasticity {day} - Session {session_idx} ({s["clock"]})',
            stim_duration=float(np.mean(ep.time_duration)),
        )
        within_session_stats.append((s, stats))

        fig.tight_layout()

        save_figure(
            fig,
            f'{day}_Session_{session_idx}_First30_vs_Last30'
        )

        plt.show()


print('\n=== Within-session statistics ===')
for s, st in within_session_stats:
    session_no = sessions_by_day[s['date']].index(s) + 1
    print(
        f"{s['date']} session {session_no}: "
        f"first30 mean delta={st['delta_a_mean']:.3f} Hz, "
        f"last30 mean delta={st['delta_b_mean']:.3f} Hz, "
        f"difference={st['difference']:.3f} Hz, "
        f"U={st['U']:.1f}, p={st['p']:.6g}"
    )


# %% [markdown]
# ## 2. Within each day: first session vs last session

# %%

within_day_stats = {}

for day in [DAY1, DAY2]:
    day_sessions = sessions_by_day[day]
    first_session = day_sessions[0]
    last_session = day_sessions[-1]

    first_traces = session_traces_on_t(first_session, t_ref)
    last_traces = session_traces_on_t(last_session, t_ref)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    stats = plot_comparison(
        ax,
        t_ref,
        first_traces,
        last_traces,
        f'Session 1 ({first_session["clock"]})',
        f'Session {len(day_sessions)} ({last_session["clock"]})',
        f'Plasticity {day}: first vs last session',
        stim_duration=float(np.mean(first_session['ep'].time_duration)),
    )
    within_day_stats[day] = stats

    fig.tight_layout()
    save_figure(
        fig,
        f'{day}_Session1_vs_Session{len(day_sessions)}'
    )
    plt.show()


print('\n=== Within-day statistics ===')
for day, st in within_day_stats.items():
    print(
        f"{day}: first-session delta={st['delta_a_mean']:.3f} Hz, "
        f"last-session delta={st['delta_b_mean']:.3f} Hz, "
        f"difference={st['difference']:.3f} Hz, "
        f"U={st['U']:.1f}, p={st['p']:.6g}"
    )

# %% [markdown]
# ## 3a. Across days: last valid session on 2026-08-25 vs first session on 2026-08-26

# %%
s_0825_last = sessions_by_day[DAY1][-1]
s_0826_1 = sessions_by_day[DAY2][0]

traces_0825_last = session_traces_on_t(s_0825_last, t_ref)
traces_0826_1 = session_traces_on_t(s_0826_1, t_ref)

fig, ax = plt.subplots(figsize=(7.5, 5))
stats_transition = plot_comparison(
    ax,
    t_ref,
    traces_0825_last,
    traces_0826_1,
    f'{DAY1} session {len(sessions_by_day[DAY1])}',
    f'{DAY2} session 1',
    'Across-day transition',
    stim_duration=float(np.mean(s_0825_last['ep'].time_duration)),
)
fig.tight_layout()
save_figure(
    fig,
    f'{DAY1}_LastSession_vs_{DAY2}_Session1_raw'
)
plt.show()

print('\n=== Across-day transition ===')
print(
    f"{DAY1} session {len(sessions_by_day[DAY1])} delta={stats_transition['delta_a_mean']:.3f} Hz, "
    f"{DAY2} session 1 delta={stats_transition['delta_b_mean']:.3f} Hz, "
    f"difference={stats_transition['difference']:.3f} Hz, "
    f"U={stats_transition['U']:.1f}, p={stats_transition['p']:.6g}"
)

# %%
# Normalised
def normalize_to_baseline(traces, t, baseline=BASELINE, baseline_floor_hz=0.1):
    """
    Normalize each trial to its own prestimulus baseline:

        (FR(t) - FR_baseline) / FR_baseline

    The mean prestimulus level of each trial is therefore approximately 0.
    A small floor prevents division by zero for very low-baseline trials.
    """
    traces = np.asarray(traces, dtype=float)
    t = np.asarray(t)

    baseline_cond = (t >= baseline[0]) & (t < baseline[1])
    if not np.any(baseline_cond):
        raise ValueError('Baseline interval does not overlap t')

    baseline_fr = traces[:, baseline_cond].mean(axis=1, keepdims=True)
    baseline_safe = np.maximum(baseline_fr, baseline_floor_hz)
    return (traces - baseline_fr) / baseline_safe

def normalized_evoked_per_trial(traces, t, baseline=BASELINE, post=POST):
    """Evoked response after per-trial baseline normalization."""
    norm = normalize_to_baseline(traces, t, baseline=baseline)
    post_cond = (t >= post[0]) & (t < post[1])
    if not np.any(post_cond):
        raise ValueError('Post-stimulus interval does not overlap t')
    return norm[:, post_cond].mean(axis=1)

def plot_normalized_comparison(
    ax,
    t,
    traces_a,
    traces_b,
    label_a,
    label_b,
    title,
    stim_duration=None,
):
    """Plot two groups after per-trial baseline normalization (prestim = 0)."""
    norm_a = normalize_to_baseline(traces_a, t)
    norm_b = normalize_to_baseline(traces_b, t)

    mean_a = norm_a.mean(axis=0)
    mean_b = norm_b.mean(axis=0)

    ax.plot(t, mean_a, color=BLUE, lw=1.8, label=f'{label_a} (n={len(norm_a)})')
    ax.plot(t, mean_b, color=ORANGE, lw=1.8, label=f'{label_b} (n={len(norm_b)})')

    ax.axhline(0, color='black', lw=0.7, alpha=0.6)
    ax.axvline(0, color='black', linestyle='--', lw=0.8)
    if stim_duration is not None:
        ax.axvspan(0, stim_duration, color='grey', alpha=0.20, linewidth=0)

    evoked_a = normalized_evoked_per_trial(traces_a, t)
    evoked_b = normalized_evoked_per_trial(traces_b, t)
    result = mannwhitneyu(evoked_a, evoked_b, alternative='two-sided')
    difference = np.mean(evoked_b) - np.mean(evoked_a)

    ax.set_title(
        title + '\n' +
        r'Normalized $\Delta$(orange-blue) = %.3f, Mann-Whitney p=%s' %
        (difference, format_p(result.pvalue)),
        fontsize=9,
    )
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Relative firing-rate change from baseline')
    ax.legend(frameon=False, fontsize=8)

    return {
        'U': result.statistic,
        'p': result.pvalue,
        'evoked_a_mean': np.mean(evoked_a),
        'evoked_b_mean': np.mean(evoked_b),
        'difference': difference,
        'n_a': len(evoked_a),
        'n_b': len(evoked_b),
    }

# Normalized companion figure: prestimulus baseline = 0
fig, ax = plt.subplots(figsize=(7.5, 5))
stats_transition_norm = plot_normalized_comparison(
    ax,
    t_ref,
    traces_0825_last,
    traces_0826_1,
    f'{DAY1} session {len(sessions_by_day[DAY1])}',
    f'{DAY2} session 1',
    'Across-day transition — baseline normalized',
    stim_duration=float(np.mean(s_0825_last['ep'].time_duration)),
)
fig.tight_layout()
save_figure(
    fig,
    f'{DAY1}_LastSession_vs_{DAY2}_Session1_baseline_normalized'
)
plt.show()

print('\n=== Across-day transition: baseline normalized ===')
print(
    f"{DAY1} session {len(sessions_by_day[DAY1])} normalized evoked={stats_transition_norm['evoked_a_mean']:.3f}, "
    f"{DAY2} session 1 normalized evoked={stats_transition_norm['evoked_b_mean']:.3f}, "
    f"difference={stats_transition_norm['difference']:.3f}, "
    f"U={stats_transition_norm['U']:.1f}, p={stats_transition_norm['p']:.6g}"
)
# %% [markdown]
# ## 3b. Across days: all 2026-08-25 trials vs all 2026-08-26 trials

# %%
day1_traces = pool_sessions(sessions_by_day[DAY1], t_ref)
day2_traces = pool_sessions(sessions_by_day[DAY2], t_ref)

fig, ax = plt.subplots(figsize=(7.5, 5))
stats_days = plot_comparison(
    ax,
    t_ref,
    day1_traces,
    day2_traces,
    DAY1,
    DAY2,
    'Across-day pooled comparison',
    stim_duration=float(np.mean(sessions_by_day[DAY1][0]['ep'].time_duration)),
)
fig.tight_layout()
save_figure(
    fig,
    f'{DAY1}_vs_{DAY2}_pooled_raw'
)
plt.show()

print('\n=== Across-day pooled statistics ===')
print(
    f"{DAY1} delta={stats_days['delta_a_mean']:.3f} Hz, "
    f"{DAY2} delta={stats_days['delta_b_mean']:.3f} Hz, "
    f"difference={stats_days['difference']:.3f} Hz, "
    f"U={stats_days['U']:.1f}, p={stats_days['p']:.6g}"
)

# Normalised
# Normalized companion figure: prestimulus baseline = 0
fig, ax = plt.subplots(figsize=(7.5, 5))
stats_days_norm = plot_normalized_comparison(
    ax,
    t_ref,
    day1_traces,
    day2_traces,
    DAY1,
    DAY2,
    'Across-day pooled comparison — baseline normalized',
    stim_duration=float(np.mean(sessions_by_day[DAY1][0]['ep'].time_duration)),
)
fig.tight_layout()
save_figure(
    fig,
    f'{DAY1}_vs_{DAY2}_pooled_baseline_normalized'
)
plt.show()

print('\n=== Across-day pooled statistics: baseline normalized ===')
print(
    f"{DAY1} normalized evoked={stats_days_norm['evoked_a_mean']:.3f}, "
    f"{DAY2} normalized evoked={stats_days_norm['evoked_b_mean']:.3f}, "
    f"difference={stats_days_norm['difference']:.3f}, "
    f"U={stats_days_norm['U']:.1f}, p={stats_days_norm['p']:.6g}"
)

# %%
# Notes
# -----
# 1. The plotted mean is the population firing rate: units are averaged within
#    each trial, then trials are averaged. No temporal smoothing or SD shading is applied.
# 2. Statistics use the same unsmoothed trial-level evoked changes:
#       mean firing in POST - mean firing in BASELINE.
# 3. Mann-Whitney U is used because the compared trial groups are independent,
#    not paired observations.
# 4. The final day-vs-day test pools trials across sessions. This is useful as an
#    exploratory trial-level comparison, but trials are nested within sessions.
#    With only 2 sessions vs 2 sessions, session-level inference is underpowered;
#    do not interpret the pooled p-value as a fully hierarchical day-level test.
# %%
