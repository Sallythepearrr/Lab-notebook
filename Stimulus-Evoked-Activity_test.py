# %% [markdown]
# # Trial Averaging

# %%
import os, sys
import numpy as np
sys.path += ['physion/src'] # add src code directory for physion
import physion
import physion.utils.plot_tools as pt
pt.set_style('dark')


# %% [markdown]
# ## Load data

# %%
# load a datafile
filename = os.path.join(os.path.expanduser('~'), 'DATA', 
                        'Sally', 'PYR_WT_V1-demo-2P-2025', 'NWBs',
                        '2025_11_13-14-27-23.nwb')

data = physion.analysis.read_NWB.Data(filename,
                                     verbose=True)
# data.build_dFoF(method_for_F0='sliding_percentile',
#                 sliding_window=300.,
#                 percentile=5.,
#                 neuropil_correction_factor=0.8,
#                 verbose=True)
data.build_dFoF()
staticmethod
# %% [markdown]
# ## Build episodes (stimulus-aligned)

# %%

# find protocol of protocol-8
p_name = [p for p in data.protocols if 'protocol-15' in p][0]
episodes = physion.analysis.process_NWB.EpisodeData(data, 
                                                    quantities=['dFoF',
                                                                'pupil_diameter'],
                                                    protocol_name=p_name)
stat_test_props =  dict(interval_pre=[-2,0], 
                        interval_post=[0,2],
                        test = 'wilcoxon')

significants = []
for i in range(data.nROIs):
    summary = episodes.compute_summary_data(stat_test_props,
                                            response_significance_threshold=0.001,
                                            response_args=dict(\
                                                roiIndex=i))
    if np.sum(summary['significant']):
        significants.append(i)
# %% [markdown]
# ## Plot properties

# %%
# plot over varying angles
plot_props = dict(column_key='Image-ID',
                  #color_key='repeat',
                  with_annotation=True,
                  figsize=(9,1.8))

# %% [markdown]
# ## Pupil variations

# %%
fig, AX = physion.dataviz.episodes.trial_average.plot(episodes,
                                                      quantity='pupil_diameter',
                                                    #   with_std=False,
                                                      **plot_props)
# %% [markdown]
# ## Average over all ROIs 

# %%
fig, AX = physion.dataviz.episodes.trial_average.plot(episodes,
                                                      quantity='dFoF',
                                                      #roiIndex=range(data.nROIs),
                                                      **plot_props)

# %% [markdown]
# ## Single ROIs 

# %%
for i in significants:
  fig, AX = physion.dataviz.episodes.trial_average.plot(episodes,
                                                        roiIndex=i,
                                                        **plot_props)

# %%
from scipy import stats
for roi in significants:
  cond = getattr(episodes, 'Image-ID')==0
  dFoF = episodes.dFoF[cond, roi, :]
  fig, ax = pt.figure(ax_scale=(2,2))
  #for trial in range(dFoF.shape[0])[:2]:
  #    pt.plot(episodes.t, dFoF[trial, :], color='gray', alpha=0.3)
  pt.plot(episodes.t, dFoF.mean(axis=0), 
          sy=stats.sem(dFoF, axis=0),
          color='k', ax=ax)
  pt.set_plot(ax, xlabel='time from start (s)', ylabel='dFoF',
              title=f'ROI {roi} - Image-ID 0 - n={dFoF.shape[0]} trials')
# %%
fig, AX = physion.dataviz.episodes.trial_average.plot(episodes,
                                                      roiIndex=2,
                                                      **plot_props)

# %%
