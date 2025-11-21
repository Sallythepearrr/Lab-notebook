# %% [markdown]
# # Trial Averaging

# %%
import os, sys
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
                                     verbose=False)

# %% [markdown]
# ## Build episodes (stimulus-aligned)

# %%

# find protocol of protocol-8
p_name = [p for p in data.protocols if 'protocol-16' in p][0]
episodes = physion.analysis.process_NWB.EpisodeData(data, 
                                                    quantities=['dFoF',
                                                                'pupil_diameter'],
                                                    protocol_name=p_name)

# %% [markdown]
# ## Plot properties

# %%
# plot over varying angles
plot_props = dict(column_key='Image-ID',
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
                                                      roiIndex=range(data.nROIs),
                                                      **plot_props)

# %% [markdown]
# ## Single ROIs 

# %%
fig, AX = physion.dataviz.episodes.trial_average.plot(episodes,
                                                      roiIndex=0,
                                                      **plot_props)

# %%
fig, AX = physion.dataviz.episodes.trial_average.plot(episodes,
                                                      roiIndex=2,
                                                      **plot_props)

# %%
