# %% [markdown]
# # Plasticity of Sparse Temporal Sequences

# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

sys.path += ['physion/src'] # add src code directory for physion
from physion.utils import plot_tools as pt
from physion.dataviz.raw import plot as plot_raw

from physion.analysis.read_NWB import Data,\
    scan_folder_for_NWBfiles

from physion.analysis.episodes.build import EpisodeData

pt.set_style('dark')

# %%
folder = os.path.join(os.path.expanduser('/'), 
                      'mnt', 'data', 'NWB_npx')

notebook_folder =\
    os.path.join(os.path.expanduser('~'), 

        'OneDrive', 'Lab-Notebook', 'Data','Ephys')

if not os.path.isdir(os.path.join(folder, 'temp')):
    os.mkdir(os.path.join(folder, 'temp'))

# %%
dataset = scan_folder_for_NWBfiles(\
        os.path.join(
            os.path.expanduser('/'), 
            'mnt','data', 'NWB_npx'),
            for_protocol='Plasticity')

# %%
# doing an analysis over different viruses
# sorted = {}
# for v, virus in enumerate(\
#         np.unique(dataset['viruses'])):

#     print(' - %s:' % virus)
#     virus_cond = (virus==dataset['viruses'])

#     sorted[virus] = {}
#     for s, subject in enumerate(\
#             np.unique(dataset['subjects'][virus_cond])):

#         sorted[virus][subject] = []
#         subject_cond = dataset['subjects'][virus_cond]==subject

#         subject_files = [os.path.basename(f) for f in \
#             dataset['files'][virus_cond][subject_cond]]

#         print('     - %s:' % subject)
#         for s in subject_files:
#             print('         - [%s](../../Data/%s.md)' %\
#                 (s.replace('.nwb',''), s.replace('.nwb','')))
#         for s in dataset['files'][virus_cond][subject_cond]:
#             sorted[virus][subject].append(s)
#     print('\n')

# %% [markdown]
# # Analysis Per Session

# %%

def single_rec(filename,
               redo_figs=True):

    print('running %s' % filename)
    fn = os.path.basename(filename).replace('.nwb', '')

    md = os.path.join(notebook_folder, fn+'.md')

    text = '## Recording \n\n'
    text += '-  %s \n ' % fn
    data = Data(filename)
    if data.has_visual_stim():
        data.build_visual_stim() # real recording (possibly stopped)
        nReal = len(data.visual_stim.experiment['time_start'])
        # we rebuild a full experiment
        # data.visual_stim.init_experiment(data.visual_stim.protocol,
        #                                 data.visual_stim.protocol)
        # nFull = len(data.visual_stim.experiment['time_start'])
        # data.build_visual_stim() # back to real recording 
        # text += '- episodes: %i / %i   \n' % (nReal, nFull)
        text += '- episodes: %i \n' % nReal
    text += '\n'

    text += '## mouse & preparation \n\n'
    text += '- ID: %s  \n' % data.nwbfile.subject.subject_id
    text += '- virus: %s \n' % data.nwbfile.virus
    text += '- genotype/strain : %s / %s \n' %\
         (data.nwbfile.subject.genotype, data.nwbfile.subject.strain)
    text += '- age @rec: %s  \n' % data.nwbfile.subject.age
    text += '- DOB: %s  \n' % data.nwbfile.subject.date_of_birth.strftime('%Y-%m-%d')
    text += ' \n \n'

    text += '### Protocol (%s) \n\n' % data.metadata['protocol']
    for p in data.protocols:
        text += '- %s \n' % p
    text += ' \n'


    def get_settings(subsampling_factor=1,
                     with_visual_stim=False):
        settings = {}

        if data.has_running():
            settings['running'] = {'fig_fraction': 1,
                                    'subsampling': 5*subsampling_factor,
                                    'color': '#1f77b4'}
        if data.has_facemotion():
            settings['facemotion'] = {'fig_fraction': 1,
                                        'subsampling': 2*subsampling_factor,
                                        'color': 'purple'}
        if data.has_pupil():
            settings['pupil'] = {'fig_fraction': 2,
                                    'subsampling': 2*subsampling_factor,
                                    'color': '#d62728'}

        if data.has_LFP():
            settings['LFP'] = {'fig_fraction': 6,
                                'subsampling': 10*subsampling_factor,
                                # 'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([40,data.nROIs]), replace=False),
                                'color': 'b'}

        if with_visual_stim:
            settings['photodiode']= {'color': 'lightgrey', 'fig_fraction': 0.5, 
                                     'subsampling': 10*subsampling_factor}
            settings['visual_stim']= {'color': 'w', 'fig_fraction': 0.5}

        return settings

    # full recording view
    if redo_figs:
        fig, AX = \
            plot_raw(data, 
                    tlim=[0, data.tlim[1]], 
                    settings=get_settings(subsampling_factor=20),
                    fig_args=dict(ax_scale=(2.5,25.), 
                                bottom=.02, top=.001, left=.3, right=.3))
        pt.save(fig, os.path.join(notebook_folder, 'figs'), 
                fn+'-full.svg')

    text += '## Full View: %.1fs (%.1fmin)     \n' % (data.tlim[1], data.tlim[1]/60.)
    text += '![](figs/%s)    \n \n' % (fn+'-full.svg')

    ## zoomed view
    for i, t0 in enumerate(\
        np.linspace(0, data.tlim[-1]-60,3)):
        if redo_figs:
            fig, AX = \
                plot_raw(data, 
                        tlim=[t0, t0+60],
                        settings=get_settings(with_visual_stim=True),
                        fig_args=dict(ax_scale=(2.5,20.), 
                                    bottom=.001, top=.15, left=.3, right=.3))
            pt.save(fig, os.path.join(notebook_folder, 'figs'), 
                    fn+'-%i.svg' % (i+1))

        text += '## Zoom %i : 1min @ %.1fmin (%.1fs)    \n' % (i+1, t0/60., t0)
        text += '![](figs/%s)    \n\n' % (fn+'-%i.svg' % (i+1))

    with open(md, 'w') as f:
        f.write(text)
    return text

for i, filename in enumerate(dataset['files']):
    print(i, filename)
    text = single_rec(dataset['files'][i])

# %%
text = single_rec(dataset['files'][0])
print(text)
# %%
