# %%
import os, sys, shutil
import spikeinterface.sorters as ss
import spikeinterface.full as si

folder  = sys.argv[-1]
rec_name = folder.split(os.path.sep)[-1]

print()
print('    running "%s" ' % folder)
print('                   ', rec_name)
print()


rec = si.read_openephys(\
    os.path.join(sys.argv[-1], folder),
    stream_name='Record Node 101#OneBox-100.ProbeA')


rec = rec.select_channels(rec.get_channel_ids()[:250])


ks_folder=os.path.join(sys.argv[-1], 'kilosort4_%s' % rec_name)
if os.path.isdir(ks_folder):
    y = input(' folder "%s" already exists ! \n Do you want to delete it ? y/[n]' % ks_folder)
    if y in ['y', 'yes']:
        shutil.rmtree(ks_folder)

sorting = ss.run_sorter(sorter_name='kilosort4', 
                        recording=rec,
                        verbose=True,
                        folder=ks_folder)

