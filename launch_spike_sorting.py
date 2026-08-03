# %%
import os, sys
import spikeinterface.sorters as ss
import spikeinterface.full as si

for folder in os.listdir(sys.argv[-1]):

    print()
    print('    running "%s" ' % folder)
    print()

    rec_name = folder.split(os.path.sep)[-1]

    rec = si.read_openephys(\
        os.path.join(sys.argv[-1], folder),
        stream_name='Record Node 101#OneBox-100.ProbeA')

    sorting = ss.run_sorter(sorter_name='kilosort4', 
                            recording=rec,
                            verbose=True,
                            folder=os.path.join(sys.argv[-1], 
                                                'kilosort4_%s' % rec_name))

