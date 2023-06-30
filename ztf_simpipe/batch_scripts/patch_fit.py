import glob
import os
from optparse import OptionParser
from ztf_pipeutils.ztf_batch import BatchIt

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = OptionParser()

parser.add_option('--script', type=str, default='run_scripts/fit_lc/run_fit_lc.py',
                  help='script to run [%default]')
parser.add_option('--nprocFit', type=int, default=8,
                  help='number of procs for multiprocessing [%default]')
parser.add_option('--metaDirInput', type=str, default='infoLC',
                  help='metadata dir input[%default]')
parser.add_option('--metaDirOutput', type=str, default='fitLC',
                  help='output dir for processed data[%default]')
parser.add_option('--metaPrefix', type=str, default='meta_info_36.0_72.0',
                  help='prefix for metadata files[%default]')

opts, args = parser.parse_args()

script = opts.script

metaDirInput = opts.metaDirInput
metaDirOutput = opts.metaDirOutput
metaPrefix = opts.metaPrefix

search_path = '{}/{}*.hdf5'.format(metaDirInput, metaPrefix)
print(search_path)
fis = glob.glob(search_path)

procName='fit_{}'.format(opts.metaPrefix)

params = vars(opts)
params.pop('script')
params.pop('metaPrefix')


bb = BatchIt(processName=procName)

for fi in fis:
    metaFileInput = fi.split('/')[-1]
    metaFileOutput = metaFileInput.replace('info', 'fit')
    cmd = '{}'.format(script)
    params['metaFileInput'] = metaFileInput
    params['metaFileOutput'] = metaFileOutput

    bb.add_batch(script, params)
    #print(cmd)
    #os.system(cmd)

bb.go_batch()
