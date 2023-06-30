from optparse import OptionParser
from ztf_pipeutils.ztf_util import checkDir
from ztf_pipeutils.ztf_batch import BatchIt
import glob
from copy import deepcopy
import numpy as np
import pandas as pd

parser = OptionParser()


parser.add_option('--input_dir', type=str, default='/sps/ztf/users/gris/pixelized_newformat',
                  help='folder directory name [%default]')
parser.add_option('--output_dir', type=str, default='/sps/ztf/users/gris/metricOutput/cadenceMetric',
                  help='output directory [%default]')
parser.add_option('--metric', type=str, default='CadenceMetric',
                  help='metric name (CadenceMetric, RedMagMetric) [%default]')
parser.add_option('--nproc', type=int, default=8,
                  help='number of procs for multiprocessing [%default]')
parser.add_option('--type_data', type=str, default='pandas',
                  help='type of the table in the meta data file (pandas or AstropyTable) [%default]')
opts, args = parser.parse_args()

input_dir = opts.input_dir
output_dir = opts.output_dir
metric_name = opts.metric
nproc = opts.nproc
type_data = opts.type_data

checkDir(output_dir)

fis = glob.glob('{}/*.hdf5'.format(input_dir))
if metric_name == 'RedMagMetric':
    fis = [input_dir]

print(fis)

thescript = 'run_scripts/cadence/metric.py'

for fi in fis:
    procName = '.'.join(fi.split('/')[-1].split('.')[:-1])
    print(procName, fi.split('/')[-1])
    bb = BatchIt(processName=procName)
    params = {}
    params['fileName'] = fi.split('/')[-1]
    params['input_dir'] = input_dir
    params['outName'] = fi.split(
        '/')[-1].replace('data', metric_name.split('Metric')[0].lower())
    params['output_dir'] = output_dir
    params['metric'] = metric_name
    params['nproc'] = nproc
    params['type_data'] = type_data
    if metric_name == 'RedMagMetric':
        params['fileName'] = 'all'
        pp = deepcopy(params)
        fis = glob.glob('{}/meta_fit*.hdf5'.format(input_dir))
        fis = list(map(lambda x: x.split('/')[-1], fis))

        print(len(fis))
        arr = np.array(fis)
        print(arr, arr.shape)
        spl = np.array_split(arr, 10)
        print(spl)

        for i, vv in enumerate(spl):
            inputFile_dir = 'input_{}'.format(metric_name)
            checkDir(inputFile_dir)
            csvName = '{}/{}_{}.csv'.format(inputFile_dir, procName, i)
            df = pd.DataFrame(vv.tolist(), columns=['fileName'])
            df.to_csv(csvName, index=False)
            pp['fileName'] = csvName
            pp['outName'] = '{}_{}.hdf5'.format(params['outName'], i)
            bb.add_batch(thescript, pp)
    else:
        bb.add_batch(thescript, params)
    bb.go_batch()
