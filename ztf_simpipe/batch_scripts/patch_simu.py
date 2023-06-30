from optparse import OptionParser
from ztf_pipeutils.ztf_batch import BatchIt

parser = OptionParser()

parser.add_option('--script', type=str, default='run_scripts/simulation/run_simulation_pixels.py',
                  help='script to run [%default]')
parser.add_option('--nprocSimu', type=int, default=8,
                  help='number of procs for multiprocessing [%default]')
parser.add_option('--ntransient', type=int, default=500,
                  help='number of transients to generate [%default]')
parser.add_option('--zmin', type=float, default=0.075,
                  help='zmin for SN generation [%default]')
parser.add_option('--zmax', type=float, default=0.13,
                  help='zmax for SN generation [%default]')
parser.add_option('--color_mean', type=float, default=0.2,
                  help='mean color for SN generation [%default]')
parser.add_option('--stretch_mean', type=float, default=-2.0,
                  help='mean x1 for SN generation [%default]')
parser.add_option('--outputDirSimu', type=str, default='/sps/ztf/users/gris/dataLC',
                  help='output dir for simulation [%default]')
parser.add_option('--obsDir', type=str, default='/sps/ztf/users/gris/pixelized_newformat',
                  help='dir with observations [%default]')
parser.add_option('--obsFile', type=str, default='data_36.0_72.0.hdf5',
                  help='obs file to process [%default]')
parser.add_option('--cadDir', type=str, default='/sps/ztf/users/gris/metricOutput/cadenceMetric',
                  help='dir with cadence infos [%default]')
parser.add_option('--npixelsBatch', type=int, default=500,
                  help='npixels per batch run [%default]')

opts, args = parser.parse_args()

params = vars(opts)
params['cadFile'] = params['obsFile'].replace('data', 'cadence')
params['lcName'] = params['obsFile'].replace('data', 'LC')
params['lcName'] = params['lcName'].replace(
    '.hdf5', '_{}_{}.hdf5'.format(params['zmin'], params['zmax']))
params['metaName'] = params['lcName'].replace('LC', 'meta')
params['backupFly'] = 1

print(params)

procName = 'simu_{}_{}_{}'.format(params['obsFile'].split('.hdf5')[0],params['zmin'],params['zmax'])

thescript = params['script']
params.pop('script')
bb = BatchIt(processName=procName)
bb.add_batch(thescript, params)
bb.go_batch()
