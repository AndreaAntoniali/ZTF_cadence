import glob
import os
from optparse import OptionParser

parser = OptionParser()

parser.add_option('--obsDir', type=str, default='/sps/ztf/users/gris/pixelized_newformat',
                  help='dir with observations [%default]')
parser.add_option('--outputDirSimu', type=str, default='/sps/ztf/users/gris/dataLC',
                  help='output dir for simulation [/%default]')
parser.add_option('--ntransient', type=int, default=100,
                  help='number of transients to generate [%default]')
parser.add_option('--zmin', type=float, default=0.07,
                  help='zmin for SN generation [%default]')
parser.add_option('--zmax', type=float, default=0.08,
                  help='zmax for SN generation [%default]')
parser.add_option('--npixelsBatch', type=int, default=500,
                  help='npixels per batch run [%default]')

opts, args = parser.parse_args()

params = vars(opts)
outputDirSimu = opts.outputDirSimu
fis = glob.glob('{}/data*.hdf5'.format(opts.obsDir))
for fi in fis:
    obsFile = fi.split('/')[-1]
    params['obsFile'] = obsFile
    bb = obsFile.split('.hdf5')[0].replace('data','patch')
    params['outputDirSimu'] = '{}/{}'.format(outputDirSimu,bb)
    cmd = 'python batch_scripts/patch_simu.py'
    for key, val in params.items():
        cmd += ' --{} {}'.format(key, val)

    print(cmd)
    os.system(cmd)
