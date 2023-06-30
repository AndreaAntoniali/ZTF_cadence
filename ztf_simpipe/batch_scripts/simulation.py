from optparse import OptionParser
from ztf_pipeutils.ztf_util import checkDir
from ztf_pipeutils.ztf_batch import BatchIt
import glob

parser = OptionParser()


parser.add_option('--input_dir', type=str, default='/sps/ztf/users/gris/pixelized_newformat',
                  help='folder directory name [%default]')
parser.add_option('--output_dir', type=str, default='/sps/ztf/users/gris/Simulations/LC',
                  help='output directory [%default]')
parser.add_option('--nproc', type=int, default=8,
                  help='number of procs for multiprocessing [%default]')

opts, args = parser.parse_args()

input_dir = opts.input_dir
output_dir = opts.output_dir
nproc = opts.nproc

checkDir(output_dir)

fis = glob.glob('{}/*.hdf5'.format(input_dir))

print(fis)
thescript = 'run_scripts/simulation/run_simulation_pixels.py'

for fi in fis:
    procName = '.'.join(fi.split('/')[-1].split('.')[:-1])
    procName = procName.replace('data','simulc')
    print(procName,fi.split('/')[-1])
    bb = BatchIt(processName=procName)
    params = {}
    params['fileName'] = fi.split('/')[-1]
    params['input_dir'] = input_dir
    params['outName'] =  fi.split('/')[-1].replace('data','LC')
    params['output_dir'] = output_dir
    params['nproc'] = nproc
    params['ntransients']=10000
    params['zmax'] = 0.20
    params['color_mean']= 0.2
    params['stretch_mean'] = -2.0

    bb.add_batch(thescript,params)
    bb.go_batch()
