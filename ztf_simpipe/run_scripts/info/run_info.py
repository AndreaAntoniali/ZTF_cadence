from optparse import OptionParser
from ztf_simfit.ztf_info import get_selec, Info
from ztf_pipeutils.ztf_util import make_dict_from_config, make_dict_from_optparse
from ztf_pipeutils.ztf_util import dump_in_yaml, checkDir
import astropy
import ztf_simfit_input as simfit_input
from ztf_pipeutils.ztf_util import multiproc
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def process(vv, params, j=0, output_q=None):
    """
    method to perform simulations on multiple healpixIDs

    Parameters
    --------------
    vv : array
      list of data to process
    params: dict
      dict of parameters
    j: int, opt
      internal tag for multiprocessing (proc number) (default: 0)
    output_q: multiprocessing queue,opt
     (default: None)

    Returns
    -----------
    liste of (lc,metadata)

    """
    tab_select = params['tab_select']
    seltab = get_selec(vv, tab_select)

    if output_q is not None:
        return output_q.put({j: seltab})
    else:
        return seltab


# get all possible parameters and put in a dict
path = list(simfit_input.__path__)
confDict = make_dict_from_config(path[0], 'config_info.txt')

parser = OptionParser()
# parser : 'dynamical' generation
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0], vals[1]))
    parser.add_option('--{}'.format(key), help='{} [%default]'.format(
        vals[2]), default=vv, type=vals[0], metavar='')

opts, args = parser.parse_args()


csvInfo = opts.csvInfo
csvSelect = opts.csvSelect
metaFile = opts.metaFile
metaDir = opts.metaDir
snr = opts.snr
infoFile = opts.infoFile
outDir = opts.outputDirInfo
nproc = opts.nproc

# dump parameters in yaml file
checkDir(outDir)
nameOut = infoFile.replace('.hdf5', '.yaml')
dump_in_yaml(opts, confDict, outDir, nameOut, 'info')


# load csv  file in Table
tab_info = astropy.io.ascii.read(csvInfo, format='csv', comment='#')
tab_select = astropy.io.ascii.read(csvSelect, format='csv', comment='#')
print("csv info : \n \n {} \n {}".format(csvInfo, csvSelect))
# get infos
info = Info(metaFile, metaDir, tab_info, snr)
restab = info(bad_prefix='Rej')

params = {}
params['tab_select'] = tab_select
# apply selection
seltab = multiproc(restab, params, process, nproc)

assert(len(seltab) == len(restab))
"""
print(seltab['n_phase_neg', 'n_phase_pos',
             'n_phase_min', 'n_phase_max', 'sel'])
"""
# writing result
fOut = '{}/{}'.format(outDir, infoFile)
astropy.io.misc.hdf5.write_table_hdf5(
    seltab, fOut, path='meta', overwrite=True, serialize_meta=False)
