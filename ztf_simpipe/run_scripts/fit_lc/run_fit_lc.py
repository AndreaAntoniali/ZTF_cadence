from optparse import OptionParser
from ztf_simfit.ztf_fit import SN_fit_tab
from ztf_pipeutils.ztf_hdf5 import Read_LightCurve
import astropy
from ztf_pipeutils.ztf_util import multiproc, dump_in_yaml, checkDir
from ztf_pipeutils.ztf_util import make_dict_from_config, make_dict_from_optparse
import ztf_simfit_input as simfit_input
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def fit(metaTable, params={}, j=0, output_q=None):
    """
    Function to perfom LC fits
    with the possibility to use multiprocessing

    Parameters
    ----------------
    metaTable: astropy table
       array of data to process
    params: dict
      parameters of the function
    j: int, opt
      tag for multiprocessing (default: 0)
    output_q: multiprocessing.queue,opt
      queue for multiprocessing

    Returns
    ----------
    resfit: astropy table
      result of the fit
    """
    sn_fit = SN_fit_tab(metaTable)

    resfit = sn_fit()

    if output_q is not None:
        return output_q.put({j: resfit})
    else:
        return 0


# get all possible simulation parameters and put in a dict
path = list(simfit_input.__path__)
confDict = make_dict_from_config(path[0], 'config_fit_lc.txt')

parser = OptionParser()
# parser : 'dynamical' generation
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0], vals[1]))
    parser.add_option('--{}'.format(key), help='{} [%default]'.format(
        vals[2]), default=vv, type=vals[0], metavar='')

opts, args = parser.parse_args()

metaFileInput = opts.metaFileInput
metaDirInput = opts.metaDirInput
metaFileOutput = opts.metaFileOutput
metaDirOutput = opts.metaDirOutput
nproc = opts.nprocFit

# dump parameters in yaml file
checkDir(metaDirOutput)
nameOut = metaFileOutput.replace('.hdf5', '.yaml')
dump_in_yaml(opts, confDict, metaDirOutput, nameOut, 'fit')


if __name__ == '__main__':
    meta = Read_LightCurve(file_name=metaFileInput, inputDir=metaDirInput)
    metaTable = meta.get_table(path='meta')

    params = {}
    resfit = multiproc(metaTable, params, fit, nproc)

    print(len(metaTable), len(resfit))

    # write results in file
    fOut = '{}/{}'.format(metaDirOutput, metaFileOutput)
    astropy.io.misc.hdf5.write_table_hdf5(
        resfit, fOut, path='meta', overwrite=True, serialize_meta=False)
