from optparse import OptionParser
from ztf_pipeutils.ztf_util import make_dict_from_config, make_dict_from_optparse
from ztf_simfit.ztf_simu import Simul_lc
from ztf_pipeutils.ztf_hdf5 import Write_LightCurve
from ztf_pipeutils.ztf_util import multiproc, dump_in_yaml, checkDir
from astropy.table import Table, vstack
import ztf_simfit_input as simfit_input
import os
import pathlib
import pandas as pd


def simu(index, params, j=0, output_q=None):

    params['ntransient'] = index[0]
    params['seed'] += j
    obs = params['obs']
    ra_range = params['ra_range']
    dec_range = params['dec_range']
    del params['ra_range']
    del params['dec_range']
    del params['obs']
    # lc simulation
    simulc = Simul_lc(**params)
    lc = simulc(obs, ra_range, dec_range)
    metadata = lc.meta
    meta_rej = lc.meta_rejected
    if output_q is not None:
        return output_q.put({j: ([lc], [metadata], [meta_rej])})
    else:
        return 0


# get all possible simulation parameters and put in a dict
path = list(simfit_input.__path__)
confDict = make_dict_from_config(path[0], 'config_simu.txt')

parser = OptionParser()
# parser : 'dynamical' generation
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0], vals[1]))
    parser.add_option('--{}'.format(key), help='{} [%default]'.format(
        vals[2]), default=vv, type=vals[0], metavar='')

opts, args = parser.parse_args()

print('Start processing...')

# load the new values
params = {}
for key, vals in confDict.items():
    newval = eval('opts.{}'.format(key))
    #params[key] = (vals[0], newval)
    params[key] = newval
print(params)

if params['ztfdataDir'] == 'default':
    import ztf_data as ztf_data
    path = list(ztf_data.__path__)
    params['ztfdataDir'] = path[0]

ntransient = params['ntransient']
nproc = params['nprocSimu']
params['z_range'] = (params['zmin'], params['zmax'])
params['ra_range'] = (params['ramin'], params['ramax'])
params['dec_range'] = (params['decmin'], params['decmax'])
outputDir = params['outputDirSimu']
lcName = params['lcName']
metaName = params['metaName']
path_prefix = params['path_prefix']

# load observations
obsPath = os.path.join(params['obsDir'], params['obsFile'])
file_extension = pathlib.Path(obsPath).suffix
if file_extension == '.csv':
    obs = pd.read_csv(obsPath)
if file_extension == '.hdf5':
    obs = pd.read_hdf(obsPath)
params['obs'] = obs

# dump script parameters in yaml file
checkDir(outputDir)
nameOut = metaName.replace('.hdf5', '.yaml')
dump_in_yaml(opts, confDict, outputDir, nameOut, 'simu')

if __name__ == '__main__':
    step = int(ntransient/nproc)
    ffi = [step]*nproc
    lc_dict = multiproc(ffi, params, simu, nproc, gather=False)

    # write LC and metadata
    Write = Write_LightCurve(
        outputDir=outputDir, file_data=lcName, file_meta=metaName, path_prefix=path_prefix)
    rlc = []
    rmeta = []
    rmeta_rej = []
    for i, lcl in lc_dict.items():
        for ll in lcl[0]:
            rlc.append(ll)
        for ll in lcl[1]:
            rmeta.append(ll)
        for ll in lcl[2]:
            rmeta_rej.append(ll)

    data = Write.write_data(rlc, rmeta, rmeta_rej)

    """
    Write = Write_LightCurve(
        outputDir=outputDir, file_data=lcName, file_meta=metaName, path_prefix=path_prefix)
    rlc = []
    meta_rejected = Table()
    for i, lcl in lc_dict.items():
        if lcl.meta_rejected is not None:
            meta_rejected = vstack([meta_rejected, Table(lcl.meta_rejected)])
        for lc in lcl:
            rlc.append(lc)

    data = Write.write_data(rlc, meta_rejected)
    """
