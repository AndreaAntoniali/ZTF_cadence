import matplotlib.pyplot as plt
import ztf_cadence_input as cadence_input
from ztf_pipeutils.ztf_util import make_dict_from_config
from ztf_cadence.plot_cadence import PlotNights
from optparse import OptionParser
import pandas as pd
import glob

# get all possible simulation parameters and put in a dict
path = list(cadence_input.__path__)
confDict = make_dict_from_config(path[0], 'config_visucadence.txt')

parser = OptionParser()
# parser : 'dynamical' generation
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0], vals[1]))
    parser.add_option('--{}'.format(key), help='{} [%default]'.format(
        vals[2]), default=vv, type=vals[0], metavar='')

opts, args = parser.parse_args()

# load the new values
params = {}
for key, vals in confDict.items():
    newval = eval('opts.{}'.format(key))
    params[key] = newval

print(params)

# load observations

#df = pd.read_hdf(opts.obsFile)

fis = glob.glob('../pixelized_newformat/*.hdf5')

df = pd.DataFrame()
for fi in fis:
    dd = pd.read_hdf(fi)
    df = pd.concat((df, dd))

plot_nights = PlotNights(df, opts.nside, opts.outDir, opts.nproc)

plot_nights(opts.displaytype)
