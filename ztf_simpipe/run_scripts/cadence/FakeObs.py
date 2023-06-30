import pandas as pd
from optparse import OptionParser
import os
from ztf_cadence_utils.FakeObs import generateObsData
import numpy as np

parser = OptionParser()

parser.add_option('--cad_g', type=int, default=2,
                  help='cadence for the green band [%default]')
parser.add_option('--cad_r', type=int, default=2,
                  help='cadence for the red band [%default]')
parser.add_option('--cad_i', type=int, default=4,
                  help='cadence for the infrared band [%default]')
parser.add_option('--N_g', type=int, default=1,
                  help='Number of visits per night (g band) [%default]')
parser.add_option('--N_r', type=int, default=1,
                  help='Number of visits per night (r band) [%default]')
parser.add_option('--N_i', type=int, default=1,
                  help='Number of visits per night (i band) [%default]')

parser.add_option('--skynoise_g', type=int, default=30.692339655985883,
                  help='skynoise for green band [%default]')
parser.add_option('--skynoise_r', type=int, default=35.56558820077846,
                  help='skynoise for red band [%default]')
parser.add_option('--skynoise_i', type=int, default=33.49885752052873,
                  help='skynoise for infrared band [%default]')

parser.add_option('--MJD_min', type=int, default=2.458435*10**6,
                  help='MJD minimum [%default]')
parser.add_option('--zp_g', type=int, default=26.275,
                  help='zero point [%default]')
parser.add_option('--zp_r', type=int, default=26.325,
                  help='zero point [%default]')
parser.add_option('--zp_i', type=int, default=25.66,
                  help='zero point [%default]')

parser.add_option('--healpixID_min', type=int, default=0,
                  help='number of pixel min [%default]')
parser.add_option('--healpixID_max', type=int, default=10,
                  help='number of pixel max [%default]')
parser.add_option('--step', type=int, default=1,
                  help='step for healpixID values [%default]')

parser.add_option('--output_dir', type=str, default='dataLC',
                  help='output directory for the fake obs data [%default]')
parser.add_option('--filename', type=str, default='fake_data_obs.hdf5',
                  help='filename for the fake obs data [%default]')

parser.add_option('--seed', type=int, default=1,
                  help='seed [%default]')

opts, args = parser.parse_args()

cad_g = opts.cad_g
cad_r = opts.cad_r
cad_i = opts.cad_i



skynoise_g = opts.skynoise_g
skynoise_r = opts.skynoise_r
skynoise_i = opts.skynoise_i

zp_g = opts.zp_g
zp_r = opts.zp_r
zp_i = opts.zp_i

MJD_min = opts.MJD_min

healpixID_min = opts.healpixID_min
healpixID_max = opts.healpixID_max
step = opts.step

output_dir = opts.output_dir
filename = opts.filename

option_dict = vars(opts)
del option_dict['output_dir']
del option_dict['filename']
print(option_dict)

df = generateObsData(**option_dict)
print(df)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

name_hdf5 = '{}/{}'.format(output_dir, filename)
df.to_hdf(name_hdf5, key='fake_data', mode='w')
name_csv = name_hdf5.replace('.hdf5', '.csv')
df.to_csv(name_csv, index=False)
