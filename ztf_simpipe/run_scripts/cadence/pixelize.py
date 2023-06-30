import matplotlib.pyplot as plt
from ztf_pipeutils.ztf_util import make_dict_from_config, make_dict_from_optparse
import ztf_cadence_input as cadence_input
from ztf_cadence.pixelize import Pixelize_sky, pixels, Pixel2Obs
from ztf_cadence.align_quadrant import align_quad
from ztf_pipeutils.ztf_util import multiproc, checkDir
from optparse import OptionParser
import pandas as pd
import numpy as np
#import healpy as hp
import time


def sel_obs_sky(df_obs,
                ra_min, ra_max, dec_min, dec_max,
                width_ra, width_dec,
                raCol='ra', decCol='dec', k=1.20):
    """
    Method to select observations on a sky patch
    It is mandatory to extand the area defined by (ra_min,ra_max,dec_min,dec_max)
    to widths of quadrants.

    Parameters
    --------------
    df_obs: pandas df
      data to process
    ra_min: float
      min ra of the area
    ra_max: float
      max ra of the area
   dec_min: float
      min dec of the area
    dec_max: float
      max dec of the area
    raCol: str, opt
      ra col name in df_obs (default: ra)
    decCol: str, opt
      dec col name in df_obs (default: dec)
    k: float, opt
      security factor to grab all the data corresponding to the area (default: 1.05)

    Returns
    ----------
    data inside the area (pandas df)

    """

    ra_lower = ra_min-k*width_ra
    ra_upper =  ra_max+k*width_ra
    dec_lower = dec_min-k*width_dec
    dec_upper = dec_max+k*width_dec
    
    idx = df_obs[raCol] >= ra_lower
    idx &= df_obs[raCol] < ra_upper
    idx &= df_obs[decCol] >= dec_lower
    idx &= df_obs[decCol] < dec_upper

    return df_obs[idx]


def obs_to_pixel(df_obs, nside, nproc, alignquads):
    """
    Method to project data (quadrants) on the sky to grab pixels

    Parameters
    --------------
    df_obs: pandas df
      data to process
    nside: int
      nside param for healpix
    nproc: int
      nproc for multiprocessing
    alignquads: int
      if alignquads was done or not

    """
    params = {}
    params['nside'] = nside
    params['raCol'] = 'ra'
    params['decCol'] = 'dec'

    if opts.alignquads:
        params['raCol'] = 'ra_quad'
        params['decCol'] = 'dec_quad'

    # pixelize the data
    time_ref = time.time()
    print('to process', len(df_obs))
    dfOut = multiproc(df_obs, params, multiobs_pixel, nproc)

    print('processed', time.time()-time_ref)
    #dfOut.to_hdf(opts.outFile, key='obs')
    #dfOut.to_parquet('df.parquet.gzip', compression='gzip')
    return dfOut


def multiobs_pixel(data, params={}, j=0, output_q=None):
    """
    Function to perfom sky pixelizing from data
    with the possibility to use multiprocessing

    Parameters
    ----------------
    data : astropy table
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
    pixelize_it = Pixelize_sky(
        params['nside'], raCol=params['raCol'], decCol=params['decCol'])

    # now process

    print('processing', j, len(data))
    dfOut = pixelize_it(data)

    print('result', dfOut)
    if output_q is not None:
        return output_q.put({j: dfOut})
    else:
        return 0


def pixel_to_obs(healpixels, nside, data, width_ra, width_dec, raCol, decCol, nproc):
    """
    Method to get data corresponding to healpixels

    Parameters
    --------------
    healpixels: list(int)
      list of pixels to process
    nside: int
      nside for healpix
    data: pandas df
       data to process
    width_ra: float
      quadrant ra width
    width_dec: float
      quadrant dec width
    raCol: str
      ra col name
     decCol: str
      dec col name
    nproc: int
     number of procs for multiprocessing

    """
    params['nside'] = nside
    params['data'] = data
    params['width_ra'] = width_ra
    params['width_dec'] = width_dec
    params['raCol'] = raCol
    params['decCol'] = decCol

    print('total to process per proc', len(healpixels)/nproc)
    res = multiproc(healpixels, params, multipixel_obs, nproc)

    return res


def multipixel_obs(ppix, params={}, j=0, output_q=None):
    """
    Method to get data corresponding to healpixels using multiprocessing

    Parameters
    --------------
    ppix: list(int)
      list of pixels to process
    params: dict, opt
      parameters to run (default: none)
    j: int, opt
      index for multiprocessing (default: 0)
    output_q: multiprocessing queue, opt
      default: None

    Returns
    ----------
    pandas df of multiprocessed data

    """
    nside = params['nside']
    df_obs = params['data']
    width_ra = params['width_ra']
    width_dec = params['width_dec']
    raCol = params['raCol']
    decCol = params['decCol']

    print('to process', len(ppix))
    ppo = Pixel2Obs(nside, df_obs, width_ra,
                    width_dec, raCol=raCol, decCol=decCol)
    import time
    dtot = pd.DataFrame()
    time_ref_tot = time.time()
    for ih, healpix in enumerate(ppix):
        time_ref = time.time()
        dp = ppo(healpix)
        dtot = pd.concat((dtot, dp))

        if ih % 1000 == 0:
            print('processed', ih, time.time()-time_ref_tot)

    print('pproc', time.time()-time_ref_tot)

    if output_q is not None:
        return output_q.put({j: dtot})
    else:
        return dtot


def alignquads(df):
    """
    Method to align quadrants using multiprocessing

    Parameters
    --------------
    df: pandas df
       data to processed

    Returns
    ----------
    pandas df of processed data

    """

    pp = {}
    pp['obs'] = df
    # align quadrants
    dfo = multiproc(df['field'].unique(), pp, align_quad, opts.nproc)
    dfo = dfo.sort_values(by=['time'])

    return dfo


# get all possible simulation parameters and put in a dict
path = list(cadence_input.__path__)
confDict = make_dict_from_config(path[0], 'config_pixelize.txt')

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

# get log of observations
logObs = params['logObs']
print('loading', logObs)
df_obs_orig = pd.read_csv(logObs)

ra_min = opts.RAmin
ra_max = opts.RAmax
dec_min = opts.Decmin
dec_max = opts.Decmax

width_ra = opts.width_RA
width_dec = opts.width_Dec

df_obs = sel_obs_sky(df_obs_orig, ra_min, ra_max, dec_min,
                     dec_max, width_ra, width_dec, k=2.0)
if ra_min <= 0.05:
    add_obs = sel_obs_sky(df_obs_orig, 358., 360., dec_min,
                     dec_max, width_ra, width_dec, k=2.0)
    df_obs = pd.concat((df_obs,add_obs))

# re-align quadrant centers - if requested

if opts.alignquads:
    df_obs = alignquads(df_obs)

"""
# get obs in this window
df_obs = df_obs[idx]

# get pixels in this window
ppix = pixels(opts.nside, np.mean([ra_min, ra_max]), np.mean(
    [dec_min, dec_max]), delta_ra, delta_dec)

print(ppix, len(ppix))
"""

# from pixels -> obs
# get pixels in this window
"""
delta_ra = ra_max-ra_min
delta_dec = dec_max-dec_min

ppix = pixels(opts.nside, np.mean([ra_min, ra_max]), np.mean(
    [dec_min, dec_max]), delta_ra, delta_dec)

raCol = 'ra'
decCol = 'dec'
if opts.alignquads:
    raCol = 'ra_quad'
    decCol = 'dec_quad'
df = pixel_to_obs(ppix, opts.nside, df_obs, width_ra,
                  width_dec, raCol, decCol, opts.nproc)
"""

# from obs -> pixels
dfOut = obs_to_pixel(df_obs, opts.nside, opts.nproc, opts.alignquads)

# create outputDir if necessary
checkDir(opts.outDir)
outName = '{}/{}'.format(opts.outDir, opts.outFile)

dfOut.to_hdf(outName, key='obs')
