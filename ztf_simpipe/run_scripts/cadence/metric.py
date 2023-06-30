import pandas as pd
import numpy as np
from optparse import OptionParser
from ztf_metrics.metricWrapper import processMetric_multiproc
from ztf_pipeutils.ztf_hdf5 import Read_LightCurve
from ztf_pipeutils.ztf_util import checkDir, multiproc
import glob


def load_multi(fis, params={}, j=0, output_q=None):
    """
    Method to load data using multiptocessing

    Parameters
    --------------
   fis: list
      list of files to process
    params: dict, opt
      dict of parameters (default: {})

    Returns
    --------------
    New data frame for the differents pixels with the calculation (value) of differents metrics.
    """

    type_data = params['type_data']
    input_dir = params['input_dir']

    df = pd.DataFrame()
    for fi in fis:
        fName = fi.split('/')[-1]
        dfa = get_data(type_data, input_dir, fName)
        df = pd.concat((df, dfa))

    if output_q is not None:
        return output_q.put({j: df})
    else:
        return df


def get_data(type_data, input_dir, fileName):
    """
    Function to get data

    Parameters
    --------------
    type_data: str
      data type to load (pandas or AstropyTable)
    input_dir: str
      input data dir
    fileName: str
      filename of data to load

    Returns
    ----------
    pandas df ot data

    """
    if type_data == 'pandas':
        df = pd.read_hdf('{}/{}'.format(input_dir, fileName))
    if type_data == 'AstropyTable':
        cl = Read_LightCurve(file_name=fileName, inputDir=input_dir)
        df_A = cl.get_table(path='meta')
        df = pd.DataFrame(np.array(df_A))

    return df


parser = OptionParser()

parser.add_option('--type_data', type=str, default='pandas',
                  help='type of the table in the meta data file (pandas or AstropyTable) [%default]')
parser.add_option('--fileName', type=str, default='data_36.0_72.0.hdf5',
                  help='meta data file name [%default]')
parser.add_option('--input_dir', type=str, default='pixelized_newformat',
                  help='folder directory name [%default]')
parser.add_option('--outName', type=str, default='cadenceMetric_36.0_72.0.hdf5',
                  help='output file name [%default]')
parser.add_option('--output_dir', type=str, default='metricOutput/cadenceMetric',
                  help='output directory [%default]')
parser.add_option('--metric', type=str, default='CadenceMetric',
                  help='metric name [%default]')
parser.add_option('--nproc', type=int, default=8,
                  help='number of procs for multiprocessing [%default]')
parser.add_option('--nside', type=int, default=128,
                  help='nside healpix parameter [%default]')
parser.add_option('--coadd_night', type=int, default=1,
                  help='to perform coaddition per band and per night [%default]')
parser.add_option('--pixelList', type=str, default='all',
                  help='list of pixels to process [%default]')
parser.add_option('--npixels', type=int, default=-1,
                  help='number of pixels to process [%default]')


opts, args = parser.parse_args()

type_data = opts.type_data
fileName = opts.fileName
input_dir = opts.input_dir
outName = opts.outName
output_dir = opts.output_dir
metric_name = opts.metric
nproc = opts.nproc
nside = opts.nside
coadd_night = opts.coadd_night
pixelList = opts.pixelList
npixels = opts.npixels

if __name__ == '__main__':

    if fileName == 'all':
        search_path = '{}/meta_fit*.hdf5'.format(input_dir)
        fis = glob.glob(search_path)
        params = {}
        params['type_data'] = type_data
        params['input_dir'] = input_dir
        df = multiproc(fis, params, load_multi, nproc=8)
    else:
        if '.csv' in fileName:
            fis = pd.read_csv(fileName)['fileName'].to_list()
            params = {}
            params['type_data'] = type_data
            params['input_dir'] = input_dir
            df = multiproc(fis, params, load_multi, nproc=8)
        else:
            df = get_data(type_data, input_dir, fileName)

    df['healpixID'] = df['healpixID'].astype(str)

    # df['healpixID'] = [list(map((lambda x: 'p' + x+'p'), suits))
    #                   for suits in df['healpixID']]
    df['healpixID'] = [','.join(list(map(
        lambda orig_string: 'p'+orig_string + 'p', suits.split(',')))) for suits in df['healpixID']]
    # print(df['healpixID'])

    """
    idx = df['healpixID'] != 'None'
    df = df[idx]
    df['healpixID'] = [list(map(int, i.split(','))) for i in df['healpixID']]
    print('rr', df.columns)
    print(df['healpixID'])
    """

    resdf = processMetric_multiproc(
        metric_name, df, nproc, nside, coadd_night, npixels, pixelList)

    checkDir(output_dir)
    fName = '{}/{}'.format(output_dir, outName)
    resdf.to_hdf(fName, key='metric')
    print(resdf)
