import glob
import os
from optparse import OptionParser

def get_metaFamily(fis):
    """
    function to estimate families from a set of files

    Parameters
    ----------------
    fis: list(str)
      list of string

    Returns
    -----------
    list(str): list of families

    """

    r = []
    for fi in fis:
        tt = '_'.join(fi.split('/')[-1].split('_')[:5])
        r.append(tt)

    return set(r)
    

parser = OptionParser()

parser.add_option('--metaDirInput', type=str, default='/sps/ztf/users/gris/infoLC',
                  help='metadata dir input[%default]')
parser.add_option('--metaDirOutput', type=str, default='/sps/ztf/users/gris/fitLC',
                  help='output dir for processed data[%default]')

opts, args = parser.parse_args()

params = vars(opts)

fis = glob.glob('{}/meta*.hdf5'.format(opts.metaDirInput))
print(fis)
fams = get_metaFamily(fis)
print(fams)
for metaPrefix in fams:
    params['metaPrefix'] = metaPrefix
    cmd = 'python batch_scripts/patch_fit.py'
    for key, val in params.items():
        cmd += ' --{} {}'.format(key,val)

    print(cmd)
    os.system(cmd)

