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

parser.add_option('--inputDir', type=str, default='/sps/ztf/users/gris/dataLC',
                  help='input data dir [%default]')
parser.add_option('--outputDir',type=str,default='/sps/ztf/users/gris/infoLC', help='output dir for processed data [/%default]')

opts, args = parser.parse_args()


fis = glob.glob('{}/meta*.hdf5'.format(opts.inputDir))
print(fis)
fams = get_metaFamily(fis)
print(fams)
for metaPrefix in fams:
    cmd = 'python batch_scripts/patch_info.py'
    cmd += ' --metaDir {}'.format(opts.inputDir)
    cmd += ' --metaPrefix {}'.format(metaPrefix)
    cmd += ' --outputDirInfo {}'.format(opts.outputDir)

    print(cmd)
    os.system(cmd)

