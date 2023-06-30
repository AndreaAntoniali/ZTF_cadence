import numpy as np
import os

from optparse import OptionParser




def prod(cmd, conf, zmin, zmax, ntransient, stretch_mean, stretch_sigma, color_mean, color_sigma, config_fil

    Parameters
    ----------
    cmd : TYPE
        DESCRIPTION.
    conf : TYPE
        DESCRIPTION.
    zmin : TYPE
        DESCRIPTION.
    zmax : TYPE
        DESCRIPTION.
    ntransient : TYPE
        DESCRIPTION.
    stretch_mean : TYPE
        DESCRIPTION.
    stretch_sigma : TYPE
        DESCRIPTION.
    color_mean : TYPE
        DESCRIPTION.
    color_sigma : TYPE
        DESCRIPTION.
    config_file : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    cmd_ = cmd

    metaName = 'Meta_{}_{}_{}.hdf5'.format(zmin, zmax, conf)
    LCName = 'LC_{}_{}_{}.hdf5'.format(zmin, zmax, conf)
    LCDir = '../dataLC/{}/{}'.format(config_file,conf)
    # simu
    cmd_ += ' --obsDir=../fake_obs/{}/{}'.format(config_file, conf)
    cmd_ += ' --obsFile=fake_data_obs_{}.hdf5'.format(conf)
    cmd_ += ' --cadDir=../fake_obs/{}/{}'.format(config_file, conf)
    cmd_ += ' --cadFile=cadenceMetric_{}.hdf5'.format(conf)
    cmd_ += ' --zmin {}'.format(zmin)
    cmd_ += ' --zmax {}'.format(zmax)
    cmd_ += ' --ntransient {}'.format(ntransient)
    cmd_ += ' --metaName {}'.format(metaName)
    cmd_ += ' --lcName {}'.format(LCName)
    cmd_ += ' --outputDirSimu {}'.format(LCDir)
    cmd_ += ' --nproc 4'
    cmd_ += ' --stretch_mean={} \
            --color_mean={} \
            --stretch_sigma={} \
            --color_sigma={}'.format(stretch_mean, color_mean, stretch_sigma, color_sigma)
    # info
    metaInfoName = 'Meta_Info_{}_{}_{}.hdf5'.format(zmin, zmax, conf)
    infoDir = '../dataInfo/{}/{}'.format(config_file,conf)
    cmd_ += ' --metaFile={} \
            --metaDir={} \
            --infoFile={} \
            --outputDirInfo={}'.format(metaName, LCDir, metaInfoName, infoDir)

    # fit
    metaDirOutput = '../dataSN/{}/{}'.format(config_file, conf)
    metaFileOutput = 'SN_{}_{}_{}.hdf5'.format(zmin, zmax, conf)

    cmd_ += ' --metaFileInput={} \
            --metaDirInput={} \
            --metaFileOutput={} \
            --metaDirOutput={}'.format(metaInfoName, infoDir,
                                       metaFileOutput, metaDirOutput)

    os.system(cmd_)







parser = OptionParser()

parser.add_option('--config', type=str, default='conf1',
                  help='configuration chosen in the config.csv file [%default]')


parser.add_option('--config_file', type=str, default='config.csv',
                  help='the file with all the configuration [%default]')

parser.add_option('--z_min', type=float, default=0.01,
                  help='minimum z value for producing simulations [%default]')

parser.add_option('--z_max', type=float, default=0.20,
                  help='maximum z value for producing simulations [%default]')

parser.add_option('--n_transient', type=int, default=100,
                  help='number of transient to produce [%default]')

parser.add_option('--delta_z', type=float, default=0.01,
                  help='the bins interval of the z[%default]')

parser.add_option('--stretch_mean', type=float, default=-2,
                  help='The stretch factor x1 mean value [%default]')

parser.add_option('--color_mean', type=float, default=0.2,
                  help='the color factor c mean value [%default]')
parser.add_option('--stretch_sigma', type=float, default=-2,
                  help='The stretch factor x1 mean value [%default]')

parser.add_option('--color_sigma', type=float, default=0.2,
                  help='the color factor c mean value [%default]')

opts, args = parser.parse_args()

conf = opts.config
config_file = opts.config_file
zmin = opts.z_min
zmax = opts.z_max
ntransient = opts.n_transient
delta_z = opts.delta_z
stretch_mean = opts.stretch_mean
stretch_sigma= opts.stretch_sigma
color_mean = opts.color_mean
color_sigma = opts.color_sigma


cmd = 'python run_scripts/simu_info_fit/run_simu_info_fit_pixels.py'

z = list(np.arange(zmin, zmax, delta_z))

for i in range(len(z)-1):
    zmin = z[i]
    zmax = zmin+delta_z
    zmin = np.round(zmin, 2)
    zmax = np.round(zmax, 2)
    prod(cmd, conf = conf, zmin=zmin, zmax=zmax, ntransient=ntransient,\
         stretch_mean=stretch_mean, stretch_sigma=stretch_sigma,\
             color_mean=color_mean, color_sigma=color_sigma, config_file=config_file)