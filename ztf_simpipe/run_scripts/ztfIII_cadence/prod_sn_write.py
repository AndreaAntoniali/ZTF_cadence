import numpy as np
import os

from optparse import OptionParser




def prod(cmd, conf, zmin, zmax, ntransient, stretch_mean, stretch_sigma, color_mean, color_sigma):

    cmd_ = cmd

    metaName = 'Meta_{}_{}_{}.hdf5'.format(zmin, zmax, conf)
    LCName = 'LC_{}_{}_{}.hdf5'.format(zmin, zmax, conf)
    LCDir = '../dataLC/{}'.format(conf)
    # simu
    cmd_ += ' --obsDir=../fake_obs/{} \
            --obsFile=fake_data_obs_{}.hdf5 \
            --cadDir=../fake_obs/{} \
            --cadFile=cadenceMetric_fake_data_obs_{}.hdf5 \
            --zmin {} \
            --zmax {} \
            --ntransient {} \
            --metaName {}\
            --lcName {} \
            --outputDirSimu {} '.format(conf,conf,conf,conf, zmin, zmax, ntransient, metaName, LCName, LCDir)
    cmd_ += '--stretch_mean={} \
            --color_mean={} \
            --stretch_sigma={} \
            --color_sigma={}'.format(stretch_mean, color_mean, stretch_sigma, color_sigma)
    # info
    metaInfoName = 'Meta_Info_{}_{}_{}.hdf5'.format(zmin, zmax, conf)
    infoDir = '../dataInfo/{}'.format(conf)
    cmd_ += ' --metaFile={} \
            --metaDir={} \
            --infoFile={} \
            --outputDirInfo={}'.format(metaName, LCDir, metaInfoName, infoDir)

    # fit
    metaDirOutput = '../dataSN/{}'.format(conf)
    metaFileOutput = 'SN_{}_{}_{}.hdf5'.format(zmin, zmax,conf)

    cmd_ += ' --metaFileInput={} \
            --metaDirInput={} \
            --metaFileOutput={} \
            --metaDirOutput={}'.format(metaInfoName, infoDir,
                                       metaFileOutput, metaDirOutput)
    
    print(stretch_mean, color_mean, zmax)
    os.system(cmd_)







parser = OptionParser()

parser.add_option('--config', type=str, default='conf1',
                  help='configuration chosen in the config.csv file [%default]')

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

parser.add_option('--config_file', type=str, default='config.csv',
                  help='the file with all the configuration [%default]')

opts, args = parser.parse_args()

conf = opts.config
zmin = opts.z_min
zmax = opts.z_max
ntransient = opts.n_transient
delta_z = opts.delta_z
stretch_mean = opts.stretch_mean
stretch_sigma= opts.stretch_sigma
color_mean = opts.color_mean
color_sigma = opts.color_sigma


print(stretch_mean, color_mean, zmax)
cmd = 'python run_scripts/simu_info_fit/run_simu_info_fit_pixels.py'

z = list(np.arange(zmin, zmax, delta_z))

for i in range(len(z)-1):
    zmin = z[i]
    zmax = zmin+delta_z
    zmin = np.round(zmin, 2)
    zmax = np.round(zmax, 2)
    prod(cmd, conf = conf, zmin=zmin, zmax=zmax, ntransient=ntransient,\
         stretch_mean=stretch_mean, stretch_sigma=stretch_sigma,\
             color_mean=color_mean, color_sigma=color_sigma)