import numpy as np
import os

from optparse import OptionParser

parser = OptionParser()

parser.add_option('--config', type=str, default='conf1',
                  help='configuration chosen in the config.csv file [%default]')


opts, args = parser.parse_args()
conf = opts.config

def prod(cmd, zmin=0.01, zmax=0.20, ntransient=100):

    cmd_ = cmd

    metaName = 'Meta_{}_{}_{}.hdf5'.format(zmin, zmax, conf)
    LCName = 'LC_{}_{}_{}.hdf5'.format(zmin, zmax, conf)
    LCDir = '../dataLC/{}'.format(conf)
    # simu
    cmd_ += ' --obsDir=../fake_obs/{} \
            --obsFile=fake_data_obs_{}.hdf5 \
            --cadDir=../fake_obs/{} \
            --cadFile=cadenceMetric_fake_data_obs_{}.hdf5 \
            --stretch_mean=-2.0 --color_mean=0.2 \
            --outputDirSimu dataLC \
            --zmin {} \
            --zmax {} \
            --ntransient {} \
            --metaName {}\
            --lcName {} \
            --outputDirSimu {}'.format(conf,conf,conf,conf, zmin, zmax, ntransient, metaName, LCName, LCDir)

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

    os.system(cmd_)


cmd = 'python run_scripts/simu_info_fit/run_simu_info_fit_pixels.py'

delta_z = 0.01
z = list(np.arange(0.01, 0.2, delta_z))

for i in range(len(z)-1):
    zmin = z[i]
    zmax = zmin+delta_z
    zmin = np.round(zmin, 2)
    zmax = np.round(zmax, 2)
    prod(cmd, zmin=zmin, zmax=zmax)
