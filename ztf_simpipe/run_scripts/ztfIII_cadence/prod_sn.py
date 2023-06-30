import numpy as np
import os


def prod(cmd, zmin=0.01, zmax=0.20, ntransient=100):

    cmd_ = cmd

    metaName = 'Meta_{}_{}.hdf5'.format(zmin, zmax)
    LCName = 'LC_{}_{}.hdf5'.format(zmin, zmax)
    LCDir = '../dataLC'
    # simu
    cmd_ += ' --obsDir=../Fake_Observations \
            --obsFile=fake_data_obs.hdf5 \
            --cadDir=metricOutput/cadenceMetric \
            --cadFile=cadenceMetric_fake_data_obs.hdf5 \
            --stretch_mean=-2.0 --color_mean=0.2 \
            --outputDirSimu dataLC \
            --zmin {} \
            --zmax {} \
            --ntransient {} \
            --metaName {}\
            --lcName {} \
            --outputDirSimu {}'.format(zmin, zmax, ntransient, metaName, LCName, LCDir)

    # info
    metaInfoName = 'Meta_Info_{}_{}.hdf5'.format(zmin, zmax)
    infoDir = '../dataInfo'
    cmd_ += ' --metaFile={} \
            --metaDir={} \
            --infoFile={} \
            --outputDirInfo={}'.format(metaName, LCDir, metaInfoName, infoDir)

    # fit
    metaDirOutput = '../dataSN'
    metaFileOutput = 'SN_{}_{}.hdf5'.format(zmin, zmax)

    cmd_ += ' --metaFileInput={} \
            --metaDirInput={} \
            --metaFileOutput={} \
            --metaDirOutput={}'.format(metaInfoName, infoDir,
                                       metaFileOutput, metaDirOutput)

    os.system(cmd_)


cmd = 'python run_scripts/simu_info_fit/run_simu_info_fit_pixels.py'

delta_z = 0.09
z = list(np.arange(0.01, 0.2, delta_z))

for i in range(len(z)-1):
    zmin = z[i]
    zmax = zmin+delta_z
    zmin = np.round(zmin, 2)
    zmax = np.round(zmax, 2)
    prod(cmd, zmin=zmin, zmax=zmax)
