from ztf_pipeutils.ztf_batch import BatchIt
import numpy as np


scriptName = 'run_scripts/cadence/pixelize.py'
outDir = '/sps/ztf/users/gris/pixelized'

deltaRA = 36.

for ra in np.arange(0.,360.,deltaRA):
    ramin = ra
    ramax = ra+deltaRA
    outFile = 'data_{}_{}.hdf5'.format(ramin,ramax)
    processName = 'pixelize_{}_{}'.format(ramin,ramax)

    bbatch = BatchIt(processName=processName)

    params = {}
    params['RAmin'] = ramin
    params['RAmax'] = ramax
    params['Decmin'] = -45.
    params['outDir'] = outDir
    params['outFile'] = outFile
    params['logObs'] = 'cadence_logs/2018_all_logs_from_dr1_rcid_zp_from_masci.csv'

    bbatch.add_batch(scriptName,params)

    bbatch.go_batch()
    
