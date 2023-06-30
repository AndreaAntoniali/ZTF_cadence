import pandas as pd
from optparse import OptionParser
import matplotlib.pylab as plt
import os
from ztf_metrics.plot_metric import PlotOS
import numpy as np
import csv

parser = OptionParser()

parser.add_option('--input_dir', type=str, default='ztf_pixelized/cadence_metric',
                  help='folder directory name [%default]')

parser.add_option('--outDir', type=str, default='analyse_cadence',
                  help='output directory name [%default]')

parser.add_option('--outFileName', type=str, default='sortCadence.csv',
                  help='file name of csv [%default]')

opts, args = parser.parse_args()

input_dir = opts.input_dir

outDir = opts.outDir
outFileName = opts.outFileName

if os.path.exists('{}/.DS_Store'.format(input_dir)):
    os.remove('{}/.DS_Store'.format(input_dir))
else:
    print("Impossible de supprimer le fichier car il n'existe pas")

tab = pd.DataFrame()
for filename in os.listdir(input_dir):
    tab_file = pd.read_hdf('{}/{}'.format(input_dir, filename))
    tab = pd.concat([tab, tab_file])

df_new = pd.DataFrame({'healpixID': tab['healpixID'], 'cadence': tab['cad_all'], 'season_length': tab['season_lenght_all'],
                      'nb_bands': [3 if x!=0 else 2 for x in tab['nb_obs_ztfi']]})
df_new = df_new.sort_values(by=['cadence'])
df_new.reset_index()

if not os.path.exists(outDir):
    os.makedirs(outDir)   
    df_new.to_csv('{}/{}'.format(outDir, outFileName), index=False)
else:
    df_new.to_csv('{}/{}'.format(outDir, outFileName), index=False)
    

#with open('{}/{}'.format(outDir, outFileName), 'r') as f:
    # Créer un objet csv à partir du fichier
#    obj = csv.reader(f)
#    for ligne in obj:
#        print(ligne)

print(df_new)


