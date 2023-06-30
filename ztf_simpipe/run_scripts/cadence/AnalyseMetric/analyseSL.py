import pandas as pd
from optparse import OptionParser
import matplotlib.pylab as plt
import os
from ztf_metrics.plot_metric import PlotOS
import numpy as np
import bokeh
import bokeh.plotting

parser = OptionParser()

parser.add_option('--outDir', type=str, default='analyse_cadence',
                  help='output directory name [%default]')

parser.add_option('--fileName', type=str, default='cadence_36.0_72.0.hdf5',
                  help='file name [%default]')

parser.add_option('--input_dir', type=str, default='ztf_pixelized',
                  help='folder directory name [%default]')

parser.add_option('--figName', type=str, default='sl_plot',
                  help='figure name [%default]')

opts, args = parser.parse_args()

outDir = opts.outDir
figName = opts.figName

fileName = opts.fileName
input_dir = opts.input_dir

tab = pd.DataFrame()
for filename in os.listdir(input_dir):
    tab_file = pd.read_hdf('{}/{}'.format(input_dir, filename))
    tab = pd.concat([tab, tab_file])
n = [0, 180, 200, 220, tab['season_lenght_all'].max()]
print(n)

ratio = [] 
sl = []
band_ = []
for i in range(0,len(n)-1):
    print(n[i], n[i+1])
    for band in ['season_lenght_ztfg', 'season_lenght_ztfr', 'season_lenght_ztfi']:
        mask = (tab[band]>=n[i]) & (tab[band]<n[i+1])
        tab_sl = tab[mask]
        ratio.append((tab_sl['healpixID'].count()/tab['healpixID'].count()))
        sl.append(n[i+1]-10)
        band_.append(band)
        
        
df = pd.DataFrame({'sl': sl, 'ratio': ratio, 'band': band_})

df_g = df[df['band']=='season_lenght_ztfg']
df_r = df[df['band']=='season_lenght_ztfr']
df_i = df[df['band']=='season_lenght_ztfi']

color=['green', 'red', 'black']
add = [-5, 0.0, 5]
ls = [':', '-.', '--']
labels = ['ac g band', 'ac r band', 'ac i band']

fig, ax1 = plt.subplots(1, 1, figsize=(8,6))
for i, df_ in enumerate ([df_g, df_r, df_i]):
    df_['accumulate_ratio'] = np.add.accumulate(df_['ratio'])
    bar = ax1.bar(x = df_['sl']+add[i], height = df_['ratio'], width=5, alpha=0.5, color=color[i],
                 label='{}'.format(df_['band'].unique()[0]))
    
    p, = ax1.plot(df_['sl']+add[i], df_['accumulate_ratio'], ls=ls[i], marker='o', color=color[i], label=labels[i])

leg = ax1.legend(title='Accumulate ratio ("ac")\n& season lenght :', loc='best')

ax1.set_xlabel('season lenght')
ax1.set_ylabel(r'$N_{SN}/N_{total}$')
plt.show()

if not os.path.exists(outDir):
    os.makedirs(outDir)
    figName = '{}/{}'.format(outDir, figName)
    fig.savefig(figName)
else:
    figName = '{}/{}'.format(outDir, figName)
    fig.savefig(figName)








