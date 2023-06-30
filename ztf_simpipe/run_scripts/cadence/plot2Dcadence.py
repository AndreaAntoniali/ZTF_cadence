import pandas as pd
from optparse import OptionParser
from ztf_metrics import plt
import os
from ztf_metrics.plot_metric_for_cadence import PlotCAD
from ztf_metrics.plot_metric import PlotOS
import numpy as np
import matplotlib.colors as colors

parser = OptionParser()

parser.add_option('--nside', type=int, default=128,
                  help='output directory name [%default]')
parser.add_option('--outDir', type=str, default='plot_cadence/2D_map/cadence',
                  help='output directory name [%default]')
parser.add_option('--input_dir', type=str, default='ztf_pixelized',
                  help='folder directory name [%default]')
parser.add_option('--var_to_plot', type=str, default='cad_all,cad_ztfg,cad_ztfr,cad_ztfi',
                  help='variables to plot [%default]')

parser.add_option('--title', type=str, default='', help='title [%default]')
parser.add_option('--inum', type=int, default=1)

opts, args = parser.parse_args()

nside = opts.nside
outDir = opts.outDir

input_dir = opts.input_dir
title = opts.title
inum = opts.inum
clnm = opts.var_to_plot.split(',')

tab = pd.DataFrame()
for filename in os.listdir(input_dir):
    print(filename)
    tab_file = pd.read_hdf('{}/{}'.format(input_dir, filename))
    tab = pd.concat([tab, tab_file])

#tab = tab[tab['ebvofMW'] < 0.25]

#clnm = ['cad_all', 'cad_ztfg', 'cad_ztfr', 'cad_ztfi']

min_tab = [0.001]*len(clnm)
var = list(tab['season'].unique())
var.append('all')

"""
cl = PlotCAD(outDir=outDir)

for c in clnm:
    print(c)
    ido = tab[c] > 0.
    sel = tab[ido]
    min_tab = 0.9*sel[c].min()
    max_tab = 1.1*sel[c].max()

    fig1 = cl.visu(tab=tab, vardisp=c, healpixId='healpixID',
                   title=c, inum=inum, save=False, cbar=True)
    for i in range(1, len(var)):
        print('S_', i)
        fig2 = cl.visu(tab=tab[tab['season'] == i], vardisp=c, healpixId='healpixID', title=c+' S{}'.format(i),
                       inum=inum, save=False, cbar=False)

plt.show()
"""
cl = PlotOS(outDir=outDir)

for c in clnm:
    print(c)
    ido = tab[c] > 0.
    sel = tab[ido]
    min_tab = 0.9*sel[c].min()
    max_tab = 1.1*sel[c].max()
    cl.visu(tab, vardisp=c, healpixId='healpixID',
            title=c, inum=inum, min_tab=min_tab, max_tab=max_tab, save=False, cbar=True)

    """
    fig1 = cl.visu(tab=tab, vardisp=c, healpixId='healpixID',
                   title=c, inum=inum, save=False, cbar=True)
    for i in range(1, len(var)):
        print('S_', i)
        fig2 = cl.visu(tab=tab[tab['season'] == i], vardisp=c, healpixId='healpixID', title=c+' S{}'.format(i),
                       inum=inum, save=False, cbar=False)
    """

"""
fig, ax = plt.subplots(figsize=(10, 8))
idx = tab['zlim'] > 0.
sel = tab[idx]
ax.hist(sel['zlim'], histtype='step', bins=30, color='k', lw=2)
print('median', np.median(sel['zlim']), np.median(sel['mag']))
secax = ax.twiny()
secax.hist(sel['mag'], histtype='step', bins=30, ls='dashed', color='b', lw=2)
ax.grid()
ax.set_xlabel('$z_{complete}$')
ax.set_ylabel('Number of entries')
secax.set_xlabel('mag limit', color='b')
secax.spines['top'].set_color('b')
secax.xaxis.label.set_color('b')
secax.tick_params(axis='x', colors='b')
"""
plt.show()
