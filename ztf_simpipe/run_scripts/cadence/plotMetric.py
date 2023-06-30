import pandas as pd
from optparse import OptionParser
import matplotlib.pylab as plt
import matplotlib as mpl
import os
from ztf_metrics.plot_metric import PlotOS
import numpy as np
import matplotlib.colors as colors

parser = OptionParser()

parser.add_option('--nside', type=int, default=128)
parser.add_option('--outDir', type=str, default='plot_cadence/2D_map',
                  help='output directory name [%default]')

parser.add_option('--input_dir', type=str, default='ztf_pixelized',
                  help='folder directory name [%default]')

parser.add_option('--title', type=str, default='', help='title [%default]')
parser.add_option('--inum', type=int, default=1)

opts, args = parser.parse_args()

nside = opts.nside
outDir = opts.outDir

input_dir = opts.input_dir
title = opts.title
inum = opts.inum

tab = pd.DataFrame()
for filename in os.listdir(input_dir):
    print(filename)
    tab_file = pd.read_hdf('{}/{}'.format(input_dir, filename))
    tab = pd.concat([tab, tab_file])
print(tab.columns)

def plot2Dmap(dico, var, tab, vardisp, title, save, min_, max_, cbar):
    
    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    for variables in var :
        ipos = dico[variables][0]
        jpos = dico[variables][1]
        
        min_tab = min_
        max_tab = max_
        
        if variables=='all':
            plt.axes(axs[ipos, jpos])
            cl.visu(tab=tab, vardisp=vardisp, healpixId='healpixID', title=title+' all', inum=inum, 
                        save=save, min_tab=min_tab, max_tab=max_tab, cbar=cbar)
        else :
            tab_c_S = tab[tab['season']==variables]
    
            plt.axes(axs[ipos, jpos])
            cl.visu(tab=tab_c_S, vardisp=vardisp, healpixId='healpixID', title=title+' S{}'.format(variables), inum=inum, 
                        save=save, min_tab=min_tab, max_tab=max_tab, cbar=cbar)
    return fig 

clnm = ['season', 'cad_all', 'nb_obs_all', 'gap_all',
       'season_length_all', 'cad_ztfg', 'nb_obs_ztfg','gap_ztfg', 'season_length_ztfg', 'cad_ztfr',
       'nb_obs_ztfr', 'gap_ztfr', 'season_length_ztfr','cad_ztfi', 'nb_obs_ztfi', 'gap_ztfi', 'season_length_ztfi',
        'healpixRA', 'healpixDec']


min_tab = [0.001 ]*len(clnm)
var = list(tab['season'].unique())
var.append('all')
dico = dict(zip(var, [(0,0), (0,1), (1,0), (1,1), (2,0)]))
print(dico)

cl = PlotOS()

for c in clnm:
    print(c, ':')
    if (c=='cad_all') or (c=='cad_ztfg') or (c=='cad_ztfr') or (c=='cad_ztfi'):
        min_tab = 0.001
        max_tab = 5
        fig = plot2Dmap(dico, var, tab, vardisp=c, title=c, save=False, min_=min_tab, max_=max_tab, cbar=True)
        
        
    elif (c=='nb_obs_all') or (c=='nb_obs_ztfg') or (c=='nb_obs_ztfr') or (c=='nb_obs_ztfi'):
        min_tab = 0.001
        max_tab = 100
        fig = plot2Dmap(dico, var, tab, vardisp=c, title=c, save=False, min_=min_tab, max_=max_tab, cbar=True)
        
    else:
        tab_c = tab[c]
        min_tab = 0.001
        max_tab = tab_c.max()
        fig = plot2Dmap(dico, var, tab, vardisp=c, title=c, save=False, min_=min_tab, max_=max_tab, cbar=True)
 

    if not os.path.exists(outDir):
        os.makedirs(outDir)
        figName = '{}/_{}'.format(outDir, c)
        fig.savefig(figName)
    else:
        figName = '{}/_{}'.format(outDir, c)
        fig.savefig(figName)