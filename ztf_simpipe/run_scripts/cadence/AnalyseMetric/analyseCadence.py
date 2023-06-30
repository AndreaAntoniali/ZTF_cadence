import pandas as pd
from optparse import OptionParser
import matplotlib.pylab as plt
import os
from ztf_metrics.plot_metric import PlotOS
import numpy as np
from ztf_metrics.plot_cad import PlotCad

parser = OptionParser()

parser.add_option('--outDir', type=str, default='analyse_cadence',
                  help='output directory name [%default]')

parser.add_option('--input_dir', type=str, default='ztf_pixelized',
                  help='folder directory name [%default]')

parser.add_option('--figName', type=str, default='cadence_plot_test',
                  help='figure name [%default]')

parser.add_option('-v', action="store_true", dest="verbose",
                  default=True, help='plot all bands [%default]')
parser.add_option("-q", action="store_false", dest="verbose",
                  help='plot per bands [%default]')

opts, args = parser.parse_args()

outDir = opts.outDir
figName = opts.figName

input_dir = opts.input_dir

tab = pd.DataFrame()
for filename in os.listdir(input_dir):
    tab_file = pd.read_hdf('{}/{}'.format(input_dir, filename))
    tab = pd.concat([tab, tab_file])

tab['cad_all'] = tab['cad_all'].replace(0.0, -1)

cl = PlotCad(tab=tab)

if opts.verbose:
    fig1 = cl.hist_cad(all_band=True)
    figName_ = figName+'_all'
else:
    fig1 = cl.hist_cad()
    figName_ = figName+'_bands'

fig2 = cl.plot_cad()
plt.show()

if not os.path.exists(outDir):
    os.makedirs(outDir)
    figName = '{}/{}'.format(outDir, figName_)
    fig1.savefig(figName)
    fig2.savefig('{}/plot_cad'.format(outDir))
else:
    figName = '{}/{}'.format(outDir, figName_)
    fig1.savefig(figName)
    fig2.savefig('{}/plot_cad'.format(outDir))
