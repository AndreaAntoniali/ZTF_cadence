import matplotlib.pyplot as plt
import pandas as pd
import healpy as hp
from ztf_metrics.metricUtils import dustMap
from optparse import OptionParser
import numpy as np

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['font.size'] = 12


def plotDustMap(dustmap, ebvofMW_cut=0.25):
    """
    Function to plot the dustmap (ebvofMW) in Mollweid view

    Parameters
    ---------------
    dustmap: pandas df
      data to display
    ebvofMW_cut: float, opt
       cut for the display (default: 0.25)

    """
    npixels = np.unique(dustmap['npixels'])
    if ebvofMW_cut > 0.:
        idx = dustmap['ebvofMW'] <= ebvofMW_cut
        dustmap = dustmap[idx]

    xmin = 0.00001
    xmax = np.max(dustmap['ebvofMW'])

    norm = plt.cm.colors.Normalize(xmin, xmax)
    cmap = plt.cm.jet
    cmap.set_under('w')

    hpxmap = np.zeros(npixels, dtype=np.float)
    hpxmap = np.full(hpxmap.shape, 0.)
    hpxmap[dustmap['healpixID']] += dustmap['ebvofMW']

    hp.mollview(hpxmap, cmap=cmap, nest=True,
                min=xmin, max=xmax, norm=norm,
                title='E(B-V) MW - SFD')
    hp.graticule()


def plotDustHist(dustmap, ebvofMW_cut=0.25):
    """
    Function to plot the dustmap (ebvofMW) in Mollweid view

    Parameters
    ---------------
    dustmap: pandas df
      data to display
    ebvofMW_cut: float, opt
       cut for the display (default: 0.25)

    """
    npixels = np.unique(dustmap['npixels'])
    if ebvofMW_cut > 0.:
        idx = dustmap['ebvofMW'] <= ebvofMW_cut
        dustmap = dustmap[idx]

    fig, ax = plt.subplots()
    ax.hist(dustmap['ebvofMW'], histtype='step', bins=100)
    ax.set_xlabel('E(B-V)')
    ax.set_ylabel('Number of Entries')


parser = OptionParser()
parser.add_option("--nside", type=int, default=64,
                  help="nside healpix parameter[%default]")
parser.add_option("--healpixID", type=int, default=27238,
                  help="healpixel ID to get info from [%default]")
parser.add_option("--ebvofMW_max", type=float, default=0.25,
                  help="E(B-V) max [%default]")

opts, args = parser.parse_args()

dustmap = dustMap(nside=opts.nside)
ebvofMW_max = opts.ebvofMW_max

idx = dustmap['healpixID'] == opts.healpixID

print(dustmap[idx])

plotDustMap(dustmap, ebvofMW_cut=ebvofMW_max)
plotDustHist(dustmap, ebvofMW_cut=ebvofMW_max)

#plt.plot(vec[0], vec[1], 'ko')
plt.show()
