#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:14:16 2023

@author: cosmostage
"""


import numpy as np
from scipy.optimize import curve_fit
from astropy.cosmology import w0waCDM
from scipy.stats import norm
import Working_minuit as wk
import working_cosmo_minuit as wcm
from scipy import interpolate
from astropy.table import QTable, Table, Column
from astropy import units as u
from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5


import matplotlib.pyplot as plt

import matplotlib as mpl
from cycler import cycler

#I define the global parameters for my plots
mpl.rcParams['lines.linewidth'] = 2

mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'r', 'g', 'y', 'b'])
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.grid.which'] = 'major'
mpl.rcParams['grid.linewidth'] = 1.2
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['figure.figsize'] = [12, 8]
mpl.rcParams['font.size']= 25


mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['legend.shadow'] = True

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['xtick.top'] = True

mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.labelright'] = False

def plot_z_mu(N_novae, sample):
    '''
    Plot a sample of N_Novae SNIa with their distance modulus for their redshift.

    Parameters
    ----------
    N_novae : int
        Number of SNIa
    sample : int
        Sample to take to plot. (must be between 1 and 100 usually)

    '''
    thepath = '{}SNIa_{}'.format(N_novae, sample)
    t = read_table_hdf5('hdf5_simu', path = thepath)
    fig, ax = plt.subplots(layout='constrained')
    ax.errorbar(t['z'], t['mu'], yerr = t['sigma'], fmt='o')
    ax.set_xlabel(r'z')
    ax.set_ylabel(r'$\mu$')
    ax.set_title('Distance modulus of {} SNIa on the redshift'.format(N_novae))

def plot_z_mu_hist(N_novae):
    '''
    Plot a sample of N_Novae SNIa with their distance modulus for their redshift.

    Parameters
    ----------
    N_novae : int
        Number of SNIa
    sample : int
        Sample to take to plot. (must be between 1 and 100 usually)

    '''
    t = wcm.generate_perrett_distrib(N_novae)
    fig, ax = plt.subplots(layout='constrained')
    ax.hist(t, bins = np.arange(0.01, 1.11, 0.02), density = True)
    ax.set_xlabel(r'z')
    ax.set_title('Distance modulus of {} SNIa on the redshift'.format(N_novae))
    
    
def plot_sigma_SNIa(file_name, parameter_to_plot):
    '''
    Take into a file a sample of uncertainties (by default 100) for a given
    parameter depending on the number of SNIa and does a mean value of all the 
    samples and then plot them. 

    Parameters
    ----------
    N_novae : int
        Number of SNIa
    parameter_to_plot : str
        name of the parameter's uncertainties to plot

    '''
    p = 'cov_{}_fisher'.format(parameter_to_plot)
    fig, ax = plt.subplots(layout='constrained')
    print(p)
    mean_sig = []
    dist_sig = []
    for i in range(100, 1001, 50):
        thepath = 'fit_{}_SNIa'.format(i)
        t = read_table_hdf5(file_name, path = thepath)
        print(t[p])
        print(i)

        sig = np.sqrt(t[p])
        print(sig)
        mean_sig.append(np.nanmean(sig))
        dist_sig.append(np.nanstd(sig))
    ax.errorbar(np.arange(100, 1001, 50), mean_sig, yerr=dist_sig,  fmt='o')
    ax.set_xlabel(r'Number of SNIa')
    ax.set_ylabel(r'$\sigma$ {}'.format(parameter_to_plot))
    ax.set_title('Uncertainty for {} on the number of SNIa'.format(parameter_to_plot))
    
def plot_sigma_SNIa_two(file_name, par1, par2):
    p1 = 'cov_{}'.format(par1)
    p2 = 'cov_{}'.format(par2)
    
    mean1_sig = []
    mean2_sig = []
    dist1_sig = []
    dist2_sig = []
    for i in range(100, 1001, 50):
        
        thepath = 'fit_{}_SNIa'.format(i)
        t = read_table_hdf5(file_name, path = thepath)
        sig1, sig2 = np.sqrt(t[p1]), np.sqrt(t[p2])
        
        mean1_sig.append(np.mean(sig1))
        dist1_sig.append(np.std(sig1))
        mean2_sig.append(np.mean(sig2))
        dist2_sig.append(np.std(sig2))
    fig = plt.figure(layout='constrained')
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    
    l1 = ax1.errorbar(np.arange(100, 1001, 50), mean1_sig, yerr=dist1_sig, 
                ecolor = 'r',  fmt='o')
    l2 = ax1.errorbar(np.arange(100, 1001, 50), mean2_sig, yerr=dist2_sig, 
                ecolor = 'k',  fmt='^')
    ax1.set_xlabel(r'$N_{SNIa}$')
    ax1.set_ylabel(r'$\sigma$')
    ax1.legend((l1, l2), (r'$\sigma$({})'.format(par1),\
                          r'$\sigma$({})'.format(par2)))
    
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)
    ax2.errorbar(np.arange(100, 1001, 50), mean1_sig, yerr=dist1_sig, 
                ecolor = 'r',  fmt='o', label = r'$\sigma$({})'.format(par1))
    ax2.legend()
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax3.errorbar(np.arange(100, 1001, 50), mean2_sig, yerr=dist2_sig, 
                ecolor = 'r',  fmt='o', label = r'$\sigma$({})'.format(par2))
    ax3.legend()
    ax3.set_xlabel(r'$N_{SNIa}$')
    
def plot_FoM_SNIa(file_name, p1, p2):
    fig, ax = plt.subplots(layout='constrained')
    mean_FoM = []
    dist_FoM = []
    for i in range(100, 1001, 50):
        thepath = 'fit_{}_SNIa'.format(i)
        t = read_table_hdf5(file_name, path = thepath)
        mean_FoM.append(np.mean(t['FoM']))
        dist_FoM.append(np.std(t['FoM']))
    ax.errorbar(np.arange(100, 1001, 50), mean_FoM, yerr=dist_FoM,  fmt='o')
    ax.set_xlabel(r'Number of SNIa')
    ax.set_ylabel(r'FoM of ({},{})'.format(p1, p2))
    print(t.meta['prior'])
    if (t.meta['prior'] == 'no prior'):
        ax.set_title('FoM for ({}, {}) on the number of SNIa'.format(p1, p2))
    else:
        ax.set_title('FoM for ({}, {}) on the number of SNIa with a prior on $\Omega_m $'.format(p1, p2))

def plot_models():
    z = np.arange(0.01, 2, 0.01)
    w1 = w0waCDM(0.7, 0.3, 0.7)
    w2 = w0waCDM(0.7, 0, 1)
    w3 = w0waCDM(0.7, 1, 0)
    w4 = w0waCDM(0.7, 0.4, 0.6)
    fig, ax = plt.subplots(layout='constrained')
    ax.plot(z, w1.distmod(z).value, '--', label = r'$(\Omega_M, \Omega_{\Lambda}) = (0.3, 0.7)$')
    ax.plot(z, w2.distmod(z).value, '-.', label = r'$(\Omega_M, \Omega_{\Lambda}) = (0, 1))$')
    ax.plot(z, w3.distmod(z).value, label = r'$(\Omega_M, \Omega_{\Lambda}) = (1, 0)$')
    ax.plot(z, w4.distmod(z).value, ':', color = 'b', label = r'$(\Omega_M, \Omega_{\Lambda}) = (0.4, 0.6)$')
    ax.legend()
    ax.set_xlabel(r'Redshift z')
    ax.set_ylabel(r'Distance modulus $\mu$')
