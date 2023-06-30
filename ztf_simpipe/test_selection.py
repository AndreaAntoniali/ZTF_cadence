#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:05:07 2023

@author: cosmostage
"""

import warnings
warnings.filterwarnings("ignore")

from astropy.table import QTable, Table, Column, join, vstack
from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5

import numpy as np 
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


import pandas as pd


from scipy import interpolate
from astropy import units as u

from cycler import cycler
import time



from optparse import OptionParser

parser = OptionParser()

parser.add_option('--config', type=str, default='conf_faint1',
                  help='configuration chosen in the config.csv file [%default]')


parser.add_option('--config_file', type=str, default='conf_faint.csv',
                  help='the file with all the configuration [%default]')

opts, args = parser.parse_args()
confs = opts.config
config_file = opts.config_file
confs = confs.split(',')

def plot_error(var_x, var_y, error_y, label_x, label_y, color ='k', \
               marker='x', linestyle = '-', label = None, \
                   fig = None, ax = None):
    if fig is None:
        fig, ax = plt.subplots()
        
    ax.errorbar(var_x, var_y, error_y, color = color, marker = marker, label = label, linestyle = linestyle,  capsize = 7, alpha = 0.8)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    if label is not None:
        ax.legend()


def plot(var_x, var_y, label_x, label_y, color ='k', marker='-',\
         label = None, fig = None, ax = None):
    if fig is None:
        fig, ax = plt.subplots()
        
    ax.plot(var_x, var_y, color = color, marker = marker, label = label)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    if label is not None:
        ax.legend()
        
def browse_directory(directory_name, extension_name = '.hdf5'):
    directory = directory_name
    df = Table()
    print('\
---------------------------------------------------------- \n \
Stacking of all the z range table in the path : \n {}  \n'.format(directory))
    for filename in os.listdir(directory):
        if filename.endswith(extension_name):
              path = os.path.join(directory, filename)
              #print('Stacking file : {}'.format(path))
              df_file = read_table_hdf5(path)
              df = vstack([df, df_file])
    return df

    
def selec(confs, config_file):
    '''
    This function gives back a dictionnary containing Tables for each configurations 
    selected of the selected rows. 
    The one not kept are the rows with no data and with a fit crashing. 

    Parameters
    ----------
    confs : list of string values
        All the differents configurations to be processed
        in the form : ['conf1', 'conf2']
        
    config_file : the file containing the configurations. 
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if confs == ['all']:
        r = pd.read_csv(config_file)
        confs= r['config'].tolist()
    
    config_file= config_file.split('.')[0]
    print('\
---------------------------------------------------------- \n \
########################################################## \n \
Configurations : \n {} \n \
########################################################## \n \
---------------------------------------------------------- \n'.format(confs))
    df_conf = {}
    
    for i in confs :
        name = '../dataSN/{}/{}'.format(config_file, i)
        df = browse_directory(name)
        df_nodata = df[df['fitstatus'] == 'nodata']
        df_fitcrash = df[df['fitstatus'] == 'fitcrash']
        df_sel = df[df['fitstatus'] == 'fitok']
        df_conf[i] = df_sel
        print('\
---------------------------------------------------------- \n\
On {} rows : {} fitok, {} nodata, {} fitcrash \n\
---------------------------------------------------------- \n'.\
    format(np.shape(df)[0], np.shape(df_sel)[0], np.shape(df_nodata)[0],\
           np.shape(df_fitcrash)[0]))
    return df_conf
    
    

def param_on_z(confs, config_file):
    '''
    This function will plot one parameters on z. 

    '''
    
    df = selec(confs, config_file)
    for i in df.keys():
        min_z, max_z = np.min(df[i]['z']), np.max(df[i]['z'])
        bins = np.arange(min_z, max_z, 0.01)
        data = df[i].to_pandas()
        data = data[data['c_err']<0.1]
        z = data['z']
        plot_centers = (bins[:-1] + bins[1:])/2
        
        y = data.groupby(pd.cut(z, bins))
        z_binned = y.mean()
        z_err = y.std()
        red_shift_com = interpolate.interp1d(z_binned['c_err'], \
                                             plot_centers, fill_value = 0, bounds_error=False)(0.04)
            #This is false
        # red_shift_err = np.abs(interpolate.interp1d(z_binned['c_err']+z_err['c_err'], \
        #                                         plot_centers, fill_value = 0, bounds_error=False)(0.04) -red_shift_com)
        # print(z_binned['c_err']+z_err['c_err'])
        # print("Redshift completness = {} \pm {}".format(red_shift_com, red_shift_err))
        
        fig, ax= plt.subplots()
        
        
        plot_error(plot_centers, z_binned['c_err'], z_err['c_err'], "z", r"$\sigma(c)$",\
                   ax = ax, fig = fig, marker = 'o',color='k', linestyle='-', \
                       label = 'binned data points')
        upper_y = np.max(z_binned['c_err'])+0.01
        df
        plt.show()
param_on_z(confs, config_file)

