#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:48:10 2023

@author: cosmostage
"""

import numpy as np
from scipy.optimize import curve_fit
from astropy.cosmology import w0waCDM
from scipy.stats import norm
import Working_minuit as wk
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import interpolate
from astropy.table import QTable, Table, Column
from astropy import units as u
from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5



def rate_SNIa(z):
    '''
    For a given redshift z, returns the Perrett's rate of supernovae SNIa 
    expected at this z :  r = (1+z)**2,11

    Parameters
    ----------
    z : numerical value
        The redshift value

    Returns
    -------
    r : numerical value
        The rate of supernovae
    '''
    r = (1+z)**2.11
    return r


def generate_perrett_distrib(N, low_z = 0.01, high_z = 1.1, step = 0.01):
    '''
    Generate a distribution of redshift of supernovae SNIa according to 
    Perrett's rate : r = (1+z)**2,11

    Parameters
    ----------
    N : integer
        Number of supernovae to generate
    low_z : numerical value
        Lower bound from which the supernovae will be generated. 
        The Perrett's rate is more accurate above z = 0.1.
        The default is 0.01.
    high_z : numerical value
        Upper bound at which the supernovae will be generated. 
        The Perrett's rate is more accurate below z = 1.1 The default is 1.1.
        
    step : numerical value
        the distance between the base value of z generated.  
        The default is 0.01.

    Returns
    -------
    zsimu : array of numerical values
        An array containing the redshift value of N supernovae according to
        Perrett's rate distribution.  

    '''
    rng = np.random.default_rng()
    z = np.arange(low_z, high_z, step)
    distrib = rate_SNIa(z)
    norm = np.sum(distrib)
    weight = distrib/norm
    
    zsimu = rng.choice(z, N, p = weight)
    return zsimu


def fom(covariance_matrix, DeltaXi2 = 6.17):
    '''
    Calculates the Figure of Merit such as : FoM = pi/A
    A being : A = pi*(DeltaXi2) *sigma_w0 *sigma_wa *sqrt(1-pearson**2)

    Parameters
    ----------
    covariance_matrix : numpy array of float
        a covariance matrix 
    DeltaXi2 : float, optional
        The degree of confidence level wanted. The default is 6.3.
        Note : 
            1 sigma : CL : 68.3%, DeltaXi2 : 2.3
            2 sigma : CL : 95,4%, DeltaXi2 : 6.17
            3 sigma : CL : 9np.arange(0.01, 1.11, 0.01)9,7 % DeltaXi2 : 11.8

    Returns
    -------
    Figure_of_merit : float
        A numerical value indicating the accuracy of our covariance matrix.
    '''
    pearson = covariance_matrix[0, 1]/(np.sqrt(covariance_matrix[0, 0])*\
                                              np.sqrt(covariance_matrix[1, 1]))
        

    A = np.pi * DeltaXi2 * np.sqrt(covariance_matrix[0, 0])* np.sqrt(covariance_matrix[1, 1]) * np.sqrt(1-pearson**2)
    Figure_of_merit = np.pi/A
    
    return (Figure_of_merit)


def plot_cosmo(N, min_sig = 0.001, max_sig =0.06):
    '''
    Plot a cosmological fit of the three parameters w0, wa and omega_m for a number 
    of supernovae given. 

    Parameters
    ----------
    N : int
        number f supernovae generated according to Perret's rate 

    Returns
    -------
    None.

    '''
    z = generate_perrett_distrib(N)
    f = interpolate.interp1d([0, 1.1], [min_sig, max_sig])
    sigma_inter = f(z)
    cosmo = w0waCDM(H0=70,Om0 = 0.3, Ode0=0.7,  w0=-1, wa = 0)
    distance_modulus = np.random.normal(cosmo.distmod(z).value, sigma_inter)
    A = wk.Fit(z, distance_modulus, sigma_inter)
    m = A.minuit_fit(-1, 0, 0.3)
    par = m.values
    par_error = m.errors
    
    cosmo_fit = w0waCDM(H0=70,Om0 = par[2], Ode0=0.7,  w0=par[0], wa = par[1])
    
    par_error = np.round(par_error, 3)
    par = np.round(par, 3)
    plt.figure(figsize =(10, 8))
    plt.errorbar(z, distance_modulus, yerr=sigma_inter, fmt= '.' , \
                 label = 'Observations')
        
    plt.plot(np.linspace(0.01, 1.1, 100), \
              cosmo.distmod(np.linspace(0.01, 1.1, 100)), '',  \
              label = r'Fit for $w_0 $= {} $\pm$ {},$w_a$ = {} $\pm$ {},$\Omega_m$ = {} $\pm$ {} '\
                  .format(par[0], par_error[0],\
                          par[1], par_error[1], par[2], par_error[2]))
        
    plt.legend(fontsize = 15)
    plt.xlabel('Redshit z', fontsize = 15)
    plt.ylabel(r'distance modulus $\mu$', fontsize = 15)
    plt.title(r'Redshift againt $\mu$ for {} supernovae with a prior on $\Omega_m$'.format(len(z)),\
              fontsize = 15)
    
def get_N_fom(N_novae, N_fom, min_sig = 0.001, max_sig =0.06):
    '''
    generates N Figure of Merit for a given number of SNIa. 

    Parameters
    ----------
    N_novae : int
        Number of SNIa to generate for each FoM. 
    N_fom : int
        Number of FoM values to be computed. 
    min_sig : float, optional
        minimum value of sigma that we will interpolate. The default is 0.001.
    max_sig : float, optional
        maximum value of sigma that we will interpolate. The default is 0.06.

    Returns
    -------
    fom_fisher : 
        DESCRIPTION.
    fom_minuit : TYPE
        DESCRIPTION.

    '''
    fom_fisher = []
    fom_minuit = []
    for i in range(0, N_fom):
        
        f = interpolate.interp1d([0, 1.1], [min_sig, max_sig])
        
        z = generate_perrett_distrib(N_novae)
        sigma_inter = f(z)
        
        cosmo = w0waCDM(H0=70,Om0 = 0.3, Ode0=0.7,  w0=-1, wa = 0)
        dist = np.random.normal(cosmo.distmod(z).value,  sigma_inter)

        A = wk.Fit(z, dist, sigma_inter)
        
        fom_fisher.append(fom(A.covariance_fisher(-1, 0, 0.3)))
        fom_minuit.append(fom(A.minuit_covariance(-1, 0, 0.3)))
        
    return np.array(fom_fisher), np.array(fom_minuit)


def hist_N_fom(N_novae, N_fom, min_sig = 0.001, max_sig =0.06):
    '''
    Plot the distributions of N_fom of N_novae each. 

    Parameters
    ----------
   N_novae : int
       Number of SNIa to generate for each FoM. 
   N_fom : int
       Number of FoM values to be computed. 
   min_sig : float, optional
       minimum value of sigma that we will interpolate. The default is 0.001.
   max_sig : float, optional
       maximum value of sigma that we will interpolate. The default is 0.06.

    Returns
    -------
    None.

    '''
    
    fom_fisher, fom_minuit = get_N_fom(N_novae, \
                                       N_fom, min_sig=min_sig, max_sig= max_sig)
    plt.figure()
    plt.hist(fom_fisher, bins=np.linspace(
        np.min(fom_fisher), np.max(fom_fisher), 10),\
            histtype='stepfilled', \
             label = 'fom_fisher', alpha = 0.7)
    plt.hist(fom_minuit, bins=np.linspace(
        np.min(fom_fisher), np.max(fom_fisher), 10),  histtype='stepfilled',\
             label = 'fom_minuit', alpha = 0.7)
    plt.xlabel('FoM')
    plt.ylabel('count')
    plt.title('Histogram of the Figure of Merit of {} SNIa'.format(N_novae))
    plt.legend(title='N_fom : {}'.format(N_fom))
    
def plot_FoM_on_N_SNIa():
    '''
    Plot the figure of merit value on the number of SNIa

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(12, 8))   
    fom_f = []
    fom_m = []
    nb_nova_i = []
    for i in range(1, 120, 10):
        N_supernovae = 10*i
        nb_nova_i.append(N_supernovae)
        nb_nova_i.append(N_supernovae)
        nb_nova_i.append(N_supernovae)
        nb_nova_i.append(N_supernovae)
        nb_nova_i.append(N_supernovae)
        fom_f_i, fom_m_i = get_N_fom(N_supernovae, 5)
        fom_f.append(fom_f_i)
        fom_m.append(fom_m_i)
    print(np.array(fom_m).flatten())
    print(np.array(nb_nova_i).flatten())
    
    plt.scatter(np.array(nb_nova_i).flatten(), np.array(fom_f).flatten(),\
                c='black', s = 5, label = 'FoM Fisher')
    plt.scatter(np.array(nb_nova_i).flatten(), np.array(fom_m).flatten(),\
                c='red', s= 5, label = 'FoM iMinuit')
    plt.title('Evolution of the FoM depending of the number of SNIa', fontsize=14)
    plt.grid()
    plt.ylabel('FoM', fontsize=14)
    plt.xlabel('Number of SNIa', fontsize=14)
    plt.legend(fontsize=14)

def plot_FoM_on_N_SNIa_different_lambda(N, max_1, max_2, N_start = 100, \
                                        N_step =50):  
    '''
    Plot the figure of merit for different number of SNIa and with two 
    different maximum value of sigma. 

    Parameters
    ----------
    N : int
        Maximum number of SNIa to look at (x upper limit)

    max_1 : float
        1st higher value of sigma for the higher redshift. 
        
    max_2 : float
        1st higher value of sigma for the higher redshift. 
        
    N_start : int, optional
        The starting number of SNIa generated. The default is 100.
        
    N_step : int, optional
        The increase of the number of SNIa between each FoM calculated.
        The default is 50.
    '''
    fom_f_1 = []
    fom_m_1 = []
    fom_f_2 = []
    fom_m_2 = []
    nb_nova_i = []
    for i in range(N_start, N, N_step):
        N_supernovae = i
        nb_nova_i.append(N_supernovae)
        fom_f_i, fom_m_i = get_N_fom(N_supernovae, 1, max_sig=max_1)
        fom_f_1.append(fom_f_i)
        fom_m_1.append(fom_m_i)
        fom_f_i, fom_m_i = get_N_fom(N_supernovae, 1, max_sig=max_2)
        fom_f_2.append(fom_f_i)
        fom_m_2.append(fom_m_i)
        
    plt.figure(figsize=(12, 8))     
    plt.plot(np.array(nb_nova_i).flatten(), np.array(fom_f_1).flatten(),\
                c='black', \
                    label = r'FoM Fisher for a max $\sigma$ = {}'.format(max_1))
    plt.plot(np.array(nb_nova_i).flatten(), np.array(fom_m_1).flatten(), '--', \
                c='black',  \
                    label = r'FoM iMinuit for a max $\sigma$ = {}'.format(max_1))
    plt.plot(np.array(nb_nova_i).flatten(), np.array(fom_f_2).flatten(),\
                c='red',  \
                    label = r'FoM Fisher for a max $\sigma$ = {}'.format(max_2))
    plt.plot(np.array(nb_nova_i).flatten(), np.array(fom_m_2).flatten(),'--',\
                c='red', \
                    label = r'FoM iMinuit for a max $\sigma$ = {}'.format(max_2))
    plt.title('Evolution of the FoM depending of the number of SNIa', fontsize=14)
    plt.grid()
    plt.ylabel('FoM', fontsize=14)
    plt.xlabel('Number of SNIa', fontsize=14)
    plt.legend(fontsize=14)

def compare_minuit_fisher_sigma(N_novae)         :                   
    min_sig, max_sig = 0.001, 0.06
    z = generate_perrett_distrib(N_novae)
    f = interpolate.interp1d([0, 1.1], [min_sig, max_sig])
    sigma_inter = f(z) 
    cosmo = w0waCDM(H0=70,Om0 = 0.3, Ode0=0.7,  w0=-1, wa = 0)
    distance_modulus = np.random.normal(cosmo.distmod(z).value, sigma_inter)
    A = wk.Fit(z, distance_modulus, sigma_inter)
    m = A.minuit_fit(-1, 0, 0.3)
    par = m.values
    par_error = m.errors
    
    fisher_error = A.fisher_uncertainty_matrix(-1, 0, 0.3)
    w_0_relat = (par_error[0] - fisher_error[0])/par_error[0]*100
    w_a_relat = (par_error[1] - fisher_error[1])/par_error[1]*100
    
    # print('w_0 relative uncertainty between fisher and minuit :\n' \
    #       , w_0_relat)
    # print('w_a relative uncertainty between fisher and minuit :\n' \
    #       , w_a_relat)
    return w_0_relat, w_a_relat

    
def plot_compare_minuit_fisher_sigma_mean(N_mean, N_min, N_max, N_step = 10):
    w_0_l = []
    w_a_l = []
    for i in range(N_min, N_max, N_step):
        w_0_r = []
        w_a_r = []
        for j in range(1, N_mean+1):
            w_0_i, w_a_i = compare_minuit_fisher_sigma(i)
            w_0_r.append(w_0_i)
            w_a_r.append(w_a_i)
            
        w_0_l.append(np.mean(w_0_r))
        w_a_l.append(np.mean(w_a_r))
    plt.figure(figsize=(12,8))
    plt.plot(np.arange(N_min, N_max, N_step), w_0_l, label = r'$\Delta w_0$')
    #plt.plot(np.arange(N_min, N_max, N_step), w_a_l, label = r'$\Delta w_a$')
    plt.legend(fontsize = 16)
    plt.xlabel(r'$N_{SNIa}$', fontsize = 16)
    plt.ylabel(r'$\frac{\sigma_{minuit}-\sigma_{fisher}}{\sigma_{minuit}}$ (%)', \
               fontsize = 16)
    plt.title('Relative uncertainty between Fisher and Minuit with the values meaned {} times'\
              .format(N_mean))


def plot_uncertainties_meaned(N_mean, min_novae, max_novae, step_novae = 10, \
                              min_sig = 0.001, max_sig = 0.06):
                             
    p1_err_mean = []
    p2_err_mean = []
    p1_err_std = []
    p2_err_std = []
    for i in range(min_novae, max_novae+1, step_novae): 
        p1_err = []
        p2_err = []
        for j in range(N_mean):
            z = generate_perrett_distrib(i)
            f = interpolate.interp1d([0, 1.1], [min_sig, max_sig])
            sigma_inter = f(z) 
            cosmo = w0waCDM(H0=70,Om0 = 0.3, Ode0=0.7,  w0=-1, wa = 0)
            distance_modulus = np.random.normal(cosmo.distmod(z).value, sigma_inter)
            A = wk.Fit(z, distance_modulus, sigma_inter)
            m = A.minuit_fit(-1, 0, 0.3)
            par_error = m.errors
            p1_err.append(par_error[0])
            p2_err.append(par_error[1])
        p1_err_mean.append(np.mean(p1_err))
        p2_err_mean.append(np.mean(p2_err))
        p1_err_std.append(np.std(p1_err))
        p2_err_std.append(np.std(p2_err))
        
    plt.figure(figsize=(12, 8))
    plt.title('Uncertainties on the number of SNIa, means and deviation for {} values'.format(N_mean))
    plt.errorbar(np.arange(min_novae, max_novae+1, step_novae), \
                  p1_err_mean, yerr= p1_err_std, fmt='ok', label = r'$\sigma w_0$')
    plt.errorbar(np.arange(min_novae, max_novae+1, step_novae), \
                  p2_err_mean, yerr= p2_err_std, fmt='or', label = r'$\sigma w_a$')
    plt.xlabel(r'$N_{SNIa}$', fontsize = 18)
    plt.ylabel('Uncertainty', fontsize = 18)
    plt.legend(fontsize = 16)
    plt.grid()
    
    return p1_err_mean, p1_err_std



def writing_sim(N_novae, N_samples, sigma_min = 0.001, sigma_max = 0.06, w0 = -1, wa = 0, Om = 0.3):
    for i in range(N_samples):
        t = Table(meta={'sigma_min' : sigma_min, 'sigma_max': sigma_max,'Om': Om,\
                        'w0':w0, 'wa': wa })
        thepath = '{}SNIa_{}'.format(N_novae,i+1)
        t['z'] = generate_perrett_distrib(N_novae)
        t['sigma'] = interpolate.interp1d([0, 1.1], [sigma_min, sigma_max])(t['z'])
        cosmo = w0waCDM(H0=70,Om0 = Om, Ode0=0.7,  w0=w0, wa = wa)
        t['mu'] = np.random.normal(cosmo.distmod(t['z']), t['sigma'])
        write_table_hdf5(t, 'hdf5_simu', path=thepath , append=True, overwrite=True,compression = True)
        
# for i in range(100, 1001, 50):
#     writing_sim(i, 100)
    
def writing_fit():
    for j in range(100, 1001, 50):
        thepath = '{}SNIa_{}'.format(j,1)
        a = read_table_hdf5('hdf5_simu', path=thepath)
        R = Table(meta=a.meta)
        R.meta['prior'] = 'no prior'
        key, w0, wa, Om, cov_w0, cov_wa, cov_Om, sig_w0_wa, sig_w0_Om, sig_wa_Om,\
            cov_w0_fisher, cov_wa_fisher, cov_Om_fisher, sig_w0_wa_fisher, sig_w0_Om_fisher,\
                sig_wa_Om_fisher, Xi = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for i in range(0, 10):
                thepath = '{}SNIa_{}'.format(j,i+1)
                t = read_table_hdf5('hdf5_simu', path=thepath)
                params = t.meta['w0'], t.meta['wa'], t.meta['Om']
                F = wk.Fit(t['z'], t['mu'], t['sigma'])
                m = F.minuit_fit(*params)
                Fis_cov = F.covariance_fisher(*params)
                m_param = m.values
                cov = m.covariance
                key.append(thepath)
                w0.append(m_param[0])
                wa.append(m_param[1])
                Om.append(m_param[2])
                #Minuit covariance matrix
                cov_w0.append(cov[0][0])
                cov_wa.append(cov[1][1])
                cov_Om.append(cov[2][2])
                sig_w0_wa.append(cov[0][1])
                sig_w0_Om.append(cov[0][2])
                sig_wa_Om.append(cov[1][2])
                #Fisher covariance matrix 
                cov_w0_fisher.append(Fis_cov[0][0])
                cov_wa_fisher.append(Fis_cov[1][1])
                cov_Om_fisher.append(Fis_cov[2][2])
                sig_w0_wa_fisher.append(Fis_cov[0][1])
                sig_w0_Om_fisher.append(Fis_cov[0][2])
                sig_wa_Om_fisher.append(Fis_cov[1][2])
                
                Xi.append(F.xi_square(*params)/(j-3))
                
                
        R['key'], R['w0_fit'],R['wa_fit'],R['Om_fit'], R['cov_w0'],R['cov_wa'],R['cov_Om'],\
                 R['cov_w0_wa'], R['cov_w0_Om'], R['cov_wa_Om'],\
                 R['cov_w0_fisher'],R['cov_wa_fisher'],R['cov_Om_fisher'],\
                     R['cov_w0_wa_fisher'], R['cov_w0_Om_fisher'], R['cov_wa_Om_fisher'],\
                         R['Xi2']= \
            key, w0, wa, Om, cov_w0, cov_wa, cov_Om, sig_w0_wa, sig_w0_Om, sig_wa_Om,\
                cov_w0_fisher, cov_wa_fisher, cov_Om_fisher, \
                    sig_w0_wa_fisher, sig_w0_Om_fisher, sig_wa_Om_fisher, \
                        Xi
        thepath = 'fit_{}_SNIa'.format(j)
    
        write_table_hdf5(R, 'hdf5_fit_no_prior', path = thepath, overwrite=True, append = True)

def hist_data_fit_deviation(parameter_to_look):
    distribution = []
    p = parameter_to_look
    for i in range(100, 1001, 50):
        thepath = 'fit_{}_SNIa'.format(i)
        a = read_table_hdf5('hdf5_fit', path = thepath)
        distribution.append((a['{}_fit'.format(p)] - a.meta[p])/(a['sig_{}'.format(p)]))
        
    distribution = np.array(distribution).flatten()
    mu, sigma = norm.fit(distribution)
    x = np.linspace(-3*sigma, 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    plt.figure()
    plt.title(r'Histogram of $\frac{\Omega_m(fit) - \Omega_m(theorical)}{\sigma(\Omega_m)}$', fontsize = 16) 
    plt.plot(x, y, label = r'Fit for $\mu = {}$, $\sigma = {}$'.format(np.round(mu, 3), np.round(sigma, 3)))
    plt.hist(distribution, bins = np.linspace(-3*sigma, 3*sigma, 50), density=True)
    plt.plot(x, norm.pdf(x, 0, 1), 'k--', label = 'Normal distribution')
    plt.legend()
    plt.xlabel('Deviation')
    plt.ylabel('Probability')

def writing_fit_w_Om():
    for j in range(100, 1001, 50):
        thepath = '{}SNIa_{}'.format(j,1)
        a = read_table_hdf5('hdf5_simu', path=thepath)
        R = Table(meta=a.meta)
        R.meta['prior'] = 'no prior'
        key, w, Om, cov_w, cov_Om, cov_w_Om, Xi = \
            [], [], [], [], [], [], [], []
        for i in range(0, 100):
                thepath = '{}SNIa_{}'.format(j,i+1)
                t = read_table_hdf5('hdf5_simu', path=thepath)
                params = t.meta['w0'], t.meta['Om']
                F = wk.Fit(t['z'], t['mu'], t['sigma'])
                m = F.minuit_fit(*params)
                m_param = m.values
                cov = m.covariance
                m = F.minuit_fit(*params)
                Fis_cov = F.covariance_fisher(*params)
                m_param = m.values
                cov = m.covariance
                key.append(thepath)
                w0.append(m_param[0])
                wa.append(m_param[1])
                Om.append(m_param[2])
                #Minuit covariance matrix
                cov_w0.append(cov[0][0])
                cov_wa.append(cov[1][1])
                cov_Om.append(cov[2][2])
                sig_w0_wa.append(cov[0][1])
                sig_w0_Om.append(cov[0][2])
                sig_wa_Om.append(cov[1][2])
        R['key'], R['w_fit'],R['Om_fit'], R['cov_w'],R['cov_Om'],\
              R['cov_w_Om'], R['FoM'], R['Xi2']= \
            key, w, Om, cov_w, cov_Om, cov_w_Om, FoM, Xi
        thepath = 'fit_{}_SNIa'.format(j)
    
        write_table_hdf5(R, 'hdf5_fit_w_Om', path = thepath, overwrite=True, append = True)

def fom_file(file_name, N_novae, par1, par2, DeltaXi2 = 6.17):
    '''
    Calculates the Figure of Merit from a given file and table such as : FoM = pi/A
    A being : A = pi*(DeltaXi2) *sigma_w0 *sigma_wa *sqrt(1-pearson**2)
    
    Parameters
    ----------
    file_name : str
        the name of the file 
    N_novae : int
        the number of SNI_a taken into this table. 
    par1 : str
        the 1st parameter of the FoM
    par2 : str
        the 2nd parametre of the FoM
    DeltaXi2 : float, optional
        The degree of confidence level wanted. The default is 6.3.
        Note : 
            1 sigma : CL : 68.3%, DeltaXi2 : 2.3
            2 sigma : CL : 95,4%, DeltaXi2 : 6.17
            3 sigma : CL : 9np.arange(0.01, 1.11, 0.01)9,7 % DeltaXi2 : 11.8

    Returns
    -------
    Figure_of_merit : float
        A numerical value indicating the accuracy of our covariance matrix.
    '''
    FoM_list = []
    thepath = 'fit_{}_SNIa'.format(N_novae)
    t = read_table_hdf5(file_name, path = thepath)
    for i in range(0, len(t['key'])):
        pearson = t['cov_{}_{}'.format(par1, par2)][i]/(np.sqrt(t['cov_{}'.format(par1)][i])*\
                                              np.sqrt(t['cov_{}'.format(par2)][i]))

    
        A = np.pi * DeltaXi2 * np.sqrt(t['cov_{}'.format(par1)][i])* np.sqrt(t['cov_{}'.format(par2)][i]) * np.sqrt(1-pearson**2)
        Figure_of_merit = np.pi/A
        FoM_list.append(Figure_of_merit)
    return (FoM_list)


def writing_fisher(n_sample = 100):
    for j in range(100, 1001, 50):
        thepath = '{}SNIa_{}'.format(j,1)
        a = read_table_hdf5('hdf5_simu', path=thepath)
        R = Table(meta=a.meta)
        R.meta['prior'] = 'prior'
        key, var_w0_fisher, var_wa_fisher, var_Om_fisher, \
            cov_w0_wa_fisher, cov_w0_Om_fisher, cov_wa_Om_fisher, Xi = \
                [], [], [], [], [], [], [], []
        for i in range(0, n_sample):
                thepath = '{}SNIa_{}'.format(j,i+1)
                t = read_table_hdf5('hdf5_simu', path=thepath)
                params = t.meta['w0'], t.meta['wa'], t.meta['Om']
                F = wk.Fit(t['z'], t['mu'], t['sigma'])
                Fis_cov = F.covariance_fisher(*params)
                key.append(thepath)
                #Fisher covariance matrix 
                var_w0_fisher.append(Fis_cov[0][0])
                var_wa_fisher.append(Fis_cov[1][1])
                var_Om_fisher.append(Fis_cov[2][2])
                cov_w0_wa_fisher.append(Fis_cov[0][1])
                cov_w0_Om_fisher.append(Fis_cov[0][2])
                cov_wa_Om_fisher.append(Fis_cov[1][2])
                Xi.append(F.xi_square(*params)/(j-3))
                
                
        R['key'], R['var_w0_fisher'],R['var_wa_fisher'],R['var_Om_fisher'], \
            R['cov_w0_wa_fisher'], R['cov_w0_Om_fisher'], R['cov_wa_Om_fisher'],\
                         R['Xi2']= key, var_w0_fisher, var_wa_fisher, var_Om_fisher, \
                             cov_w0_wa_fisher, cov_w0_Om_fisher, cov_wa_Om_fisher, Xi
        thepath = 'fit_{}_SNIa'.format(j)
        write_table_hdf5(R, 'hdf5_fisher_prior_h-3_{}'.format(n_sample), path = thepath, overwrite=True, append = True)
    
def writing_fit_simple(n_sample = 100):
    for j in range(100, 1001, 50):
        thepath = '{}SNIa_{}'.format(j,1)
        a = read_table_hdf5('hdf5_simu', path=thepath)
        R = Table(meta=a.meta)
        R.meta['prior'] = 'Om prior'
        key, w0, wa, Om, var_w0, var_wa, var_Om, \
            cov_w0_wa, cov_w0_Om, cov_wa_Om, Xi = \
               [], [], [], [], [], [], [], [], [], [], []
        for i in range(0, n_sample):
                thepath = '{}SNIa_{}'.format(j,i+1)
                t = read_table_hdf5('hdf5_simu', path=thepath)
                params = t.meta['w0'], t.meta['wa'], t.meta['Om']
                F = wk.Fit(t['z'], t['mu'], t['sigma'])
                m = F.minuit_fit(*params)
                m_param = m.values
                cov = m.covariance
                key.append(thepath)
                w0.append(m_param[0])
                wa.append(m_param[1])
                Om.append(m_param[2])
                #Minuit covariance matrix
                var_w0.append(cov[0][0])
                var_wa.append(cov[1][1])
                var_Om.append(cov[2][2])
                cov_w0_wa.append(cov[0][1])
                cov_w0_Om.append(cov[0][2])
                cov_wa_Om.append(cov[1][2])
                
                
        R['key'], R['w0'], R['wa'], R['Om'], R['var_w0'],R['var_wa'],R['var_Om'],\
            R['cov_w0_wa'], R['cov_w0_Om'], R['cov_wa_Om'] = key, w0, wa, Om, var_w0, var_wa, var_Om, \
                             cov_w0_wa, cov_w0_Om, cov_wa_Om
        thepath = 'fit_{}_SNIa'.format(j)
        write_table_hdf5(R, 'hdf5_fit_prior_h-7_{}'.format(n_sample), path = thepath, overwrite=True, append = True)
    
