#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 08:05:39 2023

@author: cosmostage
"""

import matplotlib.pyplot as plt
import numpy as np
import Working_class_3 as wk
from scipy.optimize import curve_fit
from astropy.cosmology import w0waCDM


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

def function_to_fit(z_array, w0_value = -1, wa_value = 0.00):
    '''

    Parameters
    ----------
    z_array :  array of floats
        An array containing the redshift value of several supernovae SNIa.
        
    w0_value : float, optional
        Dark energy equation of state at z=0 (a=1). This is pressure/density
        for dark energy in units where c=1. The default is -1.
        
    wa_value : float, optional
        Negative derivative of the dark energy equation of state with respect
        to the scale factor. A cosmological constant has w0=-1.0 and wa=0.0. 
        The default is 0.00.

    Returns
    -------
    mu : float
        Distance modulus at each input redshift

    '''
    cosmo = w0waCDM(H0=70, Om0=0.3, Ode0=0.7, w0 = w0_value, wa = wa_value)
    mu = cosmo.distmod(z_array).value
    return mu

def cov_matrix(z_array, sig = 0.01, w0_value = -1, wa_value = 0.00):
    '''
    Given an array of redshift, calculates their distance modulus with a gaussian smear 
    and returns the covariance matrix with the fisher method and the curve_fit one. 

    Parameters
    ----------
    z_array : array of floats
        An array containing the redshift value of several supernovae SNIa.
        
    sig : float, optional
        Uncertainty given on the observation to smear the data.  
        The default is 0.01.
        
    w0_value : float, optional
        Dark energy equation of state at z=0 (a=1). This is pressure/density
        for dark energy in units where c=1. The default is -1.
        
    wa_value : float, optional
        Negative derivative of the dark energy equation of state with respect
        to the scale factor. A cosmological constant has w0=-1.0 and wa=0.0. 
        The default is 0.00.

    Returns
    -------
    pcov : numpy array of float
        the covariance matrix given by the function scipy.optimize.curve_fit().
    F_cov_mat : numpy array of float
        the covariance matrix given by the fisher method of Working_class_3.

    '''
    #We define what to eject into our class.
    sigma_wk = sig * function_to_fit(z_array, w0_value, wa_value)
    dist_wk = np.random.normal(function_to_fit(z_array), sig)
    #We create an instance of our class and inject directly the value wanted. 
    A = wk.Fit(z_array, dist_wk , sigma_wk)
    
    #We create a cosmology. 
    cosmo = w0waCDM(H0=70, Om0=0.3, Ode0=0.7, w0=w0_value, wa =wa_value)
    noised_observation = np.random.normal(function_to_fit(z_array, w0_value, wa_value), sig)
    
    copt, pcov = curve_fit(function_to_fit, z_array, noised_observation,\
                          sigma = sigma_wk, absolute_sigma=True)
    F_cov_mat = np.array(np.mat(A.fisher(w0_value, wa_value, 'distmod')).I)
    return pcov, F_cov_mat


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
            3 sigma : CL : 99,7 % DeltaXi2 : 11.8

    Returns
    -------
    Figure_of_merit : float
        A numerical value indicating the accuracy of our covariance matrix.
    '''
    pearson = covariance_matrix[0, 1]/np.sqrt(covariance_matrix[0, 0]*\
                                              covariance_matrix[1, 1])
    A = np.pi* DeltaXi2 * \
        np.sqrt(covariance_matrix[0, 0]* covariance_matrix[1, 1] * (1-pearson))
    Figure_of_merit = np.pi/A
    return (Figure_of_merit)


def plot_fit(z_array, sig, w0_value = -1, wa_value = 0.00, rounded_precision = 4, xscale = 'linear'):
    '''
    Given an array of redshift, calculates their distance modulus with a gaussian 
    smear and returns the covariance matrix with the fisher method and the 
    curve_fit one, then plot the fitted curve and the observations points 
    with their uncertainties.

    Parameters
    ----------
    z_array : array of floats
    An array containing the redshift value of several supernovae SNIa.
    
    sig : float, optional
    Uncertainty given on the observation to smear the data.  
    The default is 0.01.
    
    w0_value : float, optional
    Dark energy equation of state at z=0 (a=1). This is pressure/density
    for dark energy in units where c=1. The default is -1.
    
    wa_value : float, optional
    Negative derivative of the dark energy equation of state with respect
    to the scale factor. A cosmological constant has w0=-1.0 and wa=0.0. 
    The default is 0.00.
    
    rounded_precision : integer, optional
        To how many signigican value we want to round. The default is 4.
    
    xscale : string sequence, optional
        Gives the scale for the x axis. The default is 'linear'.
        
    Returns
    -------
    uncertainty_fit : numpy array of float
        the uncertainty matrix of the covariance matrix
        given by the function scipy.optimize.curve_fit()
        rounded up to a precision of : rounded_precision
        
    uncertainty_F : numpy array of float
        the uncertainty matrix given by the
        covariance matrix given by the fisher method of Working_class_3, 
        rounded up to a precision of : rounded_precision

    '''
    #We define what to eject into our class.
    sigma_wk = sig * function_to_fit(z_array, w0_value, wa_value)
    dist_wk = np.random.normal(function_to_fit(z_array, w0_value, wa_value), sig)
    
    #We create an instance of our class and inject directly the value wanted. 
    A = wk.Fit(z_array, dist_wk , sigma_wk)
    
    #We create a cosmology
    cosmo = w0waCDM(H0=70, Om0=0.3, Ode0=0.7, w0=w0_value, wa =wa_value)
    #We get our values of covariance and we fit our parameters.
    copt, pcov = curve_fit(function_to_fit, z_array, dist_wk,\
                          sigma = sigma_wk, absolute_sigma=True)
    #We dot that also for F. 
    cov_F = np.array(np.mat(A.fisher(-1, 0, 'distmod')).I)
    
    #We round the value 
    copt_rounded, uncertainty_F, uncertainty_fit = np.round(copt, rounded_precision), \
        np.round(np.sqrt(cov_F), rounded_precision), \
            np.round(np.sqrt(pcov), rounded_precision)
    #We plot our figure
    plt.figure(figsize =(12, 8))
    print(sigma_wk)
    plt.errorbar(z_array, dist_wk, yerr=sigma_wk, fmt= '.' , \
                 label = 'Observations')
    plt.plot(np.linspace(0.01, 1.1, 100), \
             function_to_fit(np.linspace(0.01, 1.1, 100), *copt), '',  \
             label = r'Fit for w0 = {} $\pm$ {}  , wa = {} $\pm$ {}'\
                 .format(copt_rounded[0], uncertainty_fit[0][0],\
                         copt_rounded[1], uncertainty_fit[1][1]))
        
    plt.legend()
    plt.xlabel('Redshit z', fontsize = 15)
    plt.xscale(xscale)
    plt.ylabel(r'distance modulus $\mu$', fontsize = 15)
    plt.title(r'Redshift againt $\mu$ for {} supernovae'.format(len(z_array)),\
              fontsize = 15)
    return uncertainty_F, uncertainty_fit

def compare_fom(N_max, confidence_level):
    '''
    For different numbers of supernovae generated, compare the figure of 
    merit of fisher and curve_fit. 
    Parameters
    ----------
    N_max : integer
        How many supernovae you want to generate at the upper limit. 
    confidence_level : float
    The degree of confidence level wanted. The default is 6.3.
    Note : 
        1 sigma : CL : 68.3%, DeltaXi2 : 2.3
        2 sigma : CL : 95,4%, DeltaXi2 : 6.17
        3 sigma : CL : 99,7 % DeltaXi2 : 11.8
    Returns
    -------
    l_fisher : list of float
        A list of the different Figure of Merit values at different number of 
        supernovae for the fisher method. 
    l_fit : list of float
        A list of the different Figure of Merit values at different number of 
        supernovae for the fit_curve method. 

    '''
    l_fit = []
    l_fisher = []
    for i in range(3, N_max+1, 10):
        z = generate_perrett_distrib(i)
        l_fit.append(fom(cov_matrix(z)[0], DeltaXi2=confidence_level))
        l_fisher.append(fom(cov_matrix(z)[1], DeltaXi2=confidence_level))
        
    plt.plot(np.arange(3, N_max+1, 10), l_fit, label = 'fit fom')
    plt.plot(np.arange(3, N_max+1, 10), l_fisher, label = 'fisher fom')
    plt.xlabel('Number of supernovae generated')
    plt.ylabel(r'Figure of Merit ')
    plt.title(r'Evolution of the Figure of Merit for a confidence level of \
              $\Delta\chi^2$={}'.format(confidence_level))
    plt.legend()
    return l_fisher, l_fit
    
z = generate_perrett_distrib(100)

plot_fit(z, 0.01, xscale='linear', rounded_precision=1)