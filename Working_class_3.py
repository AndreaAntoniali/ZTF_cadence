"""
Created on Mon Feb 20 15:34:59 2023

@author: cosmostage
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import matplotlib.pyplot as plt
import numpy as np

from astropy.cosmology import w0waCDM
from scipy.optimize import curve_fit


class Fit:
    def __init__(self, x, y , sigma):
        '''
        Initialize the class with the differents variables. 
        
        Parameters
        ----------
        x : array of numerical values 
            Our entry data. 
            
        y : array of numerical values 
            The results of the model with a small noise (e.g our observational data ?)
            
        sigma : array of numerical values 
            The uncertainty on our data points. 
            
        h : a float. 
            Used for differentiating our Xisquare. 
            

        '''
        self.x = x
        self.y = y
        self.sigma = sigma 
        self.h = 1.e-7
        
    def function(self, *parameters):
        '''
        Calculates the function given at the end of the parameters. 
        
        Parameters
        ----------
        *parameters : tuple of differents types of entries. 
            It presents as : (a, b, 'type of function')
            a and b must be a numerical value. 
            'type of function' indicates what function we want to use ; 
            'linear' : a*x + b
            'squareroot' : np.sqrt(a*x+b)
            'expo' : b*np.exp(x*a)
            'distmod' : distance modulus as given by astropy.cosmology for w0waCDM 
        Returns
        -------
        f : array of numerical values
            Results of the array x in the funcion chosen in the parameters via the 
            parameters given.

        '''
        if (parameters[2]=='linear'):
            f = parameters[0]*self.x + parameters[1]
            self.h = np.max(self.x*parameters[0] + parameters[1]) * 10**(-8)
            
        if (parameters[2]=='squareroot'):
            f = np.sqrt(parameters[0]*self.x + parameters[1])
            self.h = np.max(f) * (10**(-7))
            
        if (parameters[2]=='expo'):
            f = parameters[1]*np.exp(self.x*parameters[0])
            self.h = np.max(f) * (10**(-9))
            
        if (parameters[2]=='distmod'):
           cosmo = w0waCDM(H0=70, Om0=0.3, Ode0=0.7, w0 = parameters[0], wa = parameters[1])
           f = cosmo.distmod(self.x).value
           self.h = np.max(f) * (10**-8)
           
        return f
    def generate_data(self, low_x, high_x, N, sig, *parameters):
        '''
        Generates new data for our instance ; it changes the instance variable ;
        x, y, sigma. 

        Parameters
        ----------
        low_x : numerical value
            The lower limit for the array of x.
            
        high_x : numerical value
            The upper limit for the array of x.
    
        N : integersquareroot
            Number of data points generated in the range of 
            [low_x, high_x]self.h = np.max(np.sqrt(parameters[0]*self.x + parameters[1]))* (10**(-9))
            
        sigma : numerical value.
            The percent of the function taken as an uncertainty on each entry of y.
            
       *parameters : tuple of differents types of entries. 
           see the definition in function()
        '''
        np.random.seed(10)
        self.x = np.linspace(low_x, high_x, N)
        self.sigma = sig * self.function(*parameters)

        self.y = np.random.normal(self.function(*parameters), sig)

    def xi_square(self, *parameters):
        '''
        Calculate Xi_square for a data set of value x and y and a function. 

        Parameters
        ----------
        *parameters : tuple of differents types of entries. 
            see the definition in function()

        Returns
        -------
        X : numerical value 
            The Xi_square value. 

        '''
        #Numerical calculations for each entry
        #X = np.sum(((self.y-self.function(*parameters))**2)/self.sigma**2)
        diag_sigma = np.linalg.inv(np.diag(self.sigma**2))
        f = self.y -self.function(*parameters)
        X_mat = np.matmul(f * self.sigma**-2, f) #Matrix calculation of Xisquare
        return X_mat
        
    def updating_a_parameter(self, i_par, i_list, *parameters):
        '''
        Updates a parameter by an small amount and calculates the Xisquare 
        for this change for the change. Does this following a sequence. 

        Parameters
        ----------
        i_par : integer
            the index of the parameter to be updated.
            
        i_list : list of numerical values. 
            How much at each interation we modify the paramater by h. 
            
        *parameters : tuple of differents types of entries. 
              see the definition in function()

        Returns
        -------
        D : Array of numerical values
            Array of all the Xi_square calculated for each iterations, then used
            for our fisher calculations. 

        '''
        D = np.zeros(len(i_list))
        for i in range(len(i_list)):
            parameters = list(parameters)
            parameters[i_par] += i_list[i]*(self.h) 
            parameters = tuple(parameters)
            D[i] = self.xi_square(*parameters) 
        return D
    
    
    
    def updating_two_parameters(self, i_par, j_par,i_list, j_list, *parameters):
        '''
        Updates two parameter by an small amount and calculates the Xisquare 
        for this change for the change. Does this following a sequece. 

        Parameters
        ----------
        i_par : integer
            the index of the first parameter to be updated.
            
        j_par : integer
            the index of the second parameter to be updated.
            
        i_list : list of numerical values. 
            How much at each interation we modify the first paramater by h. 
            
        j_list : list of numerical values. 
            How much at each interation we modify the second paramater by h. 
            
        *parameters : tuple of differents types of entries. 
              see the definition in function()

        Returns
        -------
        D : Array of numerical values
            Array of all the Xi_square calculated for each iterations, then used
            for our fisher calculations. 

        '''
        D = np.zeros(len(i_list))
        for i in range(len(i_list)):
            parameters = list(parameters)
            parameters[i_par] += i_list[i]*self.h
            parameters[j_par] += j_list[i]*self.h
            parameters = tuple(parameters)
            D[i] = self.xi_square(*parameters) 
        return D


    def diff_Xi2_twice(self, i_par, *parameters):
        '''
        Calculate the double derivative of Xi_square. 

        Parameters
        ----------
        i_par : integer
            the index of the first parameter to be updated.
            
        *parameters : tuple of differents types of entries. 
              see the definition in function()

        Returns
        -------
        Diff : numerical value
            double derivative value  of Xi_square

        '''
        i_list = [+1, -1, -1]
        D = self.updating_a_parameter(i_par, i_list, *parameters)
        Diff = (D[0]-2*D[1]+D[2])/(self.h**2)
        return Diff

    def diff_Xi2_didj(self, i_par, j_par, *parameters):
        '''
        Calculate the cross derivative of Xi_square. 

        Parameters
        ----------
        i_par :  integer
            the index of the first parameter to be updated.
            
        j_par :  integer
            the index of the second parameter to be updated.
            
        *parameters : tuple of differents types of entries. 
              see the definition in function()

        Returns
        -------
        Diff : numerical value
            cross derivative value of Xi_square

        '''
        i_list = [+1, -2, 0, +2]
        j_list = [+1, -2, +2, -2]
        D = self.updating_two_parameters(i_par, j_par, i_list, j_list, *parameters)
        Diff = (D[0]+D[1]-D[2]-D[3])/(4*(self.h**2))
        return Diff
    
    def fisher(self, *parameters):
        '''
        Calculate the fisher function. 

        Parameters
        ----------
        *parameters : tuple of differents types of entries. 
              see the definition in function()

        Returns
        -------
        F : numpy.array of size [n_param x n_param] 
            The Fisher matrix. 
        '''
        n_param = 2
        F = np.zeros([n_param,n_param])
        for i in range(n_param):
            for j in range(n_param):
                if (i == j): 
                    F[i,j] = 0.5*self.diff_Xi2_twice(i, *parameters)
                else:
                    F[i, j] = 0.5*self.diff_Xi2_didj(i, j, *parameters)
        return F
    
    def cov_fisher(self, *parameters):
        '''
        Calculate the covariance matrix from the fisher matrix

        Parameters
        ----------
        *parameters : tuple of differents types of entries. 
              see the definition in function()

        Returns
        -------
        covariance_matrix : numpy.array of size [n_param x n_param] 
            The covariance matrix. 

        '''
        covariance_matrix = np.mat(self.fisher(*parameters)).I
        return covariance_matrix
    
    
    
    
    
    def uncertainty_matrix(self, *parameters):
        '''
        Calculate the uncertainty matrix from the fisher matrix

        Parameters
        ----------
        *parameters : tuple of differents types of entries. 
              see the definition in function()

        Returns
        -------
        uncertainty_matrix : numpy.array of size [n_param x n_param] 
            The uncertainty matrix. 

        '''
        uncertainty_matrix = np.sqrt(self.cov_fisher(*parameters))
        return uncertainty_matrix
    

def Linear(x, a, b):
    # res = x*a+b #linear
    #res2 = np.sqrt(x*a + b) #squareroot
    #res3 = b*np.exp(a*x) #expo
    
    cosmo = w0waCDM(H0=70, Om0=0.3, Ode0=0.7, w0 = a, wa = b)
    res4 = cosmo.distmod(x).value
    return res4


def compare(low_x, high_x, N, a_max, b, function):
    Relat = [] 
    delta_fisher = []
    delta_fit = []
    a_range = np.arange(1, a_max)
    
    for i in a_range:
        B = Fit(0, 0, 0)
        B.generate_data(low_x, high_x, N, 0.01, i, b, function)
        F = B.fisher(i ,b, function)
        popt, pcov = curve_fit(Linear, B.x, B.y, sigma = B.sigma, absolute_sigma=True)
        delta_fisher.append(np.sqrt(np.array(np.mat(F).I)[0][0]))
        delta_fit.append(np.sqrt(pcov[0][0]))
        Relat.append(100*(np.sqrt(np.array(np.mat(F).I)[0][0])-np.sqrt(pcov[0][0]))/np.sqrt(pcov[0][0]))
        
    if (function=='linear'):
        fu = r'ax+1'
    if (function=='squareroot'):
        fu = r'$\sqrt{ax+1}$'
    if (function=='expo'):
        fu = r'$e^{ax}$'
    plt.figure(figsize=(16, 10))
    plt.suptitle('Uncertainties against the parameter a for f(x) = {} \n Number of data points : {}'.format(fu, N), fontsize = 20)
    plt.subplot(1, 2, 1)
    plt.plot(a_range, delta_fisher, 'x', label='Fisher uncertainty')
    plt.plot(a_range, delta_fit, label='curve_fit uncertainty')
    
    plt.xlabel('Parameter a')
    plt.ylabel(r'$\Delta(x)$')
    plt.legend(fontsize=20)
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(a_range, Relat, label=r'$\frac{\Delta(Fisher) -\Delta(fit)}{\Delta(fit)}$')
    plt.xlabel('Parameter a')
    plt.ylabel('Relative Uncertainty (in %)')
    
    plt.legend(fontsize=20)
    return delta_fisher, delta_fit

#If one wants to use this function, the function Linear at line 345 must be changed accordingly
#so the curve fit has the same expression to fit as the fisher method. 
#compare(1, 2, 200, 10, 1, 'distmod')
