# Run this with:
#
#   > python multinest_analysis.py
#
# or for parallel implementation
#
#   > mpiexec -n 8 python multinest_analysis.py
# Needed libraries

import numpy as np

import pymultinest
import corner
import json
import os
from chainconsumer import ChainConsumer
from scipy import stats
import matplotlib.pyplot as plt

import time

start = time.time()

# Custom functions
#{{{
# # Custom Functions

def log_prob(x, ndim = 3, nparams = 3):
    '''
    This function computes the logarithmic posterior probability corresponding to a point
      in the parameters space that we are analyzing.
      
    Parameters:
    -----------
        x: Np array.
            Contain the values of a point in the parameter space.
            x[0] = DM mass [GeV]
            x[1] = Cross-section [cm^2]
            x[2] = theta 
            
        forwardModel: Function, optional
            forwardModel function that takes as an argument x and returns the observable.  
            
        log_lik: Function, optional
            log_lik function that takes as an argument x and returns the logarithm of the
                likelihood.    
                
        prior: Function, optional
            Prior function that takes as an argument x and returns the prior
                value.
    Returns:
    --------
        float
            Logarithmic posterior probability.
    '''
    # Initialization
    # Let's compute the observable corresponding to model parameters x
    th_observable = forwardModel(x)

    # Let's compute the log likelihood of the observable
    log_prob_val = log_lik(th_observable, real_observable)

    return log_prob_val


def forwardModel(x, which = 'rate'):
    '''
    This function computes the observable corresponding to a point
      in the parameters space that we are analyzing. 
      HERE WILL BE XENON API
      
    Parameters:
    -----------
        x: Np array.
            Contain the values of a point in the parameter space.
            x[0] = DM mass [GeV]
            x[1] = Cross-section [cm^2]
            x[2] = theta 
            
    Returns:
    --------
        Np array
            Theoretical Observable.
    '''

    s1_bck = np.random.uniform(1, 100, 3000)
    s2_bck = np.log(s1_bck) + np.random.normal(0,0.5,3000)

    nsig = int(x[1] * x[2] * 1e50) # Toy model

    if nsig > 0:
        s1_dm = np.random.uniform(1, 10 * x[0], nsig)
        s2_dm = np.log(s1_dm) + np.random.normal(0,0.5,nsig)

        s1_tot = np.hstack((s1_bck, s1_dm))
        s2_tot = np.hstack((s2_bck, s2_dm))
    else:
        s1_tot = s1_bck
        s2_tot = s2_bck
    
    if which == 'rate':
        th_observable = 3000 + nsig
    elif which == 'drate':
        th_observable, _ = np.histogram(s1_tot, bins = 30)
    elif which == 's1s2':
        th_observable,_,_ = np.histogram2d(s1_tot, s2_tot, bins = 35)

    return th_observable


def log_lik(th_observable, real_observable): 
    '''
    This function computes the logarithmic likelihood ratio between 
     the theoretical and real observable.
      
    Parameters:
    -----------
        th_observable: Np array
            Theoretical observable of a x point in the parameter space
            
        real_observable: Np array
            "Real" observable of the benchmark point
    Returns:
    --------
        float
            Logarithmic likelihood.
    '''

    log_lik = np.sqrt(np.mean( (th_observable - real_observable)**2 )) # Just a chi-square
 
    return - log_lik/2

def log_prior(x, ndim = 3, nparams = 3):
    '''
    This function computes the logarithm of the prior corresponding to a point
      in the parameters space that we are analyzing.
      
    Parameters:
    -----------
        x: Np array.
            Contain the values of a point in the parameter space.
            x[0] = Log10 of DM mass [GeV]
            x[1] = Log10 of Cross-section [cm^2]
            x[2] = theta 

    Returns:
    --------
        float
            Prior probability
    '''
    x[0] = 10**(2 * x[0] + 1)  # log-Uniform prior between 1e and 1e3
    x[1] = 10**(4 * x[1] - 49) # log-Uniform prior between 1e-49 and 1e-45
    x[2] = 3 * x[2] - 1.5      # Uniform prior between -1.5 and 1.5
    

#}}}

# Let's choose the benchmark point
m_dm  = 50 # m_{DM} []
sigma = 1e-47 # sigma [cm^2]
theta = 0.1

real_observable = forwardModel([m_dm, sigma, theta])

folder = '../data/MultinestAnalysis/BP1'

try:
    os.mkdir(folder)
except:
    pass

Name = folder + '/chains100_'
 
pymultinest.run(log_prob, Prior = log_prior, n_dims = 3, verbose = True, outputfiles_basename= Name, n_live_points = 100)

parameters = [r'$m_{DM}$', r'$\sigma$', r'$\theta$']
n_params = len(parameters)

json.dump(parameters, open(Name + 'params.json', 'w')) # save parameter names

a = pymultinest.Analyzer(outputfiles_basename=Name, n_params = n_params)

data     = a.get_data()[:,2:]
_2loglik = a.get_data()[:,1] # -2LogLik = -2*log_prob(data)
weights  = a.get_data()[:,0]

mask = weights > 1e-4

# Corner plots
#{{{
if len(np.where(mask == True)[0]) > 100: 
    corner.corner(data[mask,:], weights=weights[mask],
                  truths=[m_dm, sigma, theta],
                  range = [(1e1, 1e3),(1e-49, 1e-45),(-1.6, 1.6)],
                  labels=parameters, show_titles=True)
    plt.savefig(folder + 'multinest_2dposterior.pdf')
    plt.clf()
#}}}

# ChainConsumer plots
#{{{
truth = [m_dm, sigma, theta]

chain = ChainConsumer ()

chain.add_chain(chain = data, parameters = parameters, weights = weights)

chain.configure(kde = 1.5,
                colors = ["#1E88E5", "#D32F2F", "#111111"],
                linestyles = ["-", "-", "-"],
                sigmas=[1,2],
                sigma2d=True,
                shade = [True, True, True])

chain.analysis.get_summary ()
chain.plotter.plot(figsize = (10,10), 
                   log_scales = True,
                   #extents = [(1e1, 1e3), (1e-49, 1e-45), (-1.6, 1.6)],
                   filename = folder + 'multinest_posterior2.pdf',
                   truth = truth)

plt.clf()
#}}}

# LogLik plots
#{{{

plt.scatter(data[:,0], data[:,2], c = np.log10(_2loglik))
plt.colorbar()

ind = np.where(_2loglik < 6.18)[0]
plt.scatter(data[ind,0], data[ind,2], c = np.log10(_2loglik[ind]), cmap = 'Greys')

ind = np.where(_2loglik < 2.3)[0]
plt.scatter(data[ind,0], data[ind,2], c = np.log10(_2loglik[ind]), cmap = 'Reds')

plt.axhline(y = sigma, linestyle = ':', c = 'black')
plt.axvline(x = m_dm, linestyle = ':', c = 'black')
plt.scatter(m_dm, sigma, marker = '*', c = 'red')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e1, 1e3)
plt.xlabel(r'$m_{DM}$ [GeV]')
plt.ylabel(r'$\sigma [cm^{2}]$')
plt.savefig(folder + 'LogLik_BP1.pdf')
plt.clf()
#}}}

end = time.time()
print('Multinest take ', end - start, ' s')
