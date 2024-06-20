# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import swyft
from tqdm import tqdm
import time
from scipy import stats
import seaborn as sbn
import pandas as pd
#import h5py
import os
from scipy.ndimage import gaussian_filter
from scipy.integrate import trapezoid
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.integrate import simps
from matplotlib.pyplot import contour, show
from matplotlib.lines import Line2D
import emcee
from chainconsumer import ChainConsumer
import chainconsumer
import pymultinest
import corner
import bilby


import torch
import torchist
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import Callback

import seaborn as sns

torch.set_float32_matmul_precision('high')
pallete = np.flip(sns.color_palette("tab20c", 8), axis = 0)
cross_sec_th = -49

long_planck = 1.616199 * 1e-35 * 1e2 # cm
masa_planck = 2.435 * 1e18 # GeV
fac = (long_planck * masa_planck) / 1e6

# +
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configuration
sender_email = 'martindelosrios13@gmail.com'
app_password = 'ukgl cvyy glqk woki'  # Use the app password you generated
recipient_email = 'martindelosrios13@gmail.com'
subject = 'Termino'
message = 'Termino de correr'

# Connect to the SMTP server
smtp_server = 'smtp.gmail.com'
smtp_port = 587

# +
#from playsound import playsound
#playsound('/home/martinrios/Downloads/mario.mp3')

# +
#from torchsummary import summary
# -

# It is usefull to print the versions of the package that we are using
print('swyft version:', swyft.__version__)
print('numpy version:', np.__version__)
print('matplotlib version:', mpl.__version__)
print('torch version:', torch.__version__)

color_rate = "#d55e00"
color_drate = "#0072b2"
color_s1s2 = "#009e73"

# Check if gpu is available
if torch.cuda.is_available():
    device = 'gpu'
    print('Using GPU')
else:
    device = 'cpu'
    print('Using CPU')


# # Custom functions

def read_slice(datFolder):
    nobs_slices = 0
    for i, folder in enumerate(datFolder):
        print('Reading data from ' + folder)
        if i == 0:
            pars_slices      = np.loadtxt(folder + 'pars.txt') # pars[:,0] = mass ; pars[:,1] = cross-section ; pars[:,2] = theta
            rate_raw_slices  = np.loadtxt(folder + 'rate.txt') # rate[:,0] = total expected events ; rate[:,1] = expected signal ; rate[:,2] = # events pseudo-experiment ; rate[:,3] = # signal events pseudo-experiment 
            
            diff_rate_WIMP     = np.loadtxt(folder + 'diff_rate_WIMP.txt')
            diff_rate_er       = np.loadtxt(folder + 'diff_rate_er.txt')
            diff_rate_ac       = np.loadtxt(folder + 'diff_rate_ac.txt')
            diff_rate_cevns_SM = np.loadtxt(folder + 'diff_rate_CEVNS-SM.txt')
            diff_rate_radio    = np.loadtxt(folder + 'diff_rate_radiogenics.txt')
            diff_rate_wall     = np.loadtxt(folder + 'diff_rate_wall.txt')
            
            s1s2_WIMP_slices     = np.loadtxt(folder + 's1s2_WIMP.txt')
            s1s2_er_slices       = np.loadtxt(folder + 's1s2_er.txt')
            s1s2_ac_slices       = np.loadtxt(folder + 's1s2_ac.txt')
            s1s2_cevns_SM_slices = np.loadtxt(folder + 's1s2_CEVNS-SM.txt')
            s1s2_radio_slices    = np.loadtxt(folder + 's1s2_radiogenics.txt')
            s1s2_wall_slices     = np.loadtxt(folder + 's1s2_wall.txt')
        else:
            pars_slices      = np.vstack((pars_slices, np.loadtxt(folder + 'pars.txt'))) # pars[:,0] = mass ; pars[:,1] = cross-section ; pars[:,2] = theta
            rate_raw_slices  = np.vstack((rate_raw_slices, np.loadtxt(folder + 'rate.txt'))) # rate[:,0] = total expected events ; rate[:,1] = expected signal ; rate[:,2] = # events pseudo-experiment ; rate[:,3] = # signal events pseudo-experiment 
            
            diff_rate_WIMP     = np.vstack((diff_rate_WIMP, np.loadtxt(folder + 'diff_rate_WIMP.txt') ))
            diff_rate_er       = np.vstack((diff_rate_er, np.loadtxt(folder + 'diff_rate_er.txt') ))
            diff_rate_ac       = np.vstack((diff_rate_ac, np.loadtxt(folder + 'diff_rate_ac.txt') ))
            diff_rate_cevns_SM = np.vstack((diff_rate_cevns_SM, np.loadtxt(folder + 'diff_rate_CEVNS-SM.txt') ))
            diff_rate_radio    = np.vstack((diff_rate_radio, np.loadtxt(folder + 'diff_rate_radiogenics.txt') ))
            diff_rate_wall     = np.vstack((diff_rate_wall, np.loadtxt(folder + 'diff_rate_wall.txt') ))
            
            s1s2_WIMP_slices     = np.vstack((s1s2_WIMP_slices, np.loadtxt(folder + 's1s2_WIMP.txt')))
            s1s2_er_slices       = np.vstack((s1s2_er_slices, np.loadtxt(folder + 's1s2_er.txt')))
            s1s2_ac_slices       = np.vstack((s1s2_ac_slices, np.loadtxt(folder + 's1s2_ac.txt')))
            s1s2_cevns_SM_slices = np.vstack((s1s2_cevns_SM_slices, np.loadtxt(folder + 's1s2_CEVNS-SM.txt')))
            s1s2_radio_slices    = np.vstack((s1s2_radio_slices, np.loadtxt(folder + 's1s2_radiogenics.txt')))
            s1s2_wall_slices     = np.vstack((s1s2_wall_slices, np.loadtxt(folder + 's1s2_wall.txt')))
            
        
    nobs_slices = len(pars_slices) # Total number of observations
    print('We have ' + str(nobs_slices) + ' observations...')
    
    s1s2_slices = s1s2_WIMP_slices + s1s2_er_slices + s1s2_ac_slices + s1s2_cevns_SM_slices + s1s2_radio_slices + s1s2_wall_slices
    rate_slices = np.sum(s1s2_slices, axis = 1) # Just to have the same as on the other notebooks. This already includes the backgrounds
    s1s2_slices = s1s2_slices.reshape(nobs_slices, 97, 97)

    diff_rate_slices = diff_rate_WIMP + diff_rate_er + diff_rate_ac + diff_rate_cevns_SM + diff_rate_radio + diff_rate_wall
    
    # Let's work with the log of the mass and cross-section
    
    pars_slices[:,0] = np.log10(pars_slices[:,0])
    pars_slices[:,1] = np.log10(pars_slices[:,1])
    
    return pars_slices, rate_slices, diff_rate_slices, s1s2_slices


def plot_1dpost(x, h1, ax, low_1sigma = None, up_1sigma = None, alpha = 1, color = 'black', real_val = True):
    ax.plot(x, h1, c = color, alpha = alpha)
    if real_val: ax.axvline(x = pars_true[1], c = 'orange')
    ax.axvline(x = -42, c = 'black', linewidth = 2)

    if (low_1sigma is not None) & (up_1sigma is not None):
        ax.axvline(low_1sigma, c = 'black', linestyle = '--')
        ax.axvline(up_1sigma, c = 'black', linestyle = '--')
    
    #ax.axvline(low_2sigma, c = 'black', linestyle = '--')
    #ax.axvline(up_2sigma, c = 'black', linestyle = '--')
    
    #ax.axvline(low_3sigma, c = 'black', linestyle = ':')
    #ax.axvline(up_3sigma, c = 'black', linestyle = ':')

    ax.set_xlim(-41, -36)
    #ax.xscale('log')
    ax.set_xlabel('$log(\sigma)$')
    ax.set_ylabel('$P(\sigma|x)$')
    return ax


def email(message = 'termino'):
    # Create a MIMEText object to represent the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    # Log in to your email account with the app password
    server.login(sender_email, app_password)
    
    # Send the email
    server.sendmail(sender_email, recipient_email, msg.as_string())
    
    # Close the connection
    server.quit()
    return None


def plot1d(ax, predictions, pars_true, par = 1, 
           xlabel = '$\log_{10}(\sigma)$', ylabel = '$P(\sigma|x)\ /\ P(\sigma)$',
           flip = False, fill = True, linestyle = 'solid', color = 'black', fac = 1):
    # Let's put the results in arrays
    parameter = np.asarray(predictions[0].params[:,par,0]) * (pars_max[par] - pars_min[par]) + pars_min[par]
    ratios = np.exp(np.asarray(predictions[0].logratios[:,par]))
    
    ind_sort  = np.argsort(parameter)
    ratios    = ratios[ind_sort]
    parameter = parameter[ind_sort]
    
    # Let's compute the integrated probability for different threshold
    cuts = np.linspace(np.min(ratios), np.max(ratios), 100)
    integrals = []
    for c in cuts:
        ratios0 = np.copy(ratios)
        ratios0[np.where(ratios < c)[0]] = 0 
        integrals.append( trapezoid(ratios0, parameter) / trapezoid(ratios, parameter) )
        
    integrals = np.asarray(integrals)
    
    # Let's compute the thresholds corresponding to 0.9 and 0.95 integrated prob
    cut90 = cuts[np.argmin( np.abs(integrals - 0.9))]
    cut95 = cuts[np.argmin( np.abs(integrals - 0.95))]

    if not flip:
        ax.plot(10**parameter, fac * ratios, c = color, linestyle = linestyle)
        if fill:
            ind = np.where(ratios > cut90)[0]
            ax.fill_between(10**parameter[ind], fac * ratios[ind], [0] * len(ind), color = 'darkcyan', alpha = 0.3)
            ind = np.where(ratios > cut95)[0]
            ax.fill_between(10**parameter[ind], fac * ratios[ind], [0] * len(ind), color = 'darkcyan', alpha = 0.5)
        ax.axvline(x = 10**(pars_true[par] * (pars_max[par] - pars_min[par]) + pars_min[par]), color = 'black')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
    else:
        ax.plot(fac * ratios, 10**parameter, c = color, linestyle = linestyle)
        if fill:
            ind = np.where(ratios > cut90)[0]
            ax.fill_betweenx(10**parameter[ind], [0] * len(ind), fac * ratios[ind], color = 'darkcyan', alpha = 0.3)
            ind = np.where(ratios > cut95)[0]
            ax.fill_betweenx(10**parameter[ind], [0] * len(ind), fac * ratios[ind], color = 'darkcyan', alpha = 0.5) 
        ax.axhline(y = 10**(pars_true[par] * (pars_max[par] - pars_min[par]) + pars_min[par]), color = 'black')
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        #ax.set_xlim(-0.1,8)
        ax.set_ylim(1e-41, 1e-36)
        ax.set_yscale('log')
        
    return ax


def plot2d(ax, predictions, pars_true, fill = True, line = False, linestyle = 'solid', color = 'black'):      
    results_pars = np.asarray(predictions[1].params)
    results      = np.asarray(predictions[1].logratios)
    
    # Let's make an interpolation function 
    interp = CloughTocher2DInterpolator(results_pars[:,0,:], np.exp(results[:,0]))
    
    def interpol(log_m, log_sigma):
        m_norm = (log_m - pars_min[0]) / (pars_max[0] - pars_min[0])
        sigma_norm = (log_sigma - pars_min[1]) / (pars_max[1] - pars_min[1])
        return interp(m_norm, sigma_norm)
        
    # Let's estimate the value of the posterior in a grid
    nvals = 20
    m_values = np.logspace(0.8, 2.99, nvals)
    s_values = np.logspace(-42., -36, nvals)
    m_grid, s_grid = np.meshgrid(m_values, s_values)
    
    ds = np.log10(s_values[1]) - np.log10(s_values[0])
    dm = np.log10(m_values[1]) - np.log10(m_values[0])
    
    res = np.zeros((nvals, nvals))
    for m in range(nvals):
        for s in range(nvals):
            res[m,s] = interpol(np.log10(m_values[m]), np.log10(s_values[s]))
    res[np.isnan(res)] = 0
    #print(res)
    # Let's compute the integral
    norm = simps(simps(res, dx=dm, axis=1), dx=ds)
    #print(norm)
    
    # Let's look for the 0.9 probability threshold
    cuts = np.linspace(np.min(res), np.max(res), 100)
    integrals = []
    for c in cuts:
        res0 = np.copy(res)
        res0[np.where(res < c)[0], np.where(res < c)[1]] = 0
        integrals.append( simps(simps(res0, dx=dm, axis=1), dx=ds) / norm )
    integrals = np.asarray(integrals)
    
    cut90 = cuts[np.argmin( np.abs(integrals - 0.9))]
    cut95 = cuts[np.argmin( np.abs(integrals - 0.95))]
    #print(cut)
    if fill:
        ax.contourf(m_values, s_values, res.T, levels = [0, cut90, np.max(res)], colors = ['white','darkcyan'], alpha = 0.3, linestyles = ['solid'])
        ax.contourf(m_values, s_values, res.T, levels = [0, cut95, np.max(res)], colors = ['white','darkcyan'], alpha = 0.5, linestyles = ['solid'])
    if line:
        ax.contour(m_values, s_values, res.T, levels = [0,cut90], colors = [color], linestyles = ['solid'])
        ax.contour(m_values, s_values, res.T, levels = [0,cut95], colors = [color], linestyles = ['--'])
    
    ax.axvline(x = 10**(pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0]), color = 'black')
    ax.axhline(y = 10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1]), color = 'black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$M_{DM}$ [GeV]')
    ax.set_ylabel('$\sigma$ $[cm^{2}]$')

    return ax


# +
def plot1d_comb(ax, predictions, pars_true, par = 1, 
           xlabel = '$\log_{10}(\sigma)$', ylabel = '$P(\sigma|x)\ /\ P(\sigma)$',
           flip = False, fill = True, linestyle = 'solid', color = 'black', fac = 1):
    # Let's put the results in arrays
    parameter = np.asarray(predictions[0][0].params[:,par,0]) * (pars_max[par] - pars_min[par]) + pars_min[par]
    ratios = np.zeros_like(predictions[0][0].logratios[:,par])
    for pred in predictions:
        ratios = ratios + np.asarray(pred[0].logratios[:,par])
    ratios = np.exp(ratios)
    
    ind_sort  = np.argsort(parameter)
    ratios    = ratios[ind_sort]
    parameter = parameter[ind_sort]
    
    # Let's compute the integrated probability for different threshold
    cuts = np.linspace(np.min(ratios), np.max(ratios), 100)
    integrals = []
    for c in cuts:
        ratios0 = np.copy(ratios)
        ratios0[np.where(ratios < c)[0]] = 0 
        integrals.append( trapezoid(ratios0, parameter) / trapezoid(ratios, parameter) )
        
    integrals = np.asarray(integrals)
    
    # Let's compute the thresholds corresponding to 0.9 and 0.95 integrated prob
    cut90 = cuts[np.argmin( np.abs(integrals - 0.9))]
    cut95 = cuts[np.argmin( np.abs(integrals - 0.95))]

    if not flip:
        ax.plot(10**parameter, fac * ratios, c = color, linestyle = linestyle)
        if fill:
            ind = np.where(ratios > cut90)[0]
            ax.fill_between(10**parameter[ind], fac * ratios[ind], [0] * len(ind), color = 'darkcyan', alpha = 0.3)
            ind = np.where(ratios > cut95)[0]
            ax.fill_between(10**parameter[ind], fac * ratios[ind], [0] * len(ind), color = 'darkcyan', alpha = 0.5)
        ax.axvline(x = 10**(pars_true[par] * (pars_max[par] - pars_min[par]) + pars_min[par]), color = 'black')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
    else:
        ax.plot(fac * ratios, 10**parameter, c = color, linestyle = linestyle)
        if fill:
            ind = np.where(ratios > cut90)[0]
            ax.fill_betweenx(10**parameter[ind], [0] * len(ind), fac * ratios[ind], color = 'darkcyan', alpha = 0.3)
            ind = np.where(ratios > cut95)[0]
            ax.fill_betweenx(10**parameter[ind], [0] * len(ind), fac * ratios[ind], color = 'darkcyan', alpha = 0.5) 
        ax.axhline(y = 10**(pars_true[par] * (pars_max[par] - pars_min[par]) + pars_min[par]), color = 'black')
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        #ax.set_xlim(-0.1,8)
        ax.set_ylim(1e-41, 1e-36)
        ax.set_yscale('log')
        
    return ax


def plot2d_comb(ax, predictions, pars_true, fill = True, line = False, linestyle = 'solid', color = 'black'):    
    
    results_pars = np.asarray(predictions[0][1].params)
    results = np.zeros_like(predictions[0][1].logratios)
    for pred in predictions:
        results = results + np.asarray(pred[1].logratios)
    
    # Let's make an interpolation function 
    interp = CloughTocher2DInterpolator(results_pars[:,0,:], np.exp(results[:,0]))
    
    def interpol(log_m, log_sigma):
        m_norm = (log_m - pars_min[0]) / (pars_max[0] - pars_min[0])
        sigma_norm = (log_sigma - pars_min[1]) / (pars_max[1] - pars_min[1])
        return interp(m_norm, sigma_norm)
        
    # Let's estimate the value of the posterior in a grid
    nvals = 20
    m_values = np.logspace(0.8, 2.99, nvals)
    s_values = np.logspace(-42., -36, nvals)
    m_grid, s_grid = np.meshgrid(m_values, s_values)
    
    ds = np.log10(s_values[1]) - np.log10(s_values[0])
    dm = np.log10(m_values[1]) - np.log10(m_values[0])
    
    res = np.zeros((nvals, nvals))
    for m in range(nvals):
        for s in range(nvals):
            res[m,s] = interpol(np.log10(m_values[m]), np.log10(s_values[s]))
    res[np.isnan(res)] = 0
    # Let's compute the integral
    norm = simps(simps(res, dx=dm, axis=1), dx=ds)
    
    # Let's look for the 0.9 probability threshold
    cuts = np.linspace(np.min(res), np.max(res), 100)
    integrals = []
    for c in cuts:
        res0 = np.copy(res)
        res0[np.where(res < c)[0], np.where(res < c)[1]] = 0
        integrals.append( simps(simps(res0, dx=dm, axis=1), dx=ds) / norm )
    integrals = np.asarray(integrals)
    
    cut90 = cuts[np.argmin( np.abs(integrals - 0.9))]
    cut95 = cuts[np.argmin( np.abs(integrals - 0.95))]
    if fill:
        ax.contourf(m_values, s_values, res.T, levels = [0, cut90, np.max(res)], colors = ['white','darkcyan'], alpha = 0.3, linestyles = ['solid'])
        ax.contourf(m_values, s_values, res.T, levels = [0, cut95, np.max(res)], colors = ['white','darkcyan'], alpha = 0.5, linestyles = ['solid'])
    if line:
        ax.contour(m_values, s_values, res.T, levels = [0,cut90], colors = [color], linestyles = ['solid'])
        ax.contour(m_values, s_values, res.T, levels = [0,cut95], colors = [color], linestyles = ['--'])
    
    ax.axvline(x = 10**(pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0]), color = 'black')
    ax.axhline(y = 10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1]), color = 'black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$M_{DM}$ [GeV]')
    ax.set_ylabel('$\sigma$ $[cm^{2}]$')

    return ax


# +
def plot1d_emcee(ax, predictions, pars_true, par = 1, 
           xlabel = '$\log_{10}(\sigma)$', ylabel = '$P(\sigma|x)\ /\ P(\sigma)$',
           flip = False, fill = True, color = 'black', fac = 1, probs = [0.9, 0.95], linestyles = ['solid', '--']):
               
    # Let's put the results in arrays
    parameter = np.asarray(predictions[0][0].params[:,par,0]) * (pars_max[par] - pars_min[par]) + pars_min[par]
    ratios = np.zeros_like(predictions[0][0].logratios[:,par])
    for pred in predictions:
        ratios = ratios + np.asarray(pred[0].logratios[:,par])
    ratios = np.exp(ratios)
    
    ind_sort  = np.argsort(parameter)
    ratios    = ratios[ind_sort]
    parameter = parameter[ind_sort]
    
    # Let's compute the integrated probability for different threshold
    cuts = np.linspace(np.min(ratios), np.max(ratios), 100)
    integrals = []
    for c in cuts:
        ratios0 = np.copy(ratios)
        ratios0[np.where(ratios < c)[0]] = 0 
        integrals.append( trapezoid(ratios0, parameter) / trapezoid(ratios, parameter) )
        
    integrals = np.asarray(integrals)
    
    # Let's compute the thresholds corresponding to 0.9 and 0.95 integrated prob
    cut90 = cuts[np.argmin( np.abs(integrals - 0.9))]
    cut95 = cuts[np.argmin( np.abs(integrals - 0.95))]
    for iprob, prob in enumerate(probs):
        cut_p = cuts[np.argmin( np.abs(integrals - prob))]
        
        if not flip:
            ax.plot(parameter, fac * ratios, c = color, linestyle = linestyles[iprob])
            if fill:
                ind = np.where(ratios > cut_p)[0]
                pars_aux = parameter[ind]
                ratios_aux = ratios[ind]
                ind = np.argmin(pars_aux)
                ax.plot([pars_aux[ind], pars_aux[ind]], [fac * ratios_aux[ind], 0], color = 'darkcyan', alpha = 1)
                ind = np.argmax(pars_aux)
                ax.plot([pars_aux[ind], pars_aux[ind]], [fac * ratios_aux[ind], 0], color = 'darkcyan', alpha = 1)
                
                # #%ind = np.where(ratios > cut95)[0]
                # #%pars_aux = parameter[ind]
                # #%ratios_aux = ratios[ind]
                # #%ind = np.argmin(pars_aux)
                # #%ax.plot([pars_aux[ind], pars_aux[ind]], [fac * ratios_aux[ind], 0], color = 'darkcyan', alpha = 1, ls = '--')
                # #%ind = np.argmax(pars_aux)
                # #%ax.plot([pars_aux[ind], pars_aux[ind]], [fac * ratios_aux[ind], 0], color = 'darkcyan', alpha = 1, ls = '--')
                
                #ax.fill_between(parameter[ind], fac * ratios[ind], [0] * len(ind), color = 'darkcyan', alpha = 0.3)
                #ind = np.where(ratios > cut95)[0]
                #ax.fill_between(parameter[ind], fac * ratios[ind], [0] * len(ind), color = 'darkcyan', alpha = 0.5)
            ax.axvline(x = (pars_true[par] * (pars_max[par] - pars_min[par]) + pars_min[par]), color = 'black')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.plot(fac * ratios, parameter, c = color, linestyle = linestyles[iprob])
            if fill:
                ind = np.where(ratios > cut_p)[0]
                pars_aux = parameter[ind]
                ratios_aux = ratios[ind]
                ind = np.argmin(pars_aux)
                ax.plot([fac * ratios_aux[ind], 0],[pars_aux[ind], pars_aux[ind]],  color = 'darkcyan', alpha = 1)
                ind = np.argmax(pars_aux)
                ax.plot([fac * ratios_aux[ind], 0],[pars_aux[ind], pars_aux[ind]],  color = 'darkcyan', alpha = 1)
                
                # #%ind = np.where(ratios > cut95)[0]
                # #%pars_aux = parameter[ind]
                # #%ratios_aux = ratios[ind]
                # #%ind = np.argmin(pars_aux)
                # #%ax.plot([fac * ratios_aux[ind], 0],[pars_aux[ind], pars_aux[ind]],  color = 'darkcyan', alpha = 1, ls = '--')
                # #%ind = np.argmax(pars_aux)
                # #%ax.plot([fac * ratios_aux[ind], 0],[pars_aux[ind], pars_aux[ind]],  color = 'darkcyan', alpha = 1, ls = '--')
                
                #ax.fill_betweenx(parameter[ind], [0] * len(ind), fac * ratios[ind], color = 'darkcyan', alpha = 0.3)
                #ind = np.where(ratios > cut95)[0]
                #ax.fill_betweenx(parameter[ind], [0] * len(ind), fac * ratios[ind], color = 'darkcyan', alpha = 0.5) 
        ax.axhline(y = (pars_true[par] * (pars_max[par] - pars_min[par]) + pars_min[par]), color = 'black')
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        
    return ax


def plot2d_emcee(ax, predictions, pars_true, fill = True, line = False, color = 'black', probs = [0.9, 0.95], zorder = 0, linestyles = ['solid', '--'],
                nvals = 20, smooth = None):    
    
    results_pars = np.asarray(predictions[0][1].params)
    results = np.zeros_like(predictions[0][1].logratios)
    for pred in predictions:
        results = results + np.asarray(pred[1].logratios)
    
    # Let's make an interpolation function 
    interp = CloughTocher2DInterpolator(results_pars[:,0,:], np.exp(results[:,0]))
    
    def interpol(log_m, log_sigma):
        m_norm = (log_m - pars_min[0]) / (pars_max[0] - pars_min[0])
        sigma_norm = (log_sigma - pars_min[1]) / (pars_max[1] - pars_min[1])
        return interp(m_norm, sigma_norm)
        
    # Let's estimate the value of the posterior in a grid
    
    m_values = np.logspace(0.8, 2.99, nvals)
    s_values = np.logspace(-41., -36, nvals)
    m_grid, s_grid = np.meshgrid(m_values, s_values)
    
    ds = np.log10(s_values[1]) - np.log10(s_values[0])
    dm = np.log10(m_values[1]) - np.log10(m_values[0])
    
    res = np.zeros((nvals, nvals))
    for m in range(nvals):
        for s in range(nvals):
            res[m,s] = interpol(np.log10(m_values[m]), np.log10(s_values[s]))
    res[np.isnan(res)] = 0
    # Let's compute the integral
    norm = simps(simps(res, dx=dm, axis=1), dx=ds)
    
    # Let's look for the 0.9 probability threshold
    cuts = np.linspace(np.min(res), np.max(res), 100)
    integrals = []
    for c in cuts:
        res0 = np.copy(res)
        res0[np.where(res < c)[0], np.where(res < c)[1]] = 0
        integrals.append( simps(simps(res0, dx=dm, axis=1), dx=ds) / norm )
    integrals = np.asarray(integrals)
    
    cut90 = cuts[np.argmin( np.abs(integrals - 0.9))]
    cut95 = cuts[np.argmin( np.abs(integrals - 0.95))]
    if smooth is not None: res = gaussian_filter(res, smooth)
    for iprob, prob in enumerate(probs):
        cut_p = cuts[np.argmin( np.abs(integrals - prob))]
        
        if fill:
            ax.contourf(np.log10(m_values), np.log10(s_values), res.T, levels = [0, cut_p, np.max(res)], colors = ['white',color], alpha = 0.3, linestyles = linestyles, zorder = zorder)
            #ax.contourf(np.log10(m_values), np.log10(s_values), res.T, levels = [0, cut95, np.max(res)], colors = ['white','darkcyan'], alpha = 0.5, linestyles = ['solid'], zorder = zorder)
        if line:
            ax.contour(np.log10(m_values), np.log10(s_values), res.T, levels = [0, cut_p], colors = [color], linestyles = linestyles[iprob], zorder = zorder)
            #ax.contour(np.log10(m_values), np.log10(s_values), res.T, levels = [0,cut95], colors = [color], linestyles = ['--'], zorder = zorder)
    
    ax.axvline(x = (pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0]), color = 'black')
    ax.axhline(y = (pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1]), color = 'black')
    ax.set_xlabel('$M_{DM}$ [GeV]')
    ax.set_ylabel('$\sigma$ $[cm^{2}]$')

    return ax


# -

def plot2d_emcee_m_theta(ax, predictions, pars_true, fill = True, line = False, linestyle = 'solid', color = 'black', probs = [0.9, 0.95], zorder = 0, linestyles = ['solid', '--'],
                        nvals = 20, smooth = None):    
    
    results_pars = np.asarray(predictions[0][1].params)
    results = np.zeros_like(predictions[0][1].logratios)
    for pred in predictions:
        results = results + np.asarray(pred[1].logratios)
    
    # Let's make an interpolation function 
    interp = CloughTocher2DInterpolator(results_pars[:,1,:], np.exp(results[:,1]))
    
    def interpol(log_m, theta):
        m_norm = (log_m - pars_min[0]) / (pars_max[0] - pars_min[0])
        t_norm = (theta - pars_min[2]) / (pars_max[2] - pars_min[2])
        return interp(m_norm, t_norm)
        
    # Let's estimate the value of the posterior in a grid
    m_values = np.logspace(0.8, 2.99, nvals)
    t_values = np.linspace(-1.6, 1.6, nvals)
    m_grid, t_grid = np.meshgrid(m_values, t_values)
    
    dt = (t_values[1] - t_values[0])
    dm = np.log10(m_values[1]) - np.log10(m_values[0])
    
    res = np.zeros((nvals, nvals))
    for m in range(nvals):
        for t in range(nvals):
            res[m,t] = interpol(np.log10(m_values[m]), t_values[t])
    res[np.isnan(res)] = 0
    # Let's compute the integral
    norm = simps(simps(res, dx=dm, axis=1), dx=dt)
    
    # Let's look for the 0.9 probability threshold
    cuts = np.linspace(np.min(res), np.max(res), 100)
    integrals = []
    for c in cuts:
        res0 = np.copy(res)
        res0[np.where(res < c)[0], np.where(res < c)[1]] = 0
        integrals.append( simps(simps(res0, dx=dm, axis=1), dx=dt) / norm )
    integrals = np.asarray(integrals)
    
    cut90 = cuts[np.argmin( np.abs(integrals - 0.9))]
    cut95 = cuts[np.argmin( np.abs(integrals - 0.95))]
    if smooth is not None: res = gaussian_filter(res, smooth)
    for iprob, prob in enumerate(probs):
        cut_p = cuts[np.argmin( np.abs(integrals - prob))]
        
        if fill:
            ax.contourf(np.log10(m_values), t_values, res.T, levels = [0, cut_p, np.max(res)], colors = ['white',color], alpha = 0.3, linestyles = linestyles, zorder = zorder)
            #ax.contourf(np.log10(m_values), np.log10(s_values), res.T, levels = [0, cut95, np.max(res)], colors = ['white','darkcyan'], alpha = 0.5, linestyles = ['solid'], zorder = zorder)
        if line:
            ax.contour(np.log10(m_values), t_values, res.T, levels = [0, cut_p], colors = [color], linestyles = linestyles[iprob], zorder = zorder)
            #ax.contour(np.log10(m_values), np.log10(s_values), res.T, levels = [0,cut95], colors = [color], linestyles = ['--'], zorder = zorder)
    
    ax.axvline(x = (pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0]), color = 'black')
    ax.axhline(y = (pars_true[2] * (pars_max[2] - pars_min[2]) + pars_min[2]), color = 'black')
    ax.set_xlabel('$M_{DM}$ [GeV]')
    ax.set_ylabel('$\theta$')

    return ax


def plot2d_emcee_sigma_theta(ax, predictions, pars_true, fill = True, line = False, linestyle = 'solid', color = 'black', probs = [0.9, 0.95], zorder = 0, linestyles = ['solid', '--'],
                            nvals = 20, smooth = None):    
    
    results_pars = np.asarray(predictions[0][1].params)
    results = np.zeros_like(predictions[0][1].logratios)
    for pred in predictions:
        results = results + np.asarray(pred[1].logratios)
    
    # Let's make an interpolation function 
    interp = CloughTocher2DInterpolator(results_pars[:,2,:], np.exp(results[:,2]))
    
    def interpol(log_s, theta):
        s_norm = (log_s - pars_min[1]) / (pars_max[1] - pars_min[1])
        t_norm = (theta - pars_min[2]) / (pars_max[2] - pars_min[2])
        return interp(s_norm, t_norm)
        
    # Let's estimate the value of the posterior in a grid
    s_values = np.logspace(-41., -36, nvals)
    t_values = np.linspace(-1.6, 1.6, nvals)
    s_grid, t_grid = np.meshgrid(s_values, t_values)
    
    dt = (t_values[1] - t_values[0])
    ds = np.log10(s_values[1]) - np.log10(s_values[0])
    
    res = np.zeros((nvals, nvals))
    for s in range(nvals):
        for t in range(nvals):
            res[s,t] = interpol(np.log10(s_values[s]), t_values[t])
    res[np.isnan(res)] = 0
    # Let's compute the integral
    norm = simps(simps(res, dx=ds, axis=1), dx=dt)
    
    # Let's look for the 0.9 probability threshold
    cuts = np.linspace(np.min(res), np.max(res), 100)
    integrals = []
    for c in cuts:
        res0 = np.copy(res)
        res0[np.where(res < c)[0], np.where(res < c)[1]] = 0
        integrals.append( simps(simps(res0, dx=ds, axis=1), dx=dt) / norm )
    integrals = np.asarray(integrals)
    
    cut90 = cuts[np.argmin( np.abs(integrals - 0.9))]
    cut95 = cuts[np.argmin( np.abs(integrals - 0.95))]
    if smooth is not None: res = gaussian_filter(res, smooth)                                
    for iprob, prob in enumerate(probs):
        cut_p = cuts[np.argmin( np.abs(integrals - prob))]
        
        if fill:
            ax.contourf(np.log10(s_values), t_values, res.T, levels = [0, cut_p, np.max(res)], colors = ['white',color], alpha = 0.3, linestyles = linestyles[iprob], zorder = zorder)
            #ax.contourf(np.log10(m_values), np.log10(s_values), res.T, levels = [0, cut95, np.max(res)], colors = ['white','darkcyan'], alpha = 0.5, linestyles = ['solid'], zorder = zorder)
        if line:
            ax.contour(np.log10(s_values), t_values, res.T, levels = [0, cut_p], colors = [color], linestyles = linestyles[iprob], zorder = zorder)
            #ax.contour(np.log10(m_values), np.log10(s_values), res.T, levels = [0,cut95], colors = [color], linestyles = ['--'], zorder = zorder)
    
    ax.axvline(x = (pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1]), color = 'black')
    ax.axhline(y = (pars_true[2] * (pars_max[2] - pars_min[2]) + pars_min[2]), color = 'black')
    ax.set_xlabel('$\sigma$ $[cm^{2}]$')
    ax.set_ylabel('$\theta$')

    return ax


# # Let's load the data

# !ls ../data/andresData/O4-fulldata/O4/

# +
# where are your files?
datFolder = ['../data/andresData/O4-fulldata/O4/O4-run03/',
             '../data/andresData/O4-fulldata/O4/O4-run04/']
nobs = 0
for i, folder in enumerate(datFolder):
    print(i)
    if i == 0:
        pars      = np.loadtxt(folder + 'pars.txt') # pars[:,0] = mass ; pars[:,1] = cross-section ; pars[:,2] = theta
        rate_raw  = np.loadtxt(folder + 'rate.txt') # rate[:,0] = total expected events ; rate[:,1] = expected signal ; rate[:,2] = # events pseudo-experiment ; rate[:,3] = # signal events pseudo-experiment 
        
        diff_rate_WIMP     = np.loadtxt(folder + 'diff_rate_WIMP.txt')
        diff_rate_er       = np.loadtxt(folder + 'diff_rate_er.txt')
        diff_rate_ac       = np.loadtxt(folder + 'diff_rate_ac.txt')
        diff_rate_cevns_SM = np.loadtxt(folder + 'diff_rate_CEVNS-SM.txt')
        diff_rate_radio    = np.loadtxt(folder + 'diff_rate_radiogenics.txt')
        diff_rate_wall     = np.loadtxt(folder + 'diff_rate_wall.txt')
        
        s1s2_WIMP     = np.loadtxt(folder + 's1s2_WIMP.txt')
        s1s2_er       = np.loadtxt(folder + 's1s2_er.txt')
        s1s2_ac       = np.loadtxt(folder + 's1s2_ac.txt')
        s1s2_cevns_SM = np.loadtxt(folder + 's1s2_CEVNS-SM.txt')
        s1s2_radio    = np.loadtxt(folder + 's1s2_radiogenics.txt')
        s1s2_wall     = np.loadtxt(folder + 's1s2_wall.txt')
    else:
        pars      = np.vstack((pars, np.loadtxt(folder + 'pars.txt'))) # pars[:,0] = mass ; pars[:,1] = cross-section ; pars[:,2] = theta
        rate_raw  = np.vstack((rate_raw, np.loadtxt(folder + 'rate.txt'))) # rate[:,0] = total expected events ; rate[:,1] = expected signal ; rate[:,2] = # events pseudo-experiment ; rate[:,3] = # signal events pseudo-experiment 
        
        diff_rate_WIMP     = np.vstack(( diff_rate_WIMP, np.loadtxt(folder + 'diff_rate_WIMP.txt')))
        diff_rate_er       = np.vstack(( diff_rate_er, np.loadtxt(folder + 'diff_rate_er.txt')))
        diff_rate_ac       = np.vstack(( diff_rate_ac, np.loadtxt(folder + 'diff_rate_ac.txt')))
        diff_rate_cevns_SM = np.vstack(( diff_rate_cevns_SM, np.loadtxt(folder + 'diff_rate_CEVNS-SM.txt')))
        diff_rate_radio    = np.vstack(( diff_rate_radio, np.loadtxt(folder + 'diff_rate_radiogenics.txt')))
        diff_rate_wall     = np.vstack(( diff_rate_wall, np.loadtxt(folder + 'diff_rate_wall.txt')))
        
        s1s2_WIMP     = np.vstack((s1s2_WIMP, np.loadtxt(folder + 's1s2_WIMP.txt')))
        s1s2_er       = np.vstack((s1s2_er, np.loadtxt(folder + 's1s2_er.txt')))
        s1s2_ac       = np.vstack((s1s2_ac, np.loadtxt(folder + 's1s2_ac.txt')))
        s1s2_cevns_SM = np.vstack((s1s2_cevns_SM, np.loadtxt(folder + 's1s2_CEVNS-SM.txt')))
        s1s2_radio    = np.vstack((s1s2_radio, np.loadtxt(folder + 's1s2_radiogenics.txt')))
        s1s2_wall     = np.vstack((s1s2_wall, np.loadtxt(folder + 's1s2_wall.txt')))
        
    
nobs = len(pars) # Total number of observations
print('We have ' + str(nobs) + ' observations...')

diff_rate = diff_rate_WIMP + diff_rate_er + diff_rate_ac + diff_rate_cevns_SM + diff_rate_radio + diff_rate_wall

s1s2 = s1s2_WIMP + s1s2_er + s1s2_ac + s1s2_cevns_SM + s1s2_radio + s1s2_wall
rate = np.sum(s1s2, axis = 1) # Just to have the same as on the other notebooks. This already includes the backgrounds
s1s2 = s1s2.reshape(nobs, 97, 97)

# Let's work with the log of the mass and cross-section

pars[:,0] = np.log10(pars[:,0])
pars[:,1] = np.log10(pars[:,1])

# -

# This should be always zero
i = np.random.randint(nobs)
rate_raw[i,2] - rate[i]
rate_raw[i,2] - np.sum(diff_rate[i,:])

# +
###################
# shape of things #
###################
# we should get the same number of events in every file

print(pars.shape)
print(rate.shape)
print(diff_rate.shape)

# these are heavy guys:
# signal:
print(s1s2_WIMP.shape)
# backgronds:
print(s1s2_er.shape)
print(s1s2_ac.shape)
print(s1s2_cevns_SM.shape)
print(s1s2_radio.shape)
print(s1s2_wall.shape)

###############
# EXTRA FILES # backgrounds
###############
print(np.loadtxt(folder+'s1s2_CEVNS-NSI.txt').shape)
print(np.loadtxt(folder+'s1s2_EVES-NSI.txt').shape)
print(np.loadtxt(folder+'s1s2_EVES-SM.txt').shape)
# -

print(pars.shape)
print(rate.shape)
print(diff_rate.shape)
print(s1s2.shape)

# +
# Let's split in training, validation and testing

ntrain = int(70 * nobs / 100)
nval   = int(25 * nobs / 100)
ntest  = int(5 * nobs / 100)

np.random.seed(28890)
ind = np.random.choice(np.arange(nobs), size = nobs, replace = False)

train_ind = ind[:ntrain]
val_ind   = ind[ntrain:(ntrain + nval)]
test_ind  = ind[(ntrain + nval):]

pars_trainset = pars[train_ind,:]
pars_valset   = pars[val_ind,:]
pars_testset  = pars[test_ind,:]

rate_trainset = rate[train_ind]
rate_valset   = rate[val_ind]
rate_testset  = rate[test_ind]

diff_rate_trainset = diff_rate[train_ind,:]
diff_rate_valset   = diff_rate[val_ind,:]
diff_rate_testset  = diff_rate[test_ind,:]

s1s2_trainset = s1s2[train_ind,:,:]
s1s2_valset   = s1s2[val_ind,:,:]
s1s2_testset  = s1s2[test_ind,:,:]

# -

# ## Data to match emcee

# !ls ../data/andresData/O4-fulldata/O4/04-examples-to-match-emcee/examples-to-match-emcee/mDM50GeV-sigma23e-40-thetapidiv2

# +
# where are your files?
datFolder = ['../data/andresData/O4-fulldata/O4/04-examples-to-match-emcee/examples-to-match-emcee/mDM50GeV-sigma23e-40-thetapidiv2/']
emcee_nobs = 0
for i, folder in enumerate(datFolder):
    print(i)
    if i == 0:
        emcee_pars      = np.loadtxt(folder + 'pars.txt') # pars[:,0] = mass ; pars[:,1] = cross-section ; pars[:,2] = theta
        emcee_rate_raw  = np.loadtxt(folder + 'rate.txt') # rate[:,0] = total expected events ; rate[:,1] = expected signal ; rate[:,2] = # events pseudo-experiment ; rate[:,3] = # signal events pseudo-experiment 
        
        emcee_diff_rate_WIMP     = np.loadtxt(folder + 'diff_rate_WIMP.txt')
        emcee_diff_rate_er       = np.loadtxt(folder + 'diff_rate_er.txt')
        emcee_diff_rate_ac       = np.loadtxt(folder + 'diff_rate_ac.txt')
        emcee_diff_rate_cevns_SM = np.loadtxt(folder + 'diff_rate_CEVNS-SM.txt')
        emcee_diff_rate_radio    = np.loadtxt(folder + 'diff_rate_radiogenics.txt')
        emcee_diff_rate_wall     = np.loadtxt(folder + 'diff_rate_wall.txt')
        
        emcee_s1s2_WIMP     = np.loadtxt(folder + 's1s2_WIMP.txt')
        emcee_s1s2_er       = np.loadtxt(folder + 's1s2_er.txt')
        emcee_s1s2_ac       = np.loadtxt(folder + 's1s2_ac.txt')
        emcee_s1s2_cevns_SM = np.loadtxt(folder + 's1s2_CEVNS-SM.txt')
        emcee_s1s2_radio    = np.loadtxt(folder + 's1s2_radiogenics.txt')
        emcee_s1s2_wall     = np.loadtxt(folder + 's1s2_wall.txt')
    else:
        emcee_pars      = np.vstack((emcee_pars, np.loadtxt(folder + 'pars.txt'))) # pars[:,0] = mass ; pars[:,1] = cross-section ; pars[:,2] = theta
        emcee_rate_raw  = np.vstack((emcee_rate_raw, np.loadtxt(folder + 'rate.txt'))) # rate[:,0] = total expected events ; rate[:,1] = expected signal ; rate[:,2] = # events pseudo-experiment ; rate[:,3] = # signal events pseudo-experiment 
        
        emcee_diff_rate_WIMP     = np.vstack(( emcee_diff_rate_WIMP, np.loadtxt(folder + 'diff_rate_WIMP.txt')))
        emcee_diff_rate_er       = np.vstack(( emcee_diff_rate_er, np.loadtxt(folder + 'diff_rate_er.txt')))
        emcee_diff_rate_ac       = np.vstack(( emcee_diff_rate_ac, np.loadtxt(folder + 'diff_rate_ac.txt')))
        emcee_diff_rate_cevns_SM = np.vstack(( emcee_diff_rate_cevns_SM, np.loadtxt(folder + 'diff_rate_CEVNS-SM.txt')))
        emcee_diff_rate_radio    = np.vstack(( emcee_diff_rate_radio, np.loadtxt(folder + 'diff_rate_radiogenics.txt')))
        emcee_diff_rate_wall     = np.vstack(( emcee_diff_rate_wall, np.loadtxt(folder + 'diff_rate_wall.txt')))
        
        emcee_s1s2_WIMP     = np.vstack((emcee_s1s2_WIMP, np.loadtxt(folder + 's1s2_WIMP.txt')))
        emcee_s1s2_er       = np.vstack((emcee_s1s2_er, np.loadtxt(folder + 's1s2_er.txt')))
        emcee_s1s2_ac       = np.vstack((emcee_s1s2_ac, np.loadtxt(folder + 's1s2_ac.txt')))
        emcee_s1s2_cevns_SM = np.vstack((emcee_s1s2_cevns_SM, np.loadtxt(folder + 's1s2_CEVNS-SM.txt')))
        emcee_s1s2_radio    = np.vstack((emcee_s1s2_radio, np.loadtxt(folder + 's1s2_radiogenics.txt')))
        emcee_s1s2_wall     = np.vstack((emcee_s1s2_wall, np.loadtxt(folder + 's1s2_wall.txt')))
        
    
emcee_nobs = len(emcee_pars) # Total number of observations
print('We have ' + str(emcee_nobs) + ' observations...')

emcee_diff_rate = emcee_diff_rate_WIMP + emcee_diff_rate_er + emcee_diff_rate_ac + emcee_diff_rate_cevns_SM + emcee_diff_rate_radio + emcee_diff_rate_wall

emcee_s1s2 = emcee_s1s2_WIMP + emcee_s1s2_er + emcee_s1s2_ac + emcee_s1s2_cevns_SM + emcee_s1s2_radio + emcee_s1s2_wall
emcee_rate = np.sum(emcee_s1s2, axis = 1) # Just to have the same as on the other notebooks. This already includes the backgrounds
emcee_s1s2 = emcee_s1s2.reshape(emcee_nobs, 97, 97)

# Let's work with the log of the mass and cross-section

emcee_pars[:,0] = np.log10(emcee_pars[:,0])
emcee_pars[:,1] = np.log10(emcee_pars[:,1])

# -

# ## Neutrino Floor

# !ls ../data/andresData/28-05-24-files/O1-O4-nufloor/O1-nufloor/

neutrino_floor_minuspidiv2 = np.loadtxt('../data/andresData/28-05-24-files/O1-O4-nufloor/O4-nufloor/floor_rate_minuspidiv2.txt', skiprows = 1, delimiter = ',')
neutrino_floor_minuspidiv4 = np.loadtxt('../data/andresData/28-05-24-files/O1-O4-nufloor/O4-nufloor/floor_rate_minuspidiv4.txt', skiprows = 1, delimiter = ',')
neutrino_floor_pluspidiv2 = np.loadtxt('../data/andresData/28-05-24-files/O1-O4-nufloor/O4-nufloor/floor_rate_pidiv2.txt', skiprows = 1, delimiter = ',')
neutrino_floor_pluspidiv4 = np.loadtxt('../data/andresData/28-05-24-files/O1-O4-nufloor/O4-nufloor/floor_rate_pidiv4.txt', skiprows = 1, delimiter = ',')
neutrino_floor_zero = np.loadtxt('../data/andresData/28-05-24-files/O1-O4-nufloor/O4-nufloor/floor_rate_zero.txt', skiprows = 1, delimiter = ',')
neutrino_mDM = np.loadtxt('../data/andresData/28-05-24-files/O1-O4-nufloor/O4-nufloor/mDM_range.txt', skiprows = 1, delimiter = ',')

# ## Xenon data
#
# from https://arxiv.org/pdf/2007.08796.pdf (Figure 6)

xenon_nt_5s   = np.loadtxt('../data/xenon_nt_5sigma.csv', skiprows = 1, delimiter = ',')
xenon_nt_3s   = np.loadtxt('../data/xenon_nt_3sigma.csv', skiprows = 1, delimiter = ',')
xenon_nt_90cl = np.loadtxt('../data/xenon_nt_90cl.csv', skiprows = 1, delimiter = ',')

# !ls ../data/andresData/BL-constraints-PARAO4

# +
masses = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/masses.txt')[:30]

rate_90_CL_pi2  = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-rate-thetapi2.txt')
rate_90_CL_pi4  = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-rate-thetapi4.txt')
rate_90_CL_0    = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-rate-theta0.txt')
rate_90_CL_mpi2 = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-rate-thetampi2.txt')
rate_90_CL_mpi4 = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-rate-thetampi4.txt')

rate_current_pi2  = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-rate-thetapi2-current.txt')
rate_current_pi4  = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-rate-thetapi4-current.txt')
rate_current_0    = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-rate-theta0-current.txt')
rate_current_mpi2 = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-rate-thetampi2-current.txt')
rate_current_mpi4 = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-rate-thetampi4-current.txt')

s1s2_90_CL_pi2  = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-s1s2-thetapi2.txt')
s1s2_90_CL_pi4  = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-s1s2-thetapi4.txt')
s1s2_90_CL_0    = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-s1s2-theta0.txt')
s1s2_90_CL_mpi2 = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-s1s2-thetampi2.txt')
s1s2_90_CL_mpi4 = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-s1s2-thetampi4.txt')

s1s2_current_pi2  = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-s1s2-thetapi2-current.txt')
s1s2_current_pi4  = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-s1s2-thetapi4-current.txt')
s1s2_current_0    = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-s1s2-theta0-current.txt')
s1s2_current_mpi2 = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-s1s2-thetampi2-current.txt')
s1s2_current_mpi4 = np.loadtxt('../data/andresData/BL-constraints-PARAO4/BL-constraints/BL-s1s2-thetampi4-current.txt')
# -

# !ls ../data/andresData/28-05-24-files/multinest-21046-and-51047/s1s2bin

# +
folder = '../data/andresData/28-05-24-files/multinest-21046-and-51047/'
parameters = [r'$m_{DM}$', r'$\sigma$', r'$\theta$']
n_params = len(parameters)

a = pymultinest.Analyzer(outputfiles_basename= folder + '/rate/mDM50_sigma2e-46_thetapidiv2_', n_params = n_params)

multinest_data_rate     = a.get_data()[:,2:]
multinest_2loglik_rate = a.get_data()[:,1] # -2LogLik = -2*log_prob(data)
multinest_weights_rate  = a.get_data()[:,0]

a = pymultinest.Analyzer(outputfiles_basename= folder + '/drate/mDM50_sigma2e-46_thetapidiv2_', n_params = n_params)

multinest_data_drate     = a.get_data()[:,2:]
multinest_2loglik_drate = a.get_data()[:,1] # -2LogLik = -2*log_prob(data)
multinest_weights_drate  = a.get_data()[:,0]

a = pymultinest.Analyzer(outputfiles_basename= folder + '/s1s2bin/mDM50_sigma2e-46_thetapidiv2_', n_params = n_params)

multinest_data_s1s2     = a.get_data()[:,2:]
multinest_2loglik_s1s2 = a.get_data()[:,1] # -2LogLik = -2*log_prob(data)
multinest_weights_s1s2  = a.get_data()[:,0]
# -

values = a.get_equal_weighted_posterior()

plt.hist(values[:,2])

# ## Bilby

# !ls ../data/andresData/O4-fulldata/O4/O4-bilby-rate-drate-231040-OBSsaved0

bilby_rate  = bilby.result.read_in_result(filename='../data/andresData/O4-fulldata/O4/O4-bilby-rate-drate-231040-OBSsaved0/rate-231040_result.json')
bilby_drate = bilby.result.read_in_result(filename='../data/andresData/O4-fulldata/O4/O4-bilby-rate-drate-231040-OBSsaved0/drate-231040_result.json')
#bilby_s1s2  = bilby.result.read_in_result(filename='../data/andresData/O4-fulldata/O4/O4-bilby-rate-drate-231040-OBSsaved0/s1s2bin-OP2-21047_result.js31040


# # Let's play with SWYFT

# ## Using only the total rate

# ### Training

x_rate = np.log10(rate_trainset) # Observable. Input data.

# +
# Let's normalize everything between 0 and 1

pars_min = np.min(pars_trainset, axis = 0)
pars_max = np.max(pars_trainset, axis = 0)

pars_norm = (pars_trainset - pars_min) / (pars_max - pars_min)

x_min_rate = np.min(x_rate, axis = 0)
x_max_rate = np.max(x_rate, axis = 0)

x_norm_rate = (x_rate - x_min_rate) / (x_max_rate - x_min_rate)

# +
fig,ax = plt.subplots(2,2, gridspec_kw = {'hspace':0.5, 'wspace':0.5})

ax[0,0].hist(x_norm_rate)
ax[0,0].set_xlabel('# Events')

ax[1,0].hist(pars_norm[:,0])
ax[1,0].set_xlabel('$M_{DM}$')

ax[0,1].hist(pars_norm[:,1])
ax[0,1].set_xlabel('$\sigma$')

ax[1,1].hist(pars_norm[:,2])
ax[1,1].set_xlabel('$\\theta$')

# -

x_norm_rate = x_norm_rate.reshape(len(x_norm_rate), 1)
print(x_norm_rate.shape)
print(pars_norm.shape)

# +
# We have to build a swyft.Samples object that will handle the data
samples_rate = swyft.Samples(x = x_norm_rate, z = pars_norm)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_rate = swyft.SwyftDataModule(samples_rate, fractions = [0.7, 0.25, 0.05])


# -

# Now let's define a network that estimates all the 1D and 2D marginal posteriors
class Network_rate(swyft.SwyftModule):
    def __init__(self):
        super().__init__()
        marginals = ((0, 1), (0, 2), (1, 2))
        self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = 1, num_params = 3, varnames = 'pars_norm')
        self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features = 1, marginals = marginals, varnames = 'pars_norm')

    def forward(self, A, B):
        logratios1 = self.logratios1(A['x'], B['z'])
        logratios2 = self.logratios2(A['x'], B['z'])
        return logratios1, logratios2


# +
from pytorch_lightning.callbacks import Callback

class MetricTracker(Callback):

    def __init__(self):
        self.collection = []
        self.val_loss = []
        self.train_loss = []
    
    def on_validation_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics # access it here
        if 'train_loss' in elogs.keys():
            self.val_loss.append(elogs['val_loss'])
            self.train_loss.append(elogs['train_loss'])
            self.collection.append(elogs)

cb = MetricTracker()
# -

# Let's configure, instantiate and traint the network
torch.manual_seed(28890)
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta = 0., patience=100, verbose=False, mode='min')
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O4_newTrain_rate_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
trainer_rate = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 2000, precision = 64, callbacks=[early_stopping_callback, checkpoint_callback, cb])
network_rate = Network_rate()

# +
x_test_rate = np.log10(rate_testset)
x_norm_test_rate = (x_test_rate - x_min_rate) / (x_max_rate - x_min_rate)
x_norm_test_rate = x_norm_test_rate.reshape(len(x_norm_test_rate), 1)

pars_norm_test = (pars_testset - pars_min) / (pars_max - pars_min)

# We have to build a swyft.Samples object that will handle the data
samples_test_rate = swyft.Samples(x = x_norm_test_rate, z = pars_norm_test)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_test_rate = swyft.SwyftDataModule(samples_test_rate, fractions = [0., 0., 1], batch_size = 32)
trainer_rate.test(network_rate, dm_test_rate)

# +
fit = False
if fit:
    trainer_rate.fit(network_rate, dm_rate)
    checkpoint_callback.to_yaml("./logs/O4_newTrain_rate.yaml") 
    ckpt_path = swyft.best_from_yaml("./logs/O4_newTrain_rate.yaml")
    
else:
    ckpt_path = swyft.best_from_yaml("./logs/O4_newTrain_rate.yaml")

# ---------------------------------------------- 
# It converges to val_loss = -1.18 at epoch ~50
# ---------------------------------------------- 

# +
x_test_rate = np.log10(rate_testset)
x_norm_test_rate = (x_test_rate - x_min_rate) / (x_max_rate - x_min_rate)
x_norm_test_rate = x_norm_test_rate.reshape(len(x_norm_test_rate), 1)
pars_norm_test = (pars_testset - pars_min) / (pars_max - pars_min)

# We have to build a swyft.Samples object that will handle the data
samples_test_rate = swyft.Samples(x = x_norm_test_rate, z = pars_norm_test)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_test_rate = swyft.SwyftDataModule(samples_test_rate, fractions = [0., 0., 1], batch_size = 32)
trainer_rate.test(network_rate, dm_test_rate, ckpt_path = ckpt_path)

# ---------------------------------------------- 
# It converges to val_loss = -1. in testset
# ---------------------------------------------- 
# -

# ### Let's make some inference

# +
# Let's normalize testset between 0 and 1

pars_norm = (emcee_pars - pars_min) / (pars_max - pars_min)

x_rate = np.log10(emcee_rate)
x_norm_rate = (x_rate - x_min_rate) / (x_max_rate - x_min_rate)
x_norm_rate = x_norm_rate.reshape(len(x_norm_rate), 1)

# +
# First let's create some observation from some "true" theta parameters
i = 0#np.random.randint(24) # 239 (disc) 455 (exc) 203 (middle) #415 (49.67, 9.3e-47, -0.7)
print(i)
pars_true = pars_norm[i,:]
x_obs     = x_norm_rate[i,:]

print('Real values:' + str(pars_true * (pars_max - pars_min) + pars_min ))
print('"Normalized Observed" x value : {}'.format(x_obs))
real_val = 10**(x_obs * (x_max_rate - x_min_rate) + x_min_rate)
print('"Observed" x value : {} events'.format(real_val))

if real_val < 2930: 
    flag = 'exc'
else:
    flag = 'disc'
print(flag)
# -


B = (pars_max[1] - pars_min[1])
A = pars_min[1]

# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
pars_prior    = np.random.uniform(low = 0, high = 1, size = (100_000, 3))
#pars_prior = np.random.uniform(low = pars_min, high = pars_max, size = (100_000, 3))
#pars_prior = np.random.uniform(low = [6, 1e-50, -1.57], high = [1000, 1e-45, 1.57], size = (100_000, 3))
#pars_prior[:,0] = np.log10(pars_prior[:,0])
#pars_prior[:,1] = np.log10(pars_prior[:,1])
#pars_prior = (pars_prior - pars_min) / (pars_max - pars_min)

prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions_rate = trainer_rate.infer(network_rate, obs, prior_samples)

# +
fig,ax = plt.subplots(2,2, figsize = (6,6), 
                      gridspec_kw={'height_ratios': [0.5, 2], 'width_ratios':[2,0.5]})

plt.subplots_adjust(hspace = 0.1, wspace = 0.1)

plot1d(ax[0,0], predictions_rate, pars_true, par = 0)
plot2d(ax[1,0], predictions_rate, pars_true)
plot1d(ax[1,1], predictions_rate, pars_true, par = 1, flip = True)
ax[0,1].remove()

ax[0,0].set_xlim(8,1e3)
ax[1,0].set_xlim(8,1e3)
ax[1,0].set_ylim(1e-41,1e-36)
ax[1,1].set_ylim(1e-41,1e-36)

ax[0,0].set_xlabel('')
ax[0,0].set_ylabel('$P(m|x)$')
ax[0,0].set_xticks([])
ax[1,1].set_ylabel('')
ax[1,1].set_yticks([])
ax[1,1].set_xlabel('$P(\sigma|x)$')
#ax[1,0].grid(which = 'both')
#plt.savefig('../graph/O4_graph/2d_custom_posteriors_' + str(i) + '_rate.pdf')
# -

# ## Only using the total diff_rate

# ### Training

x_drate = np.log10(diff_rate_trainset) # Observable. Input data. 

# +
# Let's normalize everything between 0 and 1

pars_min = np.min(pars_trainset, axis = 0)
pars_max = np.max(pars_trainset, axis = 0)

pars_norm = (pars_trainset - pars_min) / (pars_max - pars_min)

x_min_drate = np.min(x_drate, axis = 0)
x_max_drate = np.max(x_drate, axis = 0)

x_norm_drate = (x_drate - x_min_drate) / (x_max_drate - x_min_drate)

# +
fig,ax = plt.subplots(2,2, gridspec_kw = {'hspace':0.5, 'wspace':0.5})


for ii in range(50):
    ax[0,0].plot(x_norm_drate[ii])
ax[0,0].set_xlabel('$E_{r}$')

ax[1,0].hist(pars_norm[:,0])
ax[1,0].set_xlabel('$M_{DM}$')

ax[0,1].hist(pars_norm[:,1])
ax[0,1].set_xlabel('$\sigma$')

ax[1,1].hist(pars_norm[:,2])
ax[1,1].set_xlabel('$\\theta$')

ax[0,0].plot(x_norm_drate[502], c = 'black')

# -

print(x_norm_drate.shape)
print(pars_norm.shape)

# +
# We have to build a swyft.Samples object that will handle the data
samples_drate = swyft.Samples(x = x_norm_drate, z = pars_norm)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_drate = swyft.SwyftDataModule(samples_drate, fractions = [0.7, 0.25, 0.05], batch_size = 32)


# -

# Now let's define a network that estimates all the 1D and 2D marginal posteriors
class Network(swyft.SwyftModule):
    def __init__(self, lr = 1e-3, gamma = 1.):
        super().__init__()
        self.optimizer_init = swyft.OptimizerInit(torch.optim.Adam, dict(lr = lr, weight_decay=1e-5),
              torch.optim.lr_scheduler.ExponentialLR, dict(gamma = gamma))
        self.net = torch.nn.Sequential(
          torch.nn.Linear(58, 500),
          torch.nn.ReLU(),
          torch.nn.Linear(500, 1000),
          torch.nn.ReLU(),
          torch.nn.Linear(1000, 500),
          torch.nn.ReLU(),
          torch.nn.Linear(500, 50),
          torch.nn.ReLU(),
          #torch.nn.Dropout(0.2),
          torch.nn.Linear(50, 5)
        )
        marginals = ((0, 1), (0, 2), (1, 2))
        self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = 5, num_params = 3, varnames = 'pars_norm')
        self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features = 5, marginals = marginals, varnames = 'pars_norm')

    def forward(self, A, B):
        img = torch.tensor(A['x'])
        #z   = torch.tensor(B['z'])
        f   = self.net(img)
        logratios1 = self.logratios1(f, B['z'])
        logratios2 = self.logratios2(f, B['z'])
        return logratios1, logratios2


# +
### Now let's define a network that estimates all the 1D and 2D marginal posteriors
###class Network(swyft.SwyftModule):
###    def __init__(self):
###        super().__init__()
###        marginals = ((0, 1), (0, 2), (1, 2))
###        self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = 58, num_params = 3, varnames = 'pars_norm')
###        self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features = 58, marginals = marginals, varnames = 'pars_norm')
###
###    def forward(self, A, B):
###        logratios1 = self.logratios1(A['x'], B['z'])
###        logratios2 = self.logratios2(A['x'], B['z'])
###        return logratios1, logratios2

# +
from pytorch_lightning.callbacks import Callback

class MetricTracker(Callback):

    def __init__(self):
        self.collection = []
        self.val_loss = []
        self.train_loss = []
    
    def on_validation_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics # access it here
        if 'train_loss' in elogs.keys():
            self.val_loss.append(elogs['val_loss'])
            self.train_loss.append(elogs['train_loss'])
            self.collection.append(elogs)

cb = MetricTracker()
# -

# Let's configure, instantiate and traint the network
torch.manual_seed(28890)
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta = 0., patience=50, verbose=False, mode='min')
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O4_newTrain_log_drate_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
trainer_drate = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 2000, precision = 64, callbacks=[early_stopping_callback, checkpoint_callback, cb])
network_drate = Network()


# +
x_test_drate = diff_rate_testset
x_norm_test_drate = (x_test_drate - x_min_drate) / (x_max_drate - x_min_drate)

pars_norm_test = (pars_testset - pars_min) / (pars_max - pars_min)

# We have to build a swyft.Samples object that will handle the data
samples_test_drate = swyft.Samples(x = x_norm_test_drate, z = pars_norm_test)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_test_drate = swyft.SwyftDataModule(samples_test_drate, fractions = [0., 0., 1], batch_size = 32)
trainer_drate.test(network_drate, dm_test_drate)

# +
fit = False
if fit:
    trainer_drate.fit(network_drate, dm_drate)
    checkpoint_callback.to_yaml("./logs/O4_newTrain_log_drate.yaml") 
    ckpt_path = swyft.best_from_yaml("./logs/O4_newTrain_log_drate.yaml")
else:
    ckpt_path = swyft.best_from_yaml("./logs/O4_newTrain_log_drate.yaml")

# ---------------------------------------------- 
# It converges to val_loss = -1.8 @ epoch 20
# ---------------------------------------------- 

# +
x_test_drate = np.log10(diff_rate_testset)
x_norm_test_drate = (x_test_drate - x_min_drate) / (x_max_drate - x_min_drate)

pars_norm_test = (pars_testset - pars_min) / (pars_max - pars_min)

# We have to build a swyft.Samples object that will handle the data
samples_test_drate = swyft.Samples(x = x_norm_test_drate, z = pars_norm_test)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_test_drate = swyft.SwyftDataModule(samples_test_drate, fractions = [0., 0., 1], batch_size = 32)
trainer_drate.test(network_drate, dm_test_drate, ckpt_path = ckpt_path)

# ---------------------------------------------- 
# It converges to -1.51 @ testset
# ---------------------------------------------- 
# -

# ### Let's make some inference

# +
# Let's normalize testset between 0 and 1

pars_norm = (emcee_pars - pars_min) / (pars_max - pars_min)

x_drate = np.log10(emcee_diff_rate)
x_norm_drate = (x_drate - x_min_drate) / (x_max_drate - x_min_drate)

# +
# First let's create some observation from some "true" theta parameters
#i = np.random.randint(24)
print(i)
pars_true = pars_norm[i,:]
x_obs     = x_norm_drate[i,:]

plt.plot(x_obs)
plt.text(5,0.5, str(np.sum(x_drate[i,:])))
if np.sum(emcee_diff_rate_WIMP[i,:]) < 300: 
    flag = 'exc'
else:
    flag = 'disc'
print(np.sum(emcee_diff_rate_WIMP[i,:]))
print(flag)
# -

pars_true * (pars_max - pars_min) + pars_min

# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
#pars_prior = np.random.uniform(low = 0, high = 1, size = (100_000, 3))

#prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions_drate = trainer_drate.infer(network_drate, obs, prior_samples)

# +
fig,ax = plt.subplots(2,2, figsize = (6,6), 
                      gridspec_kw={'height_ratios': [0.5, 2], 'width_ratios':[2,0.5]})

plt.subplots_adjust(hspace = 0.1, wspace = 0.1)

plot1d(ax[0,0], predictions_drate, pars_true, par = 0)
plot2d(ax[1,0], predictions_drate, pars_true)
plot1d(ax[1,1], predictions_drate, pars_true, par = 1, flip = True)
ax[0,1].remove()

ax[0,0].set_xlim(8,1e3)
ax[1,0].set_xlim(8,1e3)
ax[1,0].set_ylim(1e-41,1e-36)
ax[1,1].set_ylim(1e-41,1e-36)

ax[0,0].set_xlabel('')
ax[0,0].set_ylabel('$P(m|x)$')
ax[0,0].set_xticks([])
ax[1,1].set_ylabel('')
ax[1,1].set_yticks([])
ax[1,1].set_xlabel('$P(\sigma|x)$')
#ax[1,0].grid(which = 'both')
#plt.savefig('../graph/2d_custom_posteriors_' + str(i) + '_drate.pdf')
# -

# ## Using s1s2

# ### training

x_s1s2 = s1s2_trainset[:,:-1,:-1] # Observable. Input data. I am cutting a bit the images to have 64x64

# +
# Let's normalize everything between 0 and 1

pars_min = np.min(pars_trainset, axis = 0)
pars_max = np.max(pars_trainset, axis = 0)

pars_norm = (pars_trainset - pars_min) / (pars_max - pars_min)

x_min_s1s2 = np.min(x_s1s2, axis = 0)
x_max_s1s2 = np.max(x_s1s2)

x_norm_s1s2 = x_s1s2
#ind_nonzero = np.where(x_max_s1s2 > 0)
#x_norm_s1s2[:,ind_nonzero[0], ind_nonzero[1]] = (x_s1s2[:,ind_nonzero[0], ind_nonzero[1]] - x_min_s1s2[ind_nonzero[0], ind_nonzero[1]]) / (x_max_s1s2[ind_nonzero[0], ind_nonzero[1]] - x_min_s1s2[ind_nonzero[0], ind_nonzero[1]])
#x_norm_s1s2 = x_s1s2 / x_max_s1s2


# +
fig,ax = plt.subplots(2,2, gridspec_kw = {'hspace':0.5, 'wspace':0.5})

ax[0,0].hist(x_norm_s1s2[:,50,30])
ax[0,0].set_xlabel('# Events')

ax[1,0].hist(pars_norm[:,0])
ax[1,0].set_xlabel('$M_{DM}$')

ax[0,1].hist(pars_norm[:,1])
ax[0,1].set_xlabel('$\sigma$')

ax[1,1].hist(pars_norm[:,2])
ax[1,1].set_xlabel('$\\theta$')

# -

x_norm_s1s2 = x_norm_s1s2.reshape(len(x_norm_s1s2), 1, 96, 96) # The shape need to be (#obs, #channels, dim, dim)
print(x_norm_s1s2.shape)
print(pars_norm.shape)

# +
# We have to build a swyft.Samples object that will handle the data
samples_s1s2 = swyft.Samples(x = x_norm_s1s2, z = pars_norm)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_s1s2 = swyft.SwyftDataModule(samples_s1s2, fractions = [0.7, 0.25, 0.05], batch_size = 32)


# -

# Now let's define a network that estimates all the 1D and 2D marginal posteriors
class Network(swyft.SwyftModule):
    def __init__(self, lr = 1e-3, gamma = 1.):
        super().__init__()
        self.optimizer_init = swyft.OptimizerInit(torch.optim.Adam, dict(lr = lr, weight_decay=1e-5),
              torch.optim.lr_scheduler.ExponentialLR, dict(gamma = gamma))
        self.net = torch.nn.Sequential(
          torch.nn.Conv2d(1, 10, kernel_size=5),
          torch.nn.MaxPool2d(2),
          torch.nn.ReLU(),
          torch.nn.Dropout(0.2),
          torch.nn.Conv2d(10, 20, kernel_size=5, padding=2),
          torch.nn.MaxPool2d(2),
          torch.nn.ReLU(),
          torch.nn.Dropout(0.2),
          torch.nn.Flatten(),
          torch.nn.Linear(10580, 50),
          torch.nn.ReLU(),
          torch.nn.Dropout(0.2),
          torch.nn.Linear(50, 10),
        )
        marginals = ((0, 1), (0, 2), (1, 2))
        self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = 10, num_params = 3, varnames = 'pars_norm')
        self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features = 10, marginals = marginals, varnames = 'pars_norm')

    def forward(self, A, B):
        img = torch.tensor(A['x'])
        #z   = torch.tensor(B['z'])
        f   = self.net(img)
        logratios1 = self.logratios1(f, B['z'])
        logratios2 = self.logratios2(f, B['z'])
        return logratios1, logratios2


# +
class MetricTracker(Callback):

    def __init__(self):
        self.collection = []
        self.val_loss = []
        self.train_loss = []
    
    def on_validation_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics # access it here
        if 'train_loss' in elogs.keys():
            self.val_loss.append(elogs['val_loss'])
            self.train_loss.append(elogs['train_loss'])
            self.collection.append(elogs)

cb = MetricTracker()
# -

# Let's configure, instantiate and traint the network
torch.manual_seed(28890)
cb = MetricTracker()
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta = 0., patience=20, verbose=False, mode='min')
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O4_newTrain_s1s2_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
trainer_s1s2 = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 2500, precision = 64, callbacks=[early_stopping_callback, checkpoint_callback, cb])
network_s1s2 = Network()

# +
x_norm_test_s1s2 = s1s2_testset[:,:-1,:-1] # Observable. Input data. I am cutting a bit the images to have 96x96
# #%x_norm_test_s1s2 = x_norm_test_s1s2 / x_max_s1s2 # Observable. Input data. I am cutting a bit the images to have 96x96
x_norm_test_s1s2 = x_norm_test_s1s2.reshape(len(x_norm_test_s1s2), 1, 96, 96)

pars_norm_test = (pars_testset - pars_min) / (pars_max - pars_min)

# We have to build a swyft.Samples object that will handle the data
samples_test_s1s2 = swyft.Samples(x = x_norm_test_s1s2, z = pars_norm_test)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_test_s1s2 = swyft.SwyftDataModule(samples_test_s1s2, fractions = [0., 0., 1], batch_size = 32)
trainer_s1s2.test(network_s1s2, dm_test_s1s2)
# -

ckpt_path = swyft.best_from_yaml("./logs/O4_newTrain_s1s2.yaml")



# +
x_norm_test_s1s2 = s1s2_testset[:,:-1,:-1] # Observable. Input data. I am cutting a bit the images to have 96x96
# #%x_norm_test_s1s2 = x_norm_test_s1s2 / x_max_s1s2 # Observable. Input data. I am cutting a bit the images to have 96x96
x_norm_test_s1s2 = x_norm_test_s1s2.reshape(len(x_norm_test_s1s2), 1, 96, 96)

pars_norm_test = (pars_testset - pars_min) / (pars_max - pars_min)

# We have to build a swyft.Samples object that will handle the data
samples_test_s1s2 = swyft.Samples(x = x_norm_test_s1s2, z = pars_norm_test)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_test_s1s2 = swyft.SwyftDataModule(samples_test_s1s2, fractions = [0., 0., 1], batch_size = 32)
trainer_s1s2.test(network_s1s2, dm_test_s1s2, ckpt_path = ckpt_path)

# ---------------------------------------
# Min val loss value at 7 epochs. -1.53 @ testset
# ---------------------------------------

# -

# ### Let's make some inference

# +
# Let's normalize testset between 0 and 1

pars_norm = (emcee_pars - pars_min) / (pars_max - pars_min)

x_norm_s1s2 = x_s1s2 = emcee_s1s2[:,:-1,:-1]
# #%x_norm_s1s2 = x_s1s2 = s1s2_testset[:,:-1,:-1] / x_max_s1s2

# +
# First let's create some observation from some "true" theta parameters
i = 0#np.random.randint(24) # 
print(i)

pars_true = pars_norm[i,:]
x_obs     = x_norm_s1s2[i,:].reshape(1,96,96)

if np.sum(x_obs) < 2930: 
    flag = 'exc'
else:
    print(np.sum(x_obs))
    flag = 'disc'
print(flag)

plt.imshow(x_obs[0].T, origin = 'lower')
# -

pars_true * (pars_max - pars_min) + pars_min

# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
#pars_prior = np.random.uniform(low = 0, high = 1, size = (100_000, 3))
#pars_prior[:,2] = np.random.normal(pars_true[2], 0.001, (len(pars_prior)))
#prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions_s1s2 = trainer_s1s2.infer(network_s1s2, obs, prior_samples)

# +
fig,ax = plt.subplots(2,2, figsize = (6,6), 
                      gridspec_kw={'height_ratios': [0.5, 2], 'width_ratios':[2,0.5]})

plt.subplots_adjust(hspace = 0.1, wspace = 0.1)

#plot1d(ax[0,0], predictions_s1s2, pars_true, par = 0)
#plot1d(ax[0,0], predictions_rate, pars_true, par = 0, fill = False, linestyle = ':', color = color_rate)
#plot1d(ax[0,0], predictions_drate, pars_true, par = 0, fill = False, linestyle = '--', color = color_drate)
plot1d_comb(ax[0,0], [predictions_rate], pars_true, par = 0, fill = False, linestyle = ':', color = color_rate)
plot1d_comb(ax[0,0], [predictions_rate, predictions_drate], pars_true, par = 0, fill = False, linestyle = '--', color = color_drate)
plot1d_comb(ax[0,0], [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 0, fill = True, linestyle = 'solid', color = color_s1s2)

#plot2d(ax[1,0], predictions_s1s2, pars_true)
plot2d(ax[1,0], predictions_rate, pars_true, fill = False, line = True, linestyle = ':', color = color_rate)
#plot2d(ax[1,0], predictions_drate, pars_true, fill = False, line = True, linestyle = '--', color = color_drate)
plot2d_comb(ax[1,0], [predictions_rate, predictions_drate], pars_true, fill = False, line = True, linestyle = '--', color = color_drate)
plot2d_comb(ax[1,0], [predictions_rate, predictions_drate, predictions_s1s2], pars_true, fill = True, line = True, linestyle = 'solid', color = color_s1s2)

#plot1d(ax[1,1], predictions_s1s2, pars_true, par = 1, flip = True)
#plot1d(ax[1,1], predictions_rate, pars_true, par = 1, flip = True, fill = False, linestyle = ':', color = color_rate)
#plot1d(ax[1,1], predictions_drate, pars_true, par = 1, flip = True, fill = False, linestyle = '--', color = color_drate)
plot1d_comb(ax[1,1], [predictions_rate], pars_true, par = 1, flip = True, fill = False, linestyle = '--', color = color_rate)
plot1d_comb(ax[1,1], [predictions_rate, predictions_drate], pars_true, par = 1, flip = True, fill = False, linestyle = '--', color = color_drate)
plot1d_comb(ax[1,1], [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 1, flip = True, fill = True, linestyle = 'solid', color = color_s1s2)

ax[0,0].set_xlim(8,1e3)
ax[1,0].set_xlim(8,1e3)
ax[1,0].set_ylim(1e-41,1e-36)
ax[1,1].set_ylim(1e-41,1e-36)

ax[0,0].set_xlabel('')
ax[0,0].set_ylabel('$P(m|x)$')
ax[0,0].set_xticks([])
ax[1,1].set_ylabel('')
ax[1,1].set_yticks([])
ax[1,1].set_xlabel('$P(\sigma|x)$')

custom_lines = []
labels = ['Total Rate', 'Dif. Rate', 'S1-S2']
markers = [':','--', 'solid']
colors = [color_rate, color_drate, color_s1s2]
for i in range(3):
    custom_lines.append( Line2D([0],[0], linestyle = markers[i], color = colors[i], 
            label = labels[i]) )

ax[0,1].axis('off')
ax[0,1].legend(handles = custom_lines, frameon = False, loc = 'lower left', bbox_to_anchor=(-0.2,0.05))
#ax[0,1].remove()

#ax[0,1].
#ax[1,0].grid(which = 'both')
#plt.savefig('../graph/2d_custom_posteriors_emcee' + str(i) + '.pdf')


# +
rate_samples = bilby_rate.samples
drate_samples = bilby_drate.samples
#s1s2_samples = bilby_s1s2.samples

fig = corner.corner(rate_samples, smooth = 2.5, levels=[0.9], bins = 30, plot_density=False, color = color_rate, fill_contours=False)

axes = fig.get_axes()

#axes[0].hist(s1s2_samples[:,0], color = 'black', bins = 30)
corner.corner(drate_samples, smooth = 2, levels=[0.9], bins = 30, plot_density=False, color = color_drate, fill_contours=False, fig = fig)

#corner.corner(s1s2_samples, smooth = 2, levels=[0.9], bins = 30, plot_density=False, color = color_s1s2, fill_contours=False, fig = fig)

plt.show()

# +
rate  = True
drate = True
s1s2  = True

prob = [0.9]

#fig = bilby_s1s2.plot_corner(outdir='../O4_graph/', color = 'grey', levels=prob, smooth = 2, bins = 30, alpha = 0.6)
fig = bilby_drate.plot_corner(outdir='../O4_graph/', color = 'grey', levels=prob, smooth = 2, bins = 30, alpha = 0.6)
#fig = bilby_rate.plot_corner(outdir='../O4_graph/', color = 'grey', levels=prob, smooth = 2, bins = 30, alpha = 0.6)

#fig = corner.corner(rate_samples, smooth = 2.5, levels = [0.9], bins = 30, plot_density = False, color = 'black', fill_contours = False, linestyles = ['--'])
#corner.corner(drate_samples, smooth = 2, levels = [0.9], bins = 30, plot_density = False, color = 'magenta', fill_contours = False, 
#              fig = fig, contour_kwargs = {'linestyles':'--'}, contourf_kwargs = {'alpha':0})
#corner.corner(s1s2_samples, smooth = 2, levels = [0.9], bins = 30, plot_density = False, color = color_s1s2, fill_contours = False, fig = fig, ls = '--')

axes = fig.get_axes()


ax = axes[0]
ax.cla()
ax.hist(drate_samples[:,0], color = 'grey', bins = 30, zorder = 0, histtype = 'step')

if rate:
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 0, 
                 fill = False, linestyles = ['solid',':'], color = color_rate, fac = 60, probs = prob)
if drate: 
    plot1d_emcee(ax, [predictions_rate, predictions_drate], pars_true, par = 0, 
             fill = False, linestyles = ['solid',':'], color = color_drate, fac = 70, probs = prob)
if s1s2:
    plot1d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 0, 
                 fill = False, linestyles = ['solid',':'], color = color_s1s2, fac = 30, probs = prob)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
ax.set_xlim([1, 3])

ax = axes[3]
if rate:
    plot2d_emcee(ax, [predictions_rate], pars_true, fill = False, line = True, linestyles = ['solid','--'], 
                 color = color_rate, probs = prob, zorder = 2, nvals = 20, smooth = 2)
if drate:
    plot2d_emcee(ax, [predictions_rate, predictions_drate], pars_true, fill = False, line = True, linestyles = ['solid','--'], 
                 color = color_drate, probs = prob, zorder = 3, nvals = 20, smooth = 2)
if s1s2:
    plot2d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                 color = color_s1s2, probs = prob, zorder = 4, nvals = 40)
ax.set_ylabel('$Log_{10}(\\sigma \ [cm^{2}])$', fontsize = 12)
ax.set_xlim([1, 3])
ax.set_ylim([-41, -36])

ax = axes[4]
ax.cla()
ax.hist(drate_samples[:,1], color = 'grey', bins = 30, zorder = 0, histtype = 'step')

if rate:
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 1, 
                 flip = False, fill = False, linestyles = ['solid', ':'], color = color_rate, fac = 50, probs = prob)
if drate:
    plot1d_emcee(ax, [predictions_rate, predictions_drate], pars_true, par = 1, 
                 flip = False, fill = False, linestyles = ['solid', ':'], color = color_drate, fac = 10, probs = prob)
if s1s2:
    plot1d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 1, 
                 flip = False, fill = False, linestyles = ['solid', ':'], color = color_s1s2, fac = 5, probs = prob)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title('')
ax.set_xlim([-41, -36])
ax.set_xticks([])
ax.set_yticks([])

ax = axes[6]

if rate:
    plot2d_emcee_m_theta(ax, [predictions_rate], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_rate, probs = prob, zorder = 2, smooth = 2)
if drate:
    plot2d_emcee_m_theta(ax, [predictions_rate, predictions_drate], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_drate, probs = prob, zorder = 2, smooth = 2)
if s1s2:
    plot2d_emcee_m_theta(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_s1s2, probs = prob, zorder = 2, smooth = 2)
ax.set_ylabel('$\\theta$', fontsize = 12)
ax.set_xlabel('$Log_{10}(M_{DM} \ [GeV])$', fontsize = 12)
ax.set_xlim([1, 3])
ax.set_ylim([-1.6, 1.6])

ax = axes[7]

if rate:
    plot2d_emcee_sigma_theta(ax, [predictions_rate], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_rate, probs = prob, zorder = 2, smooth = 2)
if drate:
    plot2d_emcee_sigma_theta(ax, [predictions_rate, predictions_drate], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_drate, probs = prob, zorder = 2, smooth = 2)
if s1s2:
    plot2d_emcee_sigma_theta(ax, [predictions_rate, predictions_drate,  predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_s1s2, probs = prob, zorder = 2, smooth = 2)
ax.set_xlabel('$Log_{10}(\\sigma \ [cm^{2}])$', fontsize = 12)
ax.set_ylabel('')
ax.set_xlim([-41., -36])
ax.set_ylim([-1.6, 1.6])

ax = axes[8]
ax.clear()
ax.hist(drate_samples[:,2], color = 'grey', bins = 30, zorder = 0, histtype = 'step', range = (-1.6,1.6))

if rate:
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 2, 
                 flip = False, fill = False, linestyles = ['solid',':'], color = color_rate, fac = 50, probs = prob)
if drate:
    plot1d_emcee(ax, [predictions_rate, predictions_drate], pars_true, par = 2, 
                 flip = False, fill = False, linestyles = ['solid',':'], color = color_drate, fac = 50, probs = prob)
if s1s2:
    plot1d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 2, 
                 flip = False, fill = False, linestyles = ['solid',':'], color = color_s1s2, fac = 50, probs = prob)
ax.set_ylabel('')
ax.set_title('')
ax.set_xlim([-1.6, 1.6])
ax.set_xticks([-1.5,0,1.5])
ax.set_xticklabels(['-1.5','0.0', '1.5'], rotation = 45)
ax.set_yticks([])
ax.text(-0.06,-42, '$\\theta$', fontsize = 12)
#ax.set_xlabel('$\\theta$', fontsize = 12)

fig.savefig('../graph/O4_graph/SWYFT_BILBY_comparison_O1_m_{:.2f}_s_{:.2f}_t_{:.2f}.pdf'.format(emcee_pars[0,0],emcee_pars[0,1],emcee_pars[0,2]), bbox_inches='tight')
#fig
# +
rate_samples = bilby_rate.samples[:,:2]
drate_samples = bilby_drate.samples[:,:2]
#s1s2_samples = bilby_s1s2.samples[:,:2]

rate  = False
drate = False
s1s2  = True

prob = [0.9]

fig,axes = plt.subplots(2,2, width_ratios = [1,0.4], height_ratios = [0.4,1])

if rate:
    corner.corner(rate_samples, fig = fig, smooth = 2.5, levels=prob, bins = 30, plot_density=False, color = 'gray', fill_contours=True)
if drate:
    corner.corner(drate_samples, fig = fig, smooth = 2.5, levels=prob, bins = 30, plot_density=False, color = 'gray', fill_contours=True)
#if s1s2:
#    corner.corner(s1s2_samples, fig = fig, smooth = 2.5, levels=prob, bins = 30, plot_density=False, color = 'gray', fill_contours=True)

#axes = fig.get_axes()

ax = axes[0,0]
ax.cla()

if rate:
    ax.hist(rate_samples[:,0], color = 'grey', bins = 30, zorder = 0, histtype = 'step')
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 0, 
                 fill = False, linestyles = ['solid',':'], color = color_rate, fac = 130, probs = prob)
if drate: 
    ax.hist(drate_samples[:,0], color = 'grey', bins = 30, zorder = 0, histtype = 'step')
    plot1d_emcee(ax, [predictions_rate, predictions_drate], pars_true, par = 0, 
             fill = False, linestyles = ['solid',':'], color = color_drate, fac = 130, probs = prob)
if s1s2:
    ax.hist(s1s2_samples[:,0], color = 'grey', bins = 30, zorder = 0, histtype = 'step')
    plot1d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 0, 
                 fill = False, linestyles = ['solid',':'], color = color_s1s2, fac = 130, probs = prob)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
ax.set_xlim([1, 3])

ax = axes[1,0]
if rate:
    plot2d_emcee(ax, [predictions_rate], pars_true, fill = False, line = True, linestyles = ['solid','--'], 
                 color = color_rate, probs = prob, zorder = 2, nvals = 20, smooth = 2)
if drate:
    plot2d_emcee(ax, [predictions_rate, predictions_drate], pars_true, fill = False, line = True, linestyles = ['solid','--'], 
                 color = color_drate, probs = prob, zorder = 3, nvals = 20, smooth = 2)
if s1s2:
    plot2d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                 color = color_s1s2, probs = prob, zorder = 4, nvals = 40)
ax.set_ylabel('$Log_{10}(\\sigma \ [cm^{2}])$', fontsize = 12)
ax.set_xlim([1, 3])
ax.set_ylim([-49.5, -43])

ax = axes[1,1]
ax.cla()

if rate:
    ax.hist(rate_samples[:,1], color = 'grey', bins = 30, zorder = 0, histtype = 'step', orientation="horizontal")
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 1, 
                 flip = True, fill = False, linestyles = ['solid', ':'], color = color_rate, fac = 70, probs = prob)
if drate:
    ax.hist(drate_samples[:,1], color = 'grey', bins = 30, zorder = 0, histtype = 'step', orientation="horizontal")
    plot1d_emcee(ax, [predictions_rate, predictions_drate], pars_true, par = 1, 
                 flip = True, fill = False, linestyles = ['solid', ':'], color = color_drate, fac = 20, probs = prob)
if s1s2:
    plot1d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 1, 
                 flip = True, fill = False, linestyles = ['solid', ':'], color = color_s1s2, fac = 10, probs = prob)
    ax.hist(s1s2_samples[:,1], color = 'grey', bins = 30, zorder = 0, histtype = 'step', orientation="horizontal")
    #hist = np.histogram(s1s2_samples[:,1], bins = 30)
    #ax.barplot(hist[0], hist[1][:-1], color = 'grey', zorder = 0)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title('')
#ax.set_xlim([-49.5, -43])
ax.set_xticks([])
ax.set_yticks([])

plt.savefig('../graph/SWYFT_BILBY_s1s2_comparison_O1_m_{:.2f}_s_{:.2f}_t_{:.2f}.pdf'.format(emcee_pars[0,0],emcee_pars[0,1],emcee_pars[0,2]), bbox_inches='tight')
# -


