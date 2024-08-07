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
color_drate = 'darkblue' #"#0072b2"
color_s1s2 = 'limegreen' #"#009e73"

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
    ax.axvline(x = -49, c = 'black', linewidth = 2)

    if (low_1sigma is not None) & (up_1sigma is not None):
        ax.axvline(low_1sigma, c = 'black', linestyle = '--')
        ax.axvline(up_1sigma, c = 'black', linestyle = '--')
    
    #ax.axvline(low_2sigma, c = 'black', linestyle = '--')
    #ax.axvline(up_2sigma, c = 'black', linestyle = '--')
    
    #ax.axvline(low_3sigma, c = 'black', linestyle = ':')
    #ax.axvline(up_3sigma, c = 'black', linestyle = ':')

    ax.set_xlim(-50, -43)
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
        ax.set_ylim(1e-50, 1e-42)
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
    s_values = np.logspace(-49.9, -43.1, nvals)
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

    fac = 1 / trapezoid(ratios, parameter)
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
        ax.set_ylim(1e-50, 1e-42)
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
    s_values = np.logspace(-49., -43.1, nvals)
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

    fac = 1 / trapezoid(ratios, parameter)
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
            #ax.axvline(x = (pars_true[par] * (pars_max[par] - pars_min[par]) + pars_min[par]), color = 'black')
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
            #ax.axhline(y = (pars_true[par] * (pars_max[par] - pars_min[par]) + pars_min[par]), color = 'black')
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
    s_values = np.logspace(-49., -43.1, nvals)
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
    
    #ax.axvline(x = (pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0]), color = 'black')
    #ax.axhline(y = (pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1]), color = 'black')
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
    cuts = np.linspace(np.min(res), np.max(res), 50)
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
    
    #ax.axvline(x = (pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0]), color = 'black')
    #ax.axhline(y = (pars_true[2] * (pars_max[2] - pars_min[2]) + pars_min[2]), color = 'black')
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
    s_values = np.logspace(-49., -43.1, nvals)
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
    
    #ax.axvline(x = (pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1]), color = 'black')
    #ax.axhline(y = (pars_true[2] * (pars_max[2] - pars_min[2]) + pars_min[2]), color = 'black')
    ax.set_xlabel('$\sigma$ $[cm^{2}]$')
    ax.set_ylabel('$\theta$')

    return ax


# # Let's load the data

# !ls ../data/andresData/SI-run0and1/SI-run01/

# +
# where are your files?
datFolder = ['../data/andresData/SI-run0and1/SI-run01/', 
             '../data/andresData/SI-run0and1/SI-run02/']
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
ind_new = np.where(pars[:,1] < -45)[0]

#nobs = len(ind_new)
#pars = pars[ind_new]

#rate = rate[ind_new]
#diff_rate = diff_rate[ind_new]
#s1s2 = s1s2[ind_new]

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

save = False
if save:
    
    pars_min = np.min(pars_trainset, axis = 0)
    pars_max = np.max(pars_trainset, axis = 0)    
    np.savetxt('O1_pars_min.txt', pars_min)
    np.savetxt('O1_pars_max.txt', pars_max)
    
    x_rate = np.log10(rate_trainset) # Observable. Input data.
    x_min_rate = np.min(x_rate, axis = 0)
    x_max_rate = np.max(x_rate, axis = 0)
    np.savetxt('O1_rate_minmax.txt', np.asarray([x_min_rate, x_max_rate]))

    x_drate = np.log10(diff_rate_trainset) # Observable. Input data. 
    x_min_drate = np.min(x_drate, axis = 0)
    x_max_drate = np.max(x_drate, axis = 0)
    np.savetxt('O1_drate_min.txt', x_min_drate)
    np.savetxt('O1_drate_max.txt', x_max_drate)

    x_s1s2 = s1s2_trainset[:,:-1,:-1] # Observable. Input data. I am cutting a bit the images to have 64x64
    x_min_s1s2 = np.min(x_s1s2, axis = 0)
    x_max_s1s2 = np.max(x_s1s2, axis = 0)
    np.savetxt('O1_s1s2_min.txt', x_min_s1s2)
    np.savetxt('O1_s1s2_max.txt', x_max_s1s2)
else:
    pars_min = np.loadtxt('O1_pars_min.txt')
    pars_max = np.loadtxt('O1_pars_max.txt')
    x_minmax_rate = np.loadtxt('O1_rate_minmax.txt')
    x_min_rate = x_minmax_rate[0]
    x_max_rate = x_minmax_rate[1]
    x_min_drate = np.loadtxt('O1_drate_min.txt')
    x_max_drate = np.loadtxt('O1_drate_max.txt')
    #x_min_s1s2 = np.loadtxt('O1_s1s2_min.txt')
    x_max_s1s2 = np.max(np.loadtxt('O1_s1s2_max.txt'))


# ## Data to match emcee

# !ls ../data/andresData/28-05-24-files/examples-to-match-emcee/mDM50GeV-sigma2e-47-thetapidiv2

# !ls ../data/andresData/new-bilby-O1-O4-saved0/new-bilby/O1/examples-to-match-emcee/mDM50GeV-sigma5e-47-thetapidiv4

# +
# where are your files?
datFolder = ['../data/andresData/28-05-24-files/examples-to-match-emcee/mDM50GeV-sigma2e-47-thetapidiv2/']
#datFolder = ['../data/andresData/new-bilby-O1-O4-saved0/new-bilby/O1/examples-to-match-emcee/mDM50GeV-sigma5e-47-thetapidiv4/']
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

neutrino_floor_minuspidiv2 = np.loadtxt('../data/andresData/28-05-24-files/O1-O4-nufloor/O1-nufloor/floor_rate_minuspidiv2.txt', skiprows = 1, delimiter = ',')
neutrino_floor_minuspidiv4 = np.loadtxt('../data/andresData/28-05-24-files/O1-O4-nufloor/O1-nufloor/floor_rate_minuspidiv4.txt', skiprows = 1, delimiter = ',')
neutrino_floor_pluspidiv2 = np.loadtxt('../data/andresData/28-05-24-files/O1-O4-nufloor/O1-nufloor/floor_rate_pidiv2.txt', skiprows = 1, delimiter = ',')
neutrino_floor_pluspidiv4 = np.loadtxt('../data/andresData/28-05-24-files/O1-O4-nufloor/O1-nufloor/floor_rate_pidiv4.txt', skiprows = 1, delimiter = ',')
neutrino_floor_zero = np.loadtxt('../data/andresData/28-05-24-files/O1-O4-nufloor/O1-nufloor/floor_rate_zero.txt', skiprows = 1, delimiter = ',')
neutrino_mDM = np.loadtxt('../data/andresData/28-05-24-files/O1-O4-nufloor/O1-nufloor/mDM_range.txt', skiprows = 1, delimiter = ',')

# ## Xenon data
#
# from https://arxiv.org/pdf/2007.08796.pdf (Figure 6)

xenon_nt_5s   = np.loadtxt('../data/xenon_nt_5sigma.csv', skiprows = 1, delimiter = ',')
xenon_nt_3s   = np.loadtxt('../data/xenon_nt_3sigma.csv', skiprows = 1, delimiter = ',')
xenon_nt_90cl = np.loadtxt('../data/xenon_nt_90cl.csv', skiprows = 1, delimiter = ',')

# !ls ../data/andresData/BL-constraints-PARAO1/BL-constraints/

# +
masses = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/masses.txt')

rate_90_CL_pi2  = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-rate-thetapi2.txt')
rate_90_CL_pi4  = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-rate-thetapi4.txt')
rate_90_CL_0    = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-rate-theta0.txt')
rate_90_CL_mpi2 = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-rate-thetampi2.txt')
rate_90_CL_mpi4 = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-rate-thetampi4.txt')

rate_current_pi2  = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-rate-thetapi2-current.txt')
rate_current_pi4  = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-rate-thetapi4-current.txt')
rate_current_0    = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-rate-theta0-current.txt')
rate_current_mpi2 = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-rate-thetampi2-current.txt')
rate_current_mpi4 = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-rate-thetampi4-current.txt')

s1s2_90_CL_pi2  = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-s1s2-thetapi2.txt')
s1s2_90_CL_pi4  = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-s1s2-thetapi4.txt')
s1s2_90_CL_0    = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-s1s2-theta0.txt')
s1s2_90_CL_mpi2 = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-s1s2-thetampi2.txt')
s1s2_90_CL_mpi4 = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-s1s2-thetampi4.txt')

s1s2_current_pi2  = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-s1s2-thetapi2-current.txt')
s1s2_current_pi4  = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-s1s2-thetapi4-current.txt')
s1s2_current_0    = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-s1s2-theta0-current.txt')
s1s2_current_mpi2 = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-s1s2-thetampi2-current.txt')
s1s2_current_mpi4 = np.loadtxt('../data/andresData/BL-constraints-PARAO1/BL-constraints/BL-s1s2-thetampi4-current.txt')

# +
#masses = [   6.,            7.15758847,    8.53851211,   10.1858593,    12.1510315,
#   14.49534716,   17.29195494,   20.6281162,    24.60792777,   29.35557,
#   35.01918154,   41.77548165,   49.83528427,   59.45007598,   70.91986302,
#   84.60253227,  100.9250182,   120.39662436,  143.62491499,  171.33467249,
#  204.39051261,  243.82386261,  290.86514447,  346.98216721,  413.92592634,
#  493.78523936,  589.05192234,  702.69854089,  838.27116191, 1000.        ]
#
#rate_90_CL_mpi2 = [1e-30, 9.910290834505643e-45, 1.9764046437241126e-45, 5.919108497192993e-46, 2.3766605492517762e-46, 1.1827198265296277e-46, 6.94952214737333e-47, 4.676316696444454e-47, 3.4984774788038983e-47, 2.8626303132543127e-47, 2.5187706622571404e-47, 2.353437548525283e-47, 2.317572577446609e-47, 2.380382461207293e-47, 2.5265487056090517e-47, 2.7540737082463723e-47, 3.072100827596053e-47, 3.476925670955084e-47, 3.98848218247829e-47, 4.609004079813401e-47, 5.369516100367399e-47, 6.284251076018573e-47, 7.388202827083679e-47, 8.710326915475426e-47, 1.0296908522138497e-46, 1.2191933553266653e-46, 1.445121614289286e-46, 1.7172094991573469e-46, 2.0399753907649344e-46, 2.4242470582926507e-46]
#rate_3sigma_CL_mpi2 = [1e-30, 1.8282154417109256e-44, 3.645993085526049e-45, 1.0919325367889005e-45, 4.384355586517443e-46, 2.181823039946922e-46, 1.2820201409104695e-46, 8.626669685448377e-47, 6.453878358979698e-47, 5.28086641074437e-47, 4.646496172619808e-47, 4.3415080833868634e-47, 4.2753490139218856e-47, 4.391212710567875e-47, 4.660844375259439e-47, 5.080599855740955e-47, 5.667308269811642e-47, 6.414121390798629e-47, 7.357792603604449e-47, 8.502492862460597e-47, 9.905504665854631e-47, 1.1592973305972363e-46, 1.3629443844477e-46, 1.6068466079486477e-46, 1.8995420810427556e-46, 2.249105941433231e-46, 2.665886187727732e-46, 3.1678436584611554e-46, 3.763267028005681e-46, 4.472140983650492e-46]
#rate_5sigma_CL_mpi2 = [1e-30, 3.084860640421453e-44, 6.15210325934557e-45, 1.8424999524306333e-45, 7.397922366813523e-46, 3.681528781130952e-46, 2.163231768195962e-46, 1.455617508252734e-46, 1.0890094382530603e-46, 8.91064644557344e-47, 7.84026560152357e-47, 7.325677730289659e-47, 7.214052091553618e-47, 7.409540933285479e-47, 7.864474677087234e-47, 8.572724903355606e-47, 9.56274150099759e-47, 1.0823008813916151e-46, 1.2415257043512687e-46, 1.4346651650929384e-46, 1.6714073354983725e-46, 1.956165459409317e-46, 2.299769770645324e-46, 2.7113058065940438e-46, 3.2052313728875645e-46, 3.7950481921138885e-46, 4.498270674476589e-46, 5.345302762546376e-46, 6.3499904813924996e-46, 7.546041438689659e-46]
#s1s2_90_CL_mpi2 = [1.0619419081102384e-44, 9.976302661413967e-46, 1.9461656000344537e-46, 5.759528655521897e-47, 2.328921334780173e-47, 1.1734622993389403e-47, 7.050120801611516e-48, 4.8676143706645146e-48, 3.742859805173811e-48, 3.1450746527405807e-48, 2.841471041666698e-48, 2.7179182161081923e-48, 2.7312911599894027e-48, 2.8533001667322844e-48, 3.0648285767816883e-48, 3.372686514513663e-48, 3.7895766192183405e-48, 4.30723242583275e-48, 4.963874828345563e-48, 5.745722005776048e-48, 6.708786336062602e-48, 7.859356477951912e-48, 9.260694662337173e-48, 1.0913295066182416e-47, 1.2919112970054339e-47, 1.530400000932227e-47, 1.8171992399636867e-47, 2.1569466085019325e-47, 2.563437751948877e-47, 3.044550959289485e-47]
#
#s1s2_3sigma_CL_mpi2 = [2.1459962062700256e-44, 2.0281327126893962e-45, 3.993545030213811e-46, 1.190445600956618e-46, 4.811837752564172e-47, 2.4201793526866577e-47, 1.4475225175585887e-47, 9.952119183971967e-48, 7.630419297056005e-48, 6.399513208075681e-48, 5.76759539362942e-48, 5.507149356009182e-48, 5.524663245804412e-48, 5.765086562435588e-48, 6.187875971556214e-48, 6.805040681946083e-48, 7.636831383194547e-48, 8.675394228028093e-48, 9.993857959129346e-48, 1.1571700881364655e-47, 1.3503419323986558e-47, 1.5811952677650325e-47, 1.8632854494309163e-47, 2.196503491800532e-47, 2.5983215365391114e-47, 3.0785718904904683e-47, 3.656240215368193e-47, 4.3377217578682767e-47, 5.155362982358702e-47, 6.126604566223954e-47]
#
#s1s2_5sigma_CL_mpi2 = [4.08103752331147e-44, 3.8728582352232607e-45, 7.6749453724152e-46, 2.2975988969221325e-46, 9.288997446135513e-47, 4.664935622870835e-47, 2.782878649228596e-47, 1.9095180953326634e-47, 1.4594221461508327e-47, 1.222298847466356e-47, 1.0995347246562162e-47, 1.047606356704776e-47, 1.0489241738296843e-47, 1.0933045131046113e-47, 1.1721516337816972e-47, 1.2873092696940238e-47, 1.443004318368611e-47, 1.6385958732743452e-47, 1.887680081811898e-47, 2.1849135296206085e-47, 2.548145522834008e-47, 2.9833902123045253e-47, 3.5167347974486045e-47, 4.14419793643973e-47, 4.900072164200674e-47, 5.809726041596581e-47, 6.895739696510738e-47, 8.1791181513169555e-47, 9.725433988761545e-47, 1.1558035478070863e-46]


#rate_90_CL_mpi4 = [1e-30, 1.43825216090322e-44, 2.8699821414076195e-45, 8.6401933958261e-46, 3.4682217023765245e-46, 1.7239280743434108e-46, 1.017451325301827e-46, 6.859557695523621e-47, 5.144456092649627e-47, 4.225699773888627e-47, 3.7350857749433793e-47, 3.5017202563560774e-47, 3.4573090458003934e-47, 3.560098691734027e-47, 3.7928759107690453e-47, 4.1550883699140577e-47, 4.629402021528735e-47, 5.247328428794538e-47, 6.023040304018992e-47, 6.96643929921065e-47, 8.124598274404164e-47, 9.51241986405831e-47, 1.1189834410612705e-46, 1.319444500308371e-46, 1.560587023962522e-46, 1.8476324867487353e-46, 2.1908366232616337e-46, 2.6036959364350005e-46, 3.0911276531609846e-46, 3.676034482964511e-46]
#rate_3sigma_CL_mpi4 = [1e-30, 2.6532392408048125e-44, 5.2944594095527764e-45, 1.59391381873503e-45, 6.398033616234805e-46, 3.180237961439163e-46, 1.876963404590438e-46, 1.2654243291028962e-46, 9.490299908821703e-47, 7.79539222059149e-47, 6.890351191752662e-47, 6.459860358546562e-47, 6.377934236020872e-47, 6.567551740632349e-47, 6.996956762320059e-47, 7.665133580107009e-47, 8.540122470423635e-47, 9.680084344059601e-47, 1.1111114620486086e-46, 1.2851408639216373e-46, 1.4987912203892635e-46, 1.7548212641058267e-46, 2.064261281339446e-46, 2.4340414944156763e-46, 2.8788988555856844e-46, 3.408440103870903e-46, 4.041560892271134e-46, 4.803185034987173e-46, 5.702406834960865e-46, 6.7813847028770745e-46]
#rate_5sigma_CL_mpi4 = [1e-30, 4.4770579269580073e-44, 8.933707923201575e-45, 2.6894935738883364e-45, 1.0795848399001583e-45, 5.366219421389376e-46, 3.1671302724026788e-46, 2.135230712315799e-46, 1.6013433157789002e-46, 1.3153589207748542e-46, 1.1626565904658753e-46, 1.0900188310485612e-46, 1.0761946426548247e-46, 1.1081903029699763e-46, 1.1806438311485e-46, 1.2933812440927907e-46, 1.4410143865219537e-46, 1.6333682940370094e-46, 1.874858201187061e-46, 2.1684971576897497e-46, 2.5289793740325873e-46, 2.9610056361028905e-46, 3.483175639574182e-46, 4.1070833436235776e-46, 4.8577022634871454e-46, 5.751306880452453e-46, 6.819540385182175e-46, 8.104622650397802e-46, 9.622065396457924e-46, 1.1442661264776677e-45]
#
#s1s2_90_CL_mpi4 = [1.5477413501073623e-44, 1.4466175002958687e-45, 2.8232228684707058e-46, 8.386878352787113e-47, 3.385337158438867e-47, 1.71132122237464e-47, 1.031634995313122e-47, 7.137543765053557e-48, 5.487970537050901e-48, 4.638448820153851e-48, 4.205247096196908e-48, 4.031492243311884e-48, 4.0636633268858154e-48, 4.2483396468809285e-48, 4.5818084859795295e-48, 5.0587099036861876e-48, 5.683715371588287e-48, 6.464603190105587e-48, 7.453528378858146e-48, 8.64802790945283e-48, 1.0095230226517636e-47, 1.1824685229688134e-47, 1.3965580938403128e-47, 1.6464141594975775e-47, 1.9494809440369615e-47, 2.3114081509129643e-47, 2.738649210668812e-47, 3.252684034789243e-47, 3.866683418694712e-47, 4.603862141786373e-47]
#
#s1s2_3sigma_CL_mpi4 = [3.128546255846256e-44, 2.9428529340905316e-45, 5.79925318493001e-46, 1.7343403794790734e-46, 7.003197494415542e-47, 3.5298794686133123e-47, 2.1189623962859752e-47, 1.459336683842081e-47, 1.119873309049649e-47, 9.432221299369674e-48, 8.53459749259063e-48, 8.171069303319607e-48, 8.221955530491628e-48, 8.585153913678582e-48, 9.248453820073871e-48, 1.0207654434324311e-47, 1.1465160664510788e-47, 1.3031958655872847e-47, 1.5011819271746491e-47, 1.7414408354718562e-47, 2.0337237282472676e-47, 2.381126283365761e-47, 2.8105161339048157e-47, 3.3149011906000035e-47, 3.9239179190462322e-47, 4.650072682653817e-47, 5.5117733233030245e-47, 6.548392712052427e-47, 7.779921195033683e-47, 9.262894245959301e-47]
#
#s1s2_5sigma_CL_mpi4 = [5.946443217992828e-44, 5.623918737770394e-45, 1.1151312752319629e-45, 3.351666958438074e-46, 1.3522679806266494e-46, 6.805150653466494e-47, 4.073788083953823e-47, 2.7979930466652327e-47, 2.1441991875866932e-47, 1.8011014273253216e-47, 1.6258255906053823e-47, 1.5540369591779803e-47, 1.560761999703445e-47, 1.6276362085429184e-47, 1.7514661658349695e-47, 1.9329024458531643e-47, 2.1685467619021242e-47, 2.463076937031817e-47, 2.835373553501041e-47, 3.289997826070074e-47, 3.841073072432922e-47, 4.495557746048668e-47, 5.304835891514678e-47, 6.259612317282813e-47, 7.404400197556772e-47, 8.772576411246788e-47, 1.0407222478429901e-46, 1.2358237957979001e-46, 1.4677954258754814e-46, 1.748359361237009e-46]
#
#
#rate_90_CL_0 = [1e-30, 1e-30, 1e-30, 1.9903219062814267e-44, 8.065968877867656e-45, 4.0630262452460636e-45, 2.425856293096977e-45, 1.6595650341176398e-45, 1.2698131517119861e-45, 1.0627244843443503e-45, 9.563797345536075e-46, 9.165552515821799e-46, 9.232228803005497e-46, 9.676631190674436e-46, 1.0459874246930938e-45, 1.1560565425831849e-45, 1.3044093220561576e-45, 1.4901011363780166e-45, 1.7208527347577682e-45, 2.001088064105342e-45, 2.3403007034244858e-45, 2.749002300258578e-45, 3.2402891039592816e-45, 3.834173264090879e-45, 4.531102356842308e-45, 5.3739231721476476e-45, 6.380533028693007e-45, 7.591370667978488e-45, 9.016779356231514e-45, 1.0723240162469789e-44]
#rate_3sigma_CL_0 = [1e-30, 1e-30, 1e-30, 3.671673714326486e-44, 1.4879865128498246e-44, 7.495305144984484e-45, 4.47512520423837e-45, 3.0615157662232174e-45, 2.3424970761262072e-45, 1.9604750025552174e-45, 1.7642996720895458e-45, 1.6908337006005942e-45, 1.7031338741047514e-45, 1.7851144257310098e-45, 1.9296000881834623e-45, 2.1326455896714858e-45, 2.406319128079929e-45, 2.74888809841544e-45, 3.174574578069321e-45, 3.691526998253023e-45, 4.3172913401608883e-45, 5.0712809459224126e-45, 5.977569401096031e-45, 7.073122491978654e-45, 8.35882701929173e-45, 9.913637440477221e-45, 1.1770547521040635e-44, 1.4004292607775532e-44, 1.6633893960786995e-44, 1.9781800758962873e-44]
#rate_5sigma_CL_0 = [1e-30, 1e-30, 1e-30, 6.19529785198132e-44, 2.510745449636043e-44, 1.2647176377056927e-44, 7.551081007969723e-45, 5.165912068913874e-45, 3.952607181775532e-45, 3.3080425290215894e-45, 2.977028114241646e-45, 2.853053394324923e-45, 2.8738107956481675e-45, 3.0121516312372613e-45, 3.2559481376161545e-45, 3.598535740665291e-45, 4.0602933304561427e-45, 4.6383380569711346e-45, 5.3566892396091635e-45, 6.228929439159464e-45, 7.284760030436621e-45, 8.557082367391114e-45, 1.0086354663324128e-44, 1.193482536563794e-44, 1.4104289255663506e-44, 1.6728011911748207e-44, 1.986104745390008e-44, 2.3629999237688835e-44, 2.8067397487335247e-44, 3.337899617073794e-44]
#
#s1s2_90_CL_0 = [1e-30, 3.2872084630794357e-44, 6.457499178981217e-45, 1.9351001877190904e-45, 7.88621675250456e-46, 4.027690268443309e-46, 2.4534211515178562e-46, 1.7193160144891477e-46, 1.3470622090062517e-46, 1.1524199341173035e-46, 1.0588143677562598e-46, 1.0332340790642102e-46, 1.0572677981383817e-46, 1.120832488632681e-46, 1.2238101275754417e-46, 1.3619451067538243e-46, 1.5458903221052447e-46, 1.7732784244892818e-46, 2.0513824940266443e-46, 2.3930966014928976e-46, 2.798533021775021e-46, 3.2956315991865383e-46, 3.891938767421522e-46, 4.6072664351003716e-46, 5.449089651347666e-46, 6.474201419276107e-46, 7.678472255301389e-46, 9.133057781680382e-46, 1.0853088240500258e-45, 1.2909059414625752e-45]
#
#s1s2_3sigma_CL_0 = [1e-30, 6.678231140729928e-44, 1.3255539797230196e-44, 3.9974395804332914e-45, 1.631660131816534e-45, 8.303973350300836e-46, 5.039473448001447e-46, 3.519129284399025e-46, 2.747639328381017e-46, 2.3460139472045995e-46, 2.152508157755323e-46, 2.0979122498368626e-46, 2.143910159634823e-46, 2.270789825313152e-46, 2.4765873108902876e-46, 2.755072896246727e-46, 3.127550014453438e-46, 3.585507191862626e-46, 4.145237351602713e-46, 4.834777701547619e-46, 5.658187076757489e-46, 6.657558679135044e-46, 7.859080845007184e-46, 9.309313225640506e-46, 1.1007127304224335e-45, 1.3070365627159282e-45, 1.5508176240748815e-45, 1.845263970010331e-45, 2.1916672731410636e-45, 2.6068618698132625e-45]
#
#s1s2_5sigma_CL_0 = [1e-30, 1e-30, 2.5498662437384414e-44, 7.716089018379301e-45, 3.152769764241796e-45, 1.6016613728679506e-45, 9.699551832363556e-46, 6.75048949961619e-46, 5.262315278688925e-46, 4.482493814028965e-46, 4.106473127806106e-46, 3.996913702204314e-46, 4.079118020170734e-46, 4.3166275524990576e-46, 4.703425084635579e-46, 5.231799011567827e-46, 5.9365404166859754e-46, 6.7996833553637675e-46, 7.85838697677591e-46, 9.168318164076503e-46, 1.072999784928511e-45, 1.2614678946366554e-45, 1.4891279360912298e-45, 1.7648972789547108e-45, 2.0852467432029457e-45, 2.475554629535603e-45, 2.939865126738766e-45, 3.496207273491527e-45, 4.15173964265213e-45, 4.941137225161864e-45]
#
#
#rate_90_CL_pi4 = [1e-30, 2.906182300415843e-44, 5.759730032806367e-45, 1.7245285666975144e-45, 6.916225863745588e-46, 3.4243847435710445e-46, 2.0127028087532424e-46, 1.348158983087235e-46, 1.0059613486124065e-46, 8.176210262061704e-47, 7.169095845003212e-47, 6.658134287925807e-47, 6.524068902449704e-47, 6.662610839642606e-47, 7.054986456995876e-47, 7.679599520848893e-47, 8.531504371874938e-47, 9.652449201093244e-47, 1.1049471468662936e-46, 1.2750694262569281e-46, 1.4843782226582946e-46, 1.735202428356784e-46, 2.039062572692539e-46, 2.402422560842984e-46, 2.838583052129327e-46, 3.3621468928031022e-46, 3.982408419310285e-46, 4.7291779167543874e-46, 5.618388021527719e-46, 6.680972554559102e-46]
#rate_3sigma_CL_pi4 = [1e-30, 5.361253237039604e-44, 1.0625338113887795e-44, 3.181355438695762e-45, 1.2758710256742852e-45, 6.317167917387456e-46, 3.712957477932563e-46, 2.487012382453642e-46, 1.8557673570447926e-46, 1.5083125074083405e-46, 1.3225251966908947e-46, 1.228267671210296e-46, 1.203536603789018e-46, 1.2290934617715881e-46, 1.3014752592944346e-46, 1.4166994613181789e-46, 1.5738570875206362e-46, 1.7806544833447335e-46, 2.0383684805935737e-46, 2.3521814073980474e-46, 2.738306799957099e-46, 3.2010365700453595e-46, 3.76158315809293e-46, 4.431880090802511e-46, 5.23651317496297e-46, 6.2023581631757155e-46, 7.346556382222748e-46, 8.72418067608579e-46, 1.0364586857807796e-45, 1.2324734169003432e-45]
#rate_5sigma_CL_pi4 = [1e-30, 9.045992863359626e-44, 1.792883891440869e-44, 5.3681310690127374e-45, 2.1528513870912262e-45, 1.0659404567094088e-45, 6.265104966603683e-46, 4.19645866253001e-46, 3.1313615782903165e-46, 2.545044698271001e-46, 2.231573179681139e-46, 2.0725382581966794e-46, 2.0308103696083186e-46, 2.0739315788158628e-46, 2.196057076442122e-46, 2.39046777294079e-46, 2.6556402356936665e-46, 3.0046011012247647e-46, 3.439487649899176e-46, 3.968965754216182e-46, 4.620468839111134e-46, 5.401318810484916e-46, 6.3471493557730445e-46, 7.478109615853301e-46, 8.835834919814091e-46, 1.0465685071693913e-45, 1.2396243923479842e-45, 1.4720751669097812e-45, 1.748896128449018e-45, 2.0796289166601098e-45]
#
#s1s2_90_CL_pi4 = [3.094470692702348e-44, 2.92785922204475e-45, 5.662111643353509e-46, 1.6772827224150381e-46, 6.766047273634877e-47, 3.400336435985329e-47, 2.042812360485882e-47, 1.4030506936780068e-47, 1.0783923451920157e-47, 9.006228853147484e-48, 8.110828455132255e-48, 7.724422861303692e-48, 7.736412969119341e-48, 8.041002079193982e-48, 8.623077880689494e-48, 9.46445515838474e-48, 1.0581311705828464e-47, 1.2053174178425845e-47, 1.382785349334657e-47, 1.6004344852597067e-47, 1.8668619168381222e-47, 2.1875401660106343e-47, 2.5711783560108646e-47, 3.028203742788492e-47, 3.5830128530557754e-47, 4.247131148374275e-47, 5.029380597656903e-47, 5.974925680735205e-47, 7.111506787448957e-47, 8.4478888423148975e-47]
#
#s1s2_3sigma_CL_pi4 = [6.250510452424507e-44, 5.950211352854165e-45, 1.1627459740378695e-45, 3.468633891920919e-46, 1.3986375319896517e-46, 7.011326161033591e-47, 4.1938518324433076e-47, 2.868213866101341e-47, 2.198677771002682e-47, 1.8318414552664468e-47, 1.6456625682946943e-47, 1.5644109415130988e-47, 1.5642290430127536e-47, 1.6232349528487527e-47, 1.7391744943843227e-47, 1.9081592722742711e-47, 2.1322579910270791e-47, 2.425691555255151e-47, 2.7818386825702904e-47, 3.2200118813749195e-47, 3.756393564536732e-47, 4.397649707793579e-47, 5.168952948233552e-47, 6.091526419468587e-47, 7.203824955608382e-47, 8.535744898954386e-47, 1.0115203274387183e-46, 1.201321671020857e-46, 1.4291231627088409e-46, 1.6981859982417085e-46]
#
#s1s2_5sigma_CL_pi4 = [1e-30, 1.1359870791025666e-44, 2.2352699296688696e-45, 6.697135370707352e-46, 2.700373874588386e-46, 1.3512628620008139e-46, 8.061613814853833e-47, 5.501296009811277e-47, 4.205063210243301e-47, 3.4992732278431447e-47, 3.135040671156521e-47, 2.973536142140724e-47, 2.9675952931579204e-47, 3.0744698818713833e-47, 3.2914806132537186e-47, 3.6083663935191013e-47, 4.027859066273931e-47, 4.576853975248807e-47, 5.24847156235881e-47, 6.076984830142735e-47, 7.084991335821603e-47, 8.288717392618537e-47, 9.746524194975366e-47, 1.148778150584972e-46, 1.3578141755941634e-46, 1.608778058879897e-46, 1.9077618755964205e-46, 2.26429655632897e-46, 2.6929202887539607e-46, 3.202184949233211e-46]
#
#
#rate_90_CL_pi2 = [1e-30, 9.910290834505643e-45, 1.9764046437241126e-45, 5.919108497192993e-46, 2.3766605492517762e-46, 1.1827198265296277e-46, 6.94952214737333e-47, 4.676316696444454e-47, 3.4984774788038983e-47, 2.8626303132543127e-47, 2.5187706622571404e-47, 2.353437548525283e-47, 2.317572577446609e-47, 2.380382461207293e-47, 2.5265487056090517e-47, 2.7540737082463723e-47, 3.072100827596053e-47, 3.476925670955084e-47, 3.98848218247829e-47, 4.609004079813401e-47, 5.369516100367399e-47, 6.284251076018573e-47, 7.388202827083679e-47, 8.710326915475426e-47, 1.0296908522138497e-46, 1.2191933553266653e-46, 1.445121614289286e-46, 1.7172094991573469e-46, 2.0399753907649344e-46, 2.4242470582926507e-46]
#rate_3sigma_CL_pi2 = [1e-30, 1.8282154417109256e-44, 3.645993085526049e-45, 1.0919325367889005e-45, 4.384355586517443e-46, 2.181823039946922e-46, 1.2820201409104695e-46, 8.626669685448377e-47, 6.453878358979698e-47, 5.28086641074437e-47, 4.646496172619808e-47, 4.3415080833868634e-47, 4.2753490139218856e-47, 4.391212710567875e-47, 4.660844375259439e-47, 5.080599855740955e-47, 5.667308269811642e-47, 6.414121390798629e-47, 7.357792603604449e-47, 8.502492862460597e-47, 9.905504665854631e-47, 1.1592973305972363e-46, 1.3629443844477e-46, 1.6068466079486477e-46, 1.8995420810427556e-46, 2.249105941433231e-46, 2.665886187727732e-46, 3.1678436584611554e-46, 3.763267028005681e-46, 4.472140983650492e-46]
#rate_5sigma_CL_pi2 = [1e-30, 3.084860640421453e-44, 6.15210325934557e-45, 1.8424999524306333e-45, 7.397922366813523e-46, 3.681528781130952e-46, 2.163231768195962e-46, 1.455617508252734e-46, 1.0890094382530603e-46, 8.91064644557344e-47, 7.84026560152357e-47, 7.325677730289659e-47, 7.214052091553618e-47, 7.409540933285479e-47, 7.864474677087234e-47, 8.572724903355606e-47, 9.56274150099759e-47, 1.0823008813916151e-46, 1.2415257043512687e-46, 1.4346651650929384e-46, 1.6714073354983725e-46, 1.956165459409317e-46, 2.299769770645324e-46, 2.7113058065940438e-46, 3.2052313728875645e-46, 3.7950481921138885e-46, 4.498270674476589e-46, 5.345302762546376e-46, 6.3499904813924996e-46, 7.546041438689659e-46]
#
#s1s2_90_CL_pi2 = [1.0619419081102384e-44, 9.976302661413967e-46, 1.9461656000344537e-46, 5.759528655521897e-47, 2.328921334780173e-47, 1.1734622993389403e-47, 7.050120801611516e-48, 4.8676143706645146e-48, 3.742859805173811e-48, 3.1450746527405807e-48, 2.841471041666698e-48, 2.7179182161081923e-48, 2.7312911599894027e-48, 2.8533001667322844e-48, 3.0648285767816883e-48, 3.372686514513663e-48, 3.7895766192183405e-48, 4.30723242583275e-48, 4.963874828345563e-48, 5.745722005776048e-48, 6.708786336062602e-48, 7.859356477951912e-48, 9.260694662337173e-48, 1.0913295066182416e-47, 1.2919112970054339e-47, 1.530400000932227e-47, 1.8171992399636867e-47, 2.1569466085019325e-47, 2.563437751948877e-47, 3.044550959289485e-47]
#
#s1s2_3sigma_CL_pi2 = [2.1459962062700256e-44, 2.0281327126893962e-45, 3.993545030213811e-46, 1.190445600956618e-46, 4.811837752564172e-47, 2.4201793526866577e-47, 1.4475225175585887e-47, 9.952119183971967e-48, 7.630419297056005e-48, 6.399513208075681e-48, 5.76759539362942e-48, 5.507149356009182e-48, 5.524663245804412e-48, 5.765086562435588e-48, 6.187875971556214e-48, 6.805040681946083e-48, 7.636831383194547e-48, 8.675394228028093e-48, 9.993857959129346e-48, 1.1571700881364655e-47, 1.3503419323986558e-47, 1.5811952677650325e-47, 1.8632854494309163e-47, 2.196503491800532e-47, 2.5983215365391114e-47, 3.0785718904904683e-47, 3.656240215368193e-47, 4.3377217578682767e-47, 5.155362982358702e-47, 6.126604566223954e-47]
#
#s1s2_5sigma_CL_pi2 = [4.08103752331147e-44, 3.8728582352232607e-45, 7.6749453724152e-46, 2.2975988969221325e-46, 9.288997446135513e-47, 4.664935622870835e-47, 2.782878649228596e-47, 1.9095180953326634e-47, 1.4594221461508327e-47, 1.222298847466356e-47, 1.0995347246562162e-47, 1.047606356704776e-47, 1.0489241738296843e-47, 1.0933045131046113e-47, 1.1721516337816972e-47, 1.2873092696940238e-47, 1.443004318368611e-47, 1.6385958732743452e-47, 1.887680081811898e-47, 2.1849135296206085e-47, 2.548145522834008e-47, 2.9833902123045253e-47, 3.5167347974486045e-47, 4.14419793643973e-47, 4.900072164200674e-47, 5.809726041596581e-47, 6.895739696510738e-47, 8.1791181513169555e-47, 9.725433988761545e-47, 1.1558035478070863e-46]
# -

# ## EMCEE data

# !ls ../data/andresData/28-05-24-files/emcee-21046-and-51047/

# +
# INPUTS

mdm_emcee = 50
sigma_emcee = np.log10(5e-47)
theta_emcee= np.pi/2

m_dm  = np.log10(mdm_emcee) # m_{DM} [GeV]
sigma = sigma_emcee # sigma [cm^2]
theta = theta_emcee

# OPEN THE SAVED DATA

h5filename = '../data/andresData/28-05-24-files/emcee-21046-and-51047/run_emcee_rate_mDM' + str(mdm_emcee) + '_sigma2e-46_theta' + str(theta_emcee) + '.h5'
reader     = emcee.backends.HDFBackend(h5filename)
MCMC_rate = reader.get_chain(flat=True)

h5filename  = '../data/andresData/28-05-24-files/emcee-21046-and-51047/run_emcee_drate_mDM' + str(mdm_emcee) + '_sigma2e-46_theta' + str(theta_emcee) + '.h5'
reader      = emcee.backends.HDFBackend(h5filename)
MCMC_drate = reader.get_chain(flat=True)

h5filename = '../data/andresData/28-05-24-files/emcee-21046-and-51047/run_emcee_s1s2bin_mDM' + str(mdm_emcee) + '_sigma2e-46_theta' + str(theta_emcee) + '.h5'
reader     = emcee.backends.HDFBackend(h5filename)
MCMC_s1s1 = reader.get_chain(flat=True)
# -

# ## Multinest

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

# !ls ../data/andresData/28-05-24-files/bilby-multinest-21047/bilby-multinest/drate/pm_PyMultiNest-drate-21047

# !ls ../data/andresData/28-05-24-files/bilby-51047-OBSsaved0/

bilby_rate = bilby.result.read_in_result(filename='../data/andresData/28-05-24-files/bilby-multinest-21047/bilby-21047-OBSsaved0/rate-21047_result.json')
bilby_drate = bilby.result.read_in_result(filename='../data/andresData/28-05-24-files/bilby-multinest-21047/bilby-21047-OBSsaved0/drate-21047_result.json')
bilby_s1s2 = bilby.result.read_in_result(filename='../data/andresData/28-05-24-files/bilby-multinest-21047/bilby-21047-OBSsaved0/s1s2bin-OP2-21047_result.json')


bilby_rate = bilby.result.read_in_result(filename='../data/andresData/28-05-24-files/bilby-51047-OBSsaved0/rate-51047_result.json')
bilby_drate = bilby.result.read_in_result(filename='../data/andresData/28-05-24-files/bilby-51047-OBSsaved0/drate-C-51047_result.json')
bilby_s1s2 = bilby.result.read_in_result(filename='../data/andresData/28-05-24-files/bilby-51047-OBSsaved0/s1s2bin-OP2-51047_result.json')

bilby_rate = bilby.result.read_in_result(filename='../data/andresData/28-05-24-files/bilby-multinest-21047/bilby-multinest/rate/PyMultiNest-rate-21047_result.json')
bilby_drate = bilby.result.read_in_result(filename='../data/andresData/28-05-24-files/bilby-multinest-21047/bilby-multinest/drate/PyMultiNest-drate-21047_result.json')

# !ls ../data/andresData/new-bilby-O1-O4-saved0/new-bilby/O1/examples-to-match-emcee/bilby-results

bilby_rate = bilby.result.read_in_result(filename='../data/andresData/new-bilby-O1-O4-saved0/new-bilby/O1/examples-to-match-emcee/bilby-results/rate-51047-thetapidiv4_result.json')
bilby_drate = bilby.result.read_in_result(filename='../data/andresData/new-bilby-O1-O4-saved0/new-bilby/O1/examples-to-match-emcee/bilby-results/drate-51047-thetapidiv4_result.json')
bilby_s1s2 = bilby.result.read_in_result(filename='../data/andresData/new-bilby-O1-O4-saved0/new-bilby/O1/examples-to-match-emcee/bilby-results/s1s2bin-OP2-51047-thetapidiv4_result.json')

# # Let's play with SWYFT

# ## Using only the total rate with background 

# ### Training

x_rate = np.log10(rate_trainset) # Observable. Input data.

x_max_rate

# +
# Let's normalize everything between 0 and 1

#pars_min = np.min(pars_trainset, axis = 0)
#pars_max = np.max(pars_trainset, axis = 0)

pars_norm = (pars_trainset - pars_min) / (pars_max - pars_min)

#x_min_rate = np.min(x_rate, axis = 0)
#x_max_rate = np.max(x_rate, axis = 0)

x_norm_rate = (x_rate - x_min_rate) / (x_max_rate - x_min_rate)
#x_norm_rate = x_rate  / x_max_rate

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
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O1_norm2_rate_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
trainer_rate = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 2000, precision = 64, callbacks=[early_stopping_callback, checkpoint_callback, cb])
network_rate = Network_rate()

# +
x_test_rate = np.log10(rate_testset)
#x_norm_test_rate = x_test_rate / x_max_rate
x_norm_test_rate = (x_test_rate - x_min_rate) / (x_max_rate - x_min_rate)
x_norm_test_rate = x_norm_test_rate.reshape(len(x_norm_test_rate), 1)

pars_norm_test = (pars_testset - pars_min) / (pars_max - pars_min)

# We have to build a swyft.Samples object that will handle the data
samples_test_rate = swyft.Samples(x = x_norm_test_rate, z = pars_norm_test)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_test_rate = swyft.SwyftDataModule(samples_test_rate, fractions = [0., 0., 1], batch_size = 32)
trainer_rate.test(network_rate, dm_test_rate)
# -

#ckpt_path = swyft.best_from_yaml("./logs/O1_norm2_rate.yaml")
ckpt_path = '/home/martinrios/martin/trabajos/eftDM/codes/logs/O1_final_rate.ckpt'
# ---------------------------------------------- 
# It converges to val_loss = -1.18 at epoch ~50
# ---------------------------------------------- 

# +
x_test_rate = np.log10(rate_testset)
#x_norm_test_rate = x_test_rate / x_max_rate
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
#x_norm_rate = x_rate / x_max_rate
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
par = 2
parameter = np.asarray(predictions_rate[0].params[:,par,0]) * (pars_max[par] - pars_min[par]) + pars_min[par]
ratios = np.exp(np.asarray(predictions_rate[0].logratios[:,par]))

ind_sort  = np.argsort(parameter)
ratios    = ratios[ind_sort]
parameter = parameter[ind_sort]

plt.plot(parameter, ratios)

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
ax[1,0].set_ylim(1e-50,1e-43)
ax[1,1].set_ylim(1e-50,1e-43)

ax[0,0].set_xlabel('')
ax[0,0].set_ylabel('$P(m|x)$')
ax[0,0].set_xticks([])
ax[1,1].set_ylabel('')
ax[1,1].set_yticks([])
ax[1,1].set_xlabel('$P(\sigma|x)$')
#ax[1,0].grid(which = 'both')
#plt.savefig('../graph/2d_custom_posteriors_' + str(i) + '_rate.pdf')
# -

# ## Only using the total diff_rate

# ### Training

x_drate = np.log10(diff_rate_trainset) # Observable. Input data. 
#x_drate = diff_rate_trainset # Observable. Input data. 

# +
# Let's normalize everything between 0 and 1

#pars_min = np.min(pars_trainset, axis = 0)
#pars_max = np.max(pars_trainset, axis = 0)

pars_norm = (pars_trainset - pars_min) / (pars_max - pars_min)

#x_min_drate = np.min(x_drate, axis = 0)
#x_max_drate = np.max(x_drate, axis = 0)

x_norm_drate = (x_drate - x_min_drate) / (x_max_drate - x_min_drate)
#x_norm_drate = x_drate / x_max_drate

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
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O1_norm_drate_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
trainer_drate = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 2000, precision = 64, callbacks=[early_stopping_callback, checkpoint_callback, cb])
network_drate = Network()


# +
x_test_drate = np.log10(diff_rate_testset)
#x_test_drate = diff_rate_testset
x_norm_test_drate = (x_test_drate - x_min_drate) / (x_max_drate - x_min_drate)
#x_norm_test_drate = x_test_drate / x_max_drate

pars_norm_test = (pars_testset - pars_min) / (pars_max - pars_min)

# We have to build a swyft.Samples object that will handle the data
samples_test_drate = swyft.Samples(x = x_norm_test_drate, z = pars_norm_test)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_test_drate = swyft.SwyftDataModule(samples_test_drate, fractions = [0., 0., 1], batch_size = 32)
trainer_drate.test(network_drate, dm_test_drate)
# -

#ckpt_path = swyft.best_from_yaml("./logs/O1_norm_drate.yaml")
ckpt_path = '/home/martinrios/martin/trabajos/eftDM/codes/logs/O1_final_drate.ckpt'
# ---------------------------------------------- 
# It converges to val_loss = -1.8 @ epoch 20
# ---------------------------------------------- 

# +
x_test_drate = np.log10(diff_rate_testset)
#x_test_drate = diff_rate_testset
x_norm_test_drate = (x_test_drate - x_min_drate) / (x_max_drate - x_min_drate)
#x_norm_test_drate = x_test_drate / x_max_drate

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
#x_drate = emcee_diff_rate
x_norm_drate = (x_drate - x_min_drate) / (x_max_drate - x_min_drate)
#x_norm_drate = x_drate / x_max_drate

# +
# First let's create some observation from some "true" theta parameters
i = 0#np.random.randint(24)
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
par = 2
parameter = np.asarray(predictions_drate[0].params[:,par,0]) * (pars_max[par] - pars_min[par]) + pars_min[par]
ratios = np.exp(np.asarray(predictions_drate[0].logratios[:,par]))

ind_sort  = np.argsort(parameter)
ratios    = ratios[ind_sort]
parameter = parameter[ind_sort]

plt.plot(parameter, ratios)

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
ax[1,0].set_ylim(1e-50,1e-43)
ax[1,1].set_ylim(1e-50,1e-43)

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

#pars_min = np.min(pars_trainset, axis = 0)
#pars_max = np.max(pars_trainset, axis = 0)

pars_norm = (pars_trainset - pars_min) / (pars_max - pars_min)

#x_min_s1s2 = np.min(x_s1s2, axis = 0)
#x_max_s1s2 = np.max(x_s1s2)

x_norm_s1s2 = x_s1s2
#ind_nonzero = np.where(x_max_s1s2 > 0)
#x_norm_s1s2[:,ind_nonzero[0], ind_nonzero[1]] = (x_s1s2[:,ind_nonzero[0], ind_nonzero[1]] - x_min_s1s2[ind_nonzero[0], ind_nonzero[1]]) / (x_max_s1s2[ind_nonzero[0], ind_nonzero[1]] - x_min_s1s2[ind_nonzero[0], ind_nonzero[1]])
x_norm_s1s2 = x_s1s2 / x_max_s1s2


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
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O1_norm_s1s2_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
trainer_s1s2 = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 2500, precision = 64, callbacks=[early_stopping_callback, checkpoint_callback, cb])
network_s1s2 = Network()

# +
x_norm_test_s1s2 = s1s2_testset[:,:-1,:-1] # Observable. Input data. I am cutting a bit the images to have 96x96
x_norm_test_s1s2 = x_norm_test_s1s2 / x_max_s1s2 # Observable. Input data. I am cutting a bit the images to have 96x96
x_norm_test_s1s2 = x_norm_test_s1s2.reshape(len(x_norm_test_s1s2), 1, 96, 96)

pars_norm_test = (pars_testset - pars_min) / (pars_max - pars_min)

# We have to build a swyft.Samples object that will handle the data
samples_test_s1s2 = swyft.Samples(x = x_norm_test_s1s2, z = pars_norm_test)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_test_s1s2 = swyft.SwyftDataModule(samples_test_s1s2, fractions = [0., 0., 1], batch_size = 32)
trainer_s1s2.test(network_s1s2, dm_test_s1s2)
# -

#ckpt_path = swyft.best_from_yaml("./logs/O1_norm_s1s2.yaml")
ckpt_path = '/home/martinrios/martin/trabajos/eftDM/codes/logs/O1_final_s1s2.ckpt'
# ---------------------------------------
# Min val loss value at 48 epochs. -3.31
# ---------------------------------------


# +
x_norm_test_s1s2 = s1s2_testset[:,:-1,:-1] # Observable. Input data. I am cutting a bit the images to have 96x96
x_norm_test_s1s2 = x_norm_test_s1s2 / x_max_s1s2 # Observable. Input data. I am cutting a bit the images to have 96x96
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

x_norm_s1s2 = emcee_s1s2[:,:-1,:-1] / x_max_s1s2
#x_norm_s1s2 = x_s1s2 = s1s2_testset[:,:-1,:-1] / x_max_s1s2

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
start = time.time()
predictions_s1s2 = trainer_s1s2.infer(network_s1s2, obs, prior_samples)
stop = time.time()
print('It takes ' + str(stop-start) + ' seconds')

# +
par = 2
parameter = np.asarray(predictions_s1s2[0].params[:,par,0]) * (pars_max[par] - pars_min[par]) + pars_min[par]
ratios = np.exp(np.asarray(predictions_s1s2[0].logratios[:,par]))

ind_sort  = np.argsort(parameter)
ratios    = ratios[ind_sort]
parameter = parameter[ind_sort]

plt.plot(parameter, ratios)

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
ax[1,0].set_ylim(1e-50,1e-43)
ax[1,1].set_ylim(1e-50,1e-43)

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
s1s2_samples = bilby_s1s2.samples

fig = corner.corner(rate_samples, smooth = 2.5, levels=[0.9], bins = 30, plot_density=False, color = color_rate, fill_contours=False)

axes = fig.get_axes()

axes[0].hist(s1s2_samples[:,0], color = 'black', bins = 30)
corner.corner(drate_samples, smooth = 2, levels=[0.9], bins = 30, plot_density=False, color = color_drate, fill_contours=False, fig = fig)

corner.corner(s1s2_samples, smooth = 2, levels=[0.9], bins = 30, plot_density=False, color = color_s1s2, fill_contours=False, fig = fig)

plt.show()
# -

from matplotlib.patches import Patch



# +
rate  = True
drate = True
s1s2  = True
prob = [0.9]
fig = bilby_s1s2.plot_corner(outdir='../graph/', color = 'gainsboro', levels=[0.9], smooth = 1, bins = 15, alpha = 0.6, truth = None)
#fig = bilby_drate.plot_corner(outdir='.', color = 'grey', levels=prob, smooth = 0.1)
#fig = bilby_rate.plot_corner(outdir='.', color = 'grey', levels=prob, smooth = 0.1)

#fig = corner.corner(rate_samples, smooth = 2.5, levels = [0.9], bins = 30, plot_density = False, color = 'black', fill_contours = False, linestyles = ['--'])
#corner.corner(drate_samples, smooth = 2, levels = [0.9], bins = 30, plot_density = False, color = 'magenta', fill_contours = False, 
#              fig = fig, contour_kwargs = {'linestyles':'--'}, contourf_kwargs = {'alpha':0})
#corner.corner(s1s2_samples, smooth = 2, levels = [0.9], bins = 30, plot_density = False, color = color_s1s2, fill_contours = False, fig = fig, ls = '--')

axes = fig.get_axes()


ax = axes[0]
ax.cla()
ax.hist(s1s2_samples[:,0], color = 'grey', bins = 15, zorder = 0, histtype = 'step', density = True)

if rate:
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 0, 
                 fill = False, linestyles = ['solid',':'], color = color_rate, fac = 180, probs = prob)
if drate: 
    plot1d_emcee(ax, [predictions_rate, predictions_drate], pars_true, par = 0, 
             fill = False, linestyles = ['solid',':'], color = color_drate, fac = 180, probs = prob)
if s1s2:
    #plot1d_emcee(ax, [predictions_s1s2], pars_true, par = 0, 
    #             fill = False, linestyles = ['solid',':'], color = color_s1s2_2, fac = 100, probs = prob)
    plot1d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 0, 
                 fill = False, linestyles = ['solid',':'], color = color_s1s2, fac = 100, probs = prob)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
ax.set_xlim([1, 3])

ax = axes[3]
if rate:
    plot2d_emcee(ax, [predictions_rate], pars_true, fill = False, line = True, linestyles = ['solid','--'], 
                 color = color_rate, probs = prob, zorder = 2, nvals = 20, smooth = 0.7)
if drate:
    plot2d_emcee(ax, [predictions_rate, predictions_drate], pars_true, fill = False, line = True, linestyles = ['solid','--'], 
                 color = color_drate, probs = prob, zorder = 3, nvals = 20, smooth = 0.7)
if s1s2:
    #plot2d_emcee(ax, [predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
    #             color = color_s1s2_2, probs = prob, zorder = 4, nvals = 40)
    plot2d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                 color = color_s1s2, probs = prob, zorder = 4, nvals = 40, smooth = 0.7)
ax.set_ylabel('Log$_{10}(\\sigma^{SI} \ $[cm$^{2}$])', fontsize = 10)
ax.set_xlim([1, 3])
ax.set_ylim([-49.5, -43])

ax = axes[4]
ax.cla()
ax.hist(s1s2_samples[:,1], color = 'grey', bins = 15, zorder = 0, histtype = 'step', density = True)

if rate:
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 1, 
                 flip = False, fill = False, linestyles = ['solid', ':'], color = color_rate, fac = 220, probs = prob)
if drate:
    plot1d_emcee(ax, [predictions_rate, predictions_drate], pars_true, par = 1, 
                 flip = False, fill = False, linestyles = ['solid', ':'], color = color_drate, fac = 190, probs = prob)
if s1s2:
    #plot1d_emcee(ax, [predictions_s1s2], pars_true, par = 1, 
    #             flip = False, fill = False, linestyles = ['solid', ':'], color = color_s1s2_2, fac = 70, probs = prob)
    plot1d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 1, 
                 flip = False, fill = False, linestyles = ['solid', ':'], color = color_s1s2, fac = 70, probs = prob)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title('')
ax.set_xlim([-49.5, -43])
ax.set_xticks([])
ax.set_yticks([])

ax = axes[6]

if rate:
    plot2d_emcee_m_theta(ax, [predictions_rate], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_rate, probs = prob, zorder = 2, smooth = 1.2)
if drate:
    plot2d_emcee_m_theta(ax, [predictions_rate, predictions_rate, predictions_drate], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_drate, probs = prob, zorder = 2, smooth = 1.2)
if s1s2:
    #plot2d_emcee_m_theta(ax, [predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
    #            color = color_s1s2_2, probs = prob, zorder = 2, smooth = 0.7)
    plot2d_emcee_m_theta(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_s1s2, probs = prob, zorder = 2, smooth = 1.2)
ax.set_ylabel('$\\theta$', fontsize = 10)
ax.set_xlabel('Log$_{10}(m_{\chi} \ $[GeV])', fontsize = 10)
ax.set_xlim([1, 3])
ax.set_ylim([-1.6, 1.6])

ax = axes[7]

if rate:
    plot2d_emcee_sigma_theta(ax, [predictions_rate], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_rate, probs = prob, zorder = 2, smooth = 0.7)
if drate:
    plot2d_emcee_sigma_theta(ax, [predictions_rate, predictions_drate], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_drate, probs = prob, zorder = 2, smooth = 0.7)
if s1s2:
    #plot2d_emcee_sigma_theta(ax, [predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
    #            color = color_s1s2_2, probs = prob, zorder = 2, smooth = 0.7)
    plot2d_emcee_sigma_theta(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_s1s2, probs = prob, zorder = 2, smooth = 0.7)
ax.set_xlabel('Log$_{10}(\\sigma^{SI} \ $[cm$^{2}$])', fontsize = 10)
ax.set_ylabel('')
ax.set_xlim([-49.5, -43])
ax.set_ylim([-1.6, 1.6])

ax = axes[8]
ax.clear()
ax.hist(s1s2_samples[:,2], color = 'grey', bins = 15, zorder = 0, histtype = 'step', range = (-1.6,1.6), density = True)

if rate:
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 2, 
                 flip = False, fill = False, linestyles = ['solid',':'], color = color_rate, fac = 100, probs = prob)
if drate:
    plot1d_emcee(ax, [predictions_rate, predictions_drate], pars_true, par = 2, 
                 flip = False, fill = False, linestyles = ['solid',':'], color = color_drate, fac = 100, probs = prob)
if s1s2:
    #plot1d_emcee(ax, [predictions_s1s2], pars_true, par = 2, 
    #             flip = False, fill = False, linestyles = ['solid',':'], color = color_s1s2_2, fac = 100, probs = prob)
    plot1d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 2, 
                 flip = False, fill = False, linestyles = ['solid',':'], color = color_s1s2, fac = 100, probs = prob)
ax.set_ylabel('')
ax.set_title('')
ax.set_xlim([-1.6, 1.6])
ax.set_xticks([-1.5,0,1.5])
ax.set_xticklabels(['-1.5','0.0', '1.5'], rotation = 45)
ax.set_yticks([])
ax.text(0.47,-0.39, '$\\theta$', fontsize = 10, transform = ax.transAxes)
#ax.set_xlabel('$\\theta$', fontsize = 12)

custom_lines = []
labels = ['Rate', 'Rate + Dif. Rate', 'Rate + Dif. Rate + cS1-cS2']
#labels = ['Rate', 'Dif. Rate', 'cS1-cS2']
#labels = ['Rate + Dif. Rate + cS1-cS2', 'cS1-cS2']
markers = ['solid','solid', 'solid']
colors = [color_rate, color_drate, color_s1s2]
#colors = [color_s1s2, color_s1s2_2]
for i in range(len(labels)):
    custom_lines.append( Line2D([0],[0], linestyle = markers[i], color = colors[i], 
            label = labels[i], lw = 2) )

custom_lines.append( Patch(facecolor='gainsboro', edgecolor='gainsboro',
                         label='MCMC cS1-cS2') )
axes[0].legend(handles = custom_lines, frameon = False, loc = 'lower left', bbox_to_anchor=(1.2,0.25), fontsize = 10)
axes[0].text(1.5,0.8, '$\\mathcal{O}_{1}$', fontsize = 12, transform = axes[0].transAxes)

axes[3].scatter(emcee_pars[0,0], emcee_pars[0,1], marker = 'D', color = 'black', zorder = 4)
axes[3].scatter(emcee_pars[0,0], emcee_pars[0,1], marker = 'D', color = 'yellow', zorder = 5, s = 10)
axes[6].scatter(emcee_pars[0,0], emcee_pars[0,2], marker = 'D', color = 'black', zorder = 4)
axes[6].scatter(emcee_pars[0,0], emcee_pars[0,2], marker = 'D', color = 'yellow', zorder = 5, s = 10)
axes[7].scatter(emcee_pars[0,1], emcee_pars[0,2], marker = 'D', color = 'black', zorder = 4)
axes[7].scatter(emcee_pars[0,1], emcee_pars[0,2], marker = 'D', color = 'yellow', zorder = 5, s = 10)

#fig.savefig('../graph/SWYFT_BILBY_comparison_O1_m_{:.2f}_s_{:.2f}_t_{:.2f}_s1s2.pdf'.format(emcee_pars[0,0],emcee_pars[0,1],emcee_pars[0,2]), bbox_inches='tight')
fig
# -
color_s1s2_2 = 'm'

# +
rate  = False
drate = False
s1s2  = True
prob = [0.9]
fig = bilby_s1s2.plot_corner(outdir='../graph/', color = 'gainsboro', levels=[0.9], smooth = 1, bins = 15, alpha = 0.6, truth = None)
#fig = bilby_drate.plot_corner(outdir='.', color = 'grey', levels=prob, smooth = 0.1)
#fig = bilby_rate.plot_corner(outdir='.', color = 'grey', levels=prob, smooth = 0.1)

#fig = corner.corner(rate_samples, smooth = 2.5, levels = [0.9], bins = 30, plot_density = False, color = 'black', fill_contours = False, linestyles = ['--'])
#corner.corner(drate_samples, smooth = 2, levels = [0.9], bins = 30, plot_density = False, color = 'magenta', fill_contours = False, 
#              fig = fig, contour_kwargs = {'linestyles':'--'}, contourf_kwargs = {'alpha':0})
#corner.corner(s1s2_samples, smooth = 2, levels = [0.9], bins = 30, plot_density = False, color = color_s1s2, fill_contours = False, fig = fig, ls = '--')

axes = fig.get_axes()


ax = axes[0]
ax.cla()
ax.hist(s1s2_samples[:,0], color = 'grey', bins = 15, zorder = 0, histtype = 'step', density = True)

if rate:
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 0, 
                 fill = False, linestyles = ['solid',':'], color = color_rate, fac = 180, probs = prob)
if drate: 
    plot1d_emcee(ax, [predictions_rate, predictions_drate], pars_true, par = 0, 
             fill = False, linestyles = ['solid',':'], color = color_drate, fac = 180, probs = prob)
if s1s2:
    plot1d_emcee(ax, [predictions_s1s2], pars_true, par = 0, 
                 fill = False, linestyles = ['solid',':'], color = color_s1s2_2, fac = 100, probs = prob)
    plot1d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 0, 
                 fill = False, linestyles = ['solid',':'], color = color_s1s2, fac = 100, probs = prob)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
ax.set_xlim([1, 3])

ax = axes[3]
if rate:
    plot2d_emcee(ax, [predictions_rate], pars_true, fill = False, line = True, linestyles = ['solid','--'], 
                 color = color_rate, probs = prob, zorder = 2, nvals = 20, smooth = 0.7)
if drate:
    plot2d_emcee(ax, [predictions_rate, predictions_drate], pars_true, fill = False, line = True, linestyles = ['solid','--'], 
                 color = color_drate, probs = prob, zorder = 3, nvals = 20, smooth = 0.7)
if s1s2:
    plot2d_emcee(ax, [predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                 color = color_s1s2_2, probs = prob, zorder = 4, nvals = 40)
    plot2d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                 color = color_s1s2, probs = prob, zorder = 4, nvals = 40, smooth = 0.7)
ax.set_ylabel('Log$_{10}(\\sigma^{SI} \ $[cm$^{2}$])', fontsize = 10)
ax.set_xlim([1, 3])
ax.set_ylim([-49.5, -43])

ax = axes[4]
ax.cla()
ax.hist(s1s2_samples[:,1], color = 'grey', bins = 15, zorder = 0, histtype = 'step', density = True)

if rate:
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 1, 
                 flip = False, fill = False, linestyles = ['solid', ':'], color = color_rate, fac = 220, probs = prob)
if drate:
    plot1d_emcee(ax, [predictions_rate, predictions_drate], pars_true, par = 1, 
                 flip = False, fill = False, linestyles = ['solid', ':'], color = color_drate, fac = 190, probs = prob)
if s1s2:
    plot1d_emcee(ax, [predictions_s1s2], pars_true, par = 1, 
                 flip = False, fill = False, linestyles = ['solid', ':'], color = color_s1s2_2, fac = 70, probs = prob)
    plot1d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 1, 
                 flip = False, fill = False, linestyles = ['solid', ':'], color = color_s1s2, fac = 70, probs = prob)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title('')
ax.set_xlim([-49.5, -43])
ax.set_xticks([])
ax.set_yticks([])

ax = axes[6]

if rate:
    plot2d_emcee_m_theta(ax, [predictions_rate], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_rate, probs = prob, zorder = 2, smooth = 0.7)
if drate:
    plot2d_emcee_m_theta(ax, [predictions_rate, predictions_drate], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_drate, probs = prob, zorder = 2, smooth = 0.7)
if s1s2:
    plot2d_emcee_m_theta(ax, [predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_s1s2_2, probs = prob, zorder = 2, smooth = 0.7)
    plot2d_emcee_m_theta(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_s1s2, probs = prob, zorder = 2, smooth = 0.7)
ax.set_ylabel('$\\theta$', fontsize = 10)
ax.set_xlabel('Log$_{10}(m_{\chi} \ $[GeV])', fontsize = 10)
ax.set_xlim([1, 3])
ax.set_ylim([-1.6, 1.6])

ax = axes[7]

if rate:
    plot2d_emcee_sigma_theta(ax, [predictions_rate], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_rate, probs = prob, zorder = 2, smooth = 0.7)
if drate:
    plot2d_emcee_sigma_theta(ax, [predictions_rate, predictions_drate], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_drate, probs = prob, zorder = 2, smooth = 0.7)
if s1s2:
    plot2d_emcee_sigma_theta(ax, [predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_s1s2_2, probs = prob, zorder = 2, smooth = 0.7)
    plot2d_emcee_sigma_theta(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                color = color_s1s2, probs = prob, zorder = 2, smooth = 0.7)
ax.set_xlabel('Log$_{10}(\\sigma^{SI} \ $[cm$^{2}$])', fontsize = 10)
ax.set_ylabel('')
ax.set_xlim([-49.5, -43])
ax.set_ylim([-1.6, 1.6])

ax = axes[8]
ax.clear()
ax.hist(s1s2_samples[:,2], color = 'grey', bins = 15, zorder = 0, histtype = 'step', range = (-1.6,1.6), density = True)

if rate:
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 2, 
                 flip = False, fill = False, linestyles = ['solid',':'], color = color_rate, fac = 100, probs = prob)
if drate:
    plot1d_emcee(ax, [predictions_rate, predictions_drate], pars_true, par = 2, 
                 flip = False, fill = False, linestyles = ['solid',':'], color = color_drate, fac = 100, probs = prob)
if s1s2:
    plot1d_emcee(ax, [predictions_s1s2], pars_true, par = 2, 
                 flip = False, fill = False, linestyles = ['solid',':'], color = color_s1s2_2, fac = 100, probs = prob)
    plot1d_emcee(ax, [predictions_rate, predictions_drate, predictions_s1s2], pars_true, par = 2, 
                 flip = False, fill = False, linestyles = ['solid',':'], color = color_s1s2, fac = 100, probs = prob)
ax.set_ylabel('')
ax.set_title('')
ax.set_xlim([-1.6, 1.6])
ax.set_xticks([-1.5,0,1.5])
ax.set_xticklabels(['-1.5','0.0', '1.5'], rotation = 45)
ax.set_yticks([])
ax.text(0.47,-0.39, '$\\theta$', fontsize = 10, transform = ax.transAxes)

custom_lines = []

labels = ['Rate + Dif. Rate + cS1-cS2', 'cS1-cS2']
markers = ['solid','solid']
colors = [color_s1s2, color_s1s2_2]
for i in range(len(labels)):
    custom_lines.append( Line2D([0],[0], linestyle = markers[i], color = colors[i], 
            label = labels[i], lw = 2) )

custom_lines.append( Patch(facecolor='gainsboro', edgecolor='gainsboro',
                         label='MCMC cS1-cS2') )
axes[0].legend(handles = custom_lines, frameon = False, loc = 'lower left', bbox_to_anchor=(1.2,0.25), fontsize = 10)
axes[0].text(1.5,0.8, '$\\mathcal{O}_{1}$', fontsize = 12, transform = axes[0].transAxes)

axes[3].scatter(emcee_pars[0,0], emcee_pars[0,1], marker = 'D', color = 'black', zorder = 4)
axes[3].scatter(emcee_pars[0,0], emcee_pars[0,1], marker = 'D', color = 'yellow', zorder = 5, s = 10)
axes[6].scatter(emcee_pars[0,0], emcee_pars[0,2], marker = 'D', color = 'black', zorder = 4)
axes[6].scatter(emcee_pars[0,0], emcee_pars[0,2], marker = 'D', color = 'yellow', zorder = 5, s = 10)
axes[7].scatter(emcee_pars[0,1], emcee_pars[0,2], marker = 'D', color = 'black', zorder = 4)
axes[7].scatter(emcee_pars[0,1], emcee_pars[0,2], marker = 'D', color = 'yellow', zorder = 5, s = 10)

#fig.savefig('../graph/SWYFT_BILBY_comparison_O1_m_{:.2f}_s_{:.2f}_t_{:.2f}_noComb.pdf'.format(emcee_pars[0,0],emcee_pars[0,1],emcee_pars[0,2]), bbox_inches='tight')
fig
# +
color_drate = 'dodgerblue'
color_s1s2 = 'm'

rate_samples = bilby_rate.samples[:,:2]
drate_samples = bilby_drate.samples[:,:2]
s1s2_samples = bilby_s1s2.samples[:,:2]

rate  = False
drate = False
s1s2  = True

prob = [0.9]

fig,axes = plt.subplots(2,2, width_ratios = [1,0.4], height_ratios = [0.4,1])

if rate:
    corner.corner(rate_samples, fig = fig, smooth = 1.8, levels=prob, bins = 15, plot_density=False, color = 'gainsboro', fill_contours=True)
if drate:
    corner.corner(drate_samples, fig = fig, smooth = 1.5, levels=prob, bins = 15, plot_density=False, color = 'gainsboro', fill_contours=True)
if s1s2:
    corner.corner(s1s2_samples, fig = fig, smooth = 1, levels=prob, bins = 15, plot_density=False, color = 'gainsboro', fill_contours=True)

#axes = fig.get_axes()

ax = axes[0,0]
ax.cla()

if rate:
    ax.hist(rate_samples[:,0], color = 'grey', bins = 15, zorder = 0, histtype = 'step', density = True)
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 0, 
                 fill = False, linestyles = ['solid',':'], color = color_rate, fac = 130, probs = prob)
if drate: 
    ax.hist(drate_samples[:,0], color = 'grey', bins = 35, zorder = 0, histtype = 'step', density = True)
    plot1d_emcee(ax, [predictions_rate, predictions_drate], pars_true, par = 0, 
             fill = False, linestyles = ['solid',':'], color = color_drate, fac = 130, probs = prob)
if s1s2:
    ax.hist(s1s2_samples[:,0], color = 'grey', bins = 15, zorder = 0, histtype = 'step', density = True)
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
    plot2d_emcee(ax, [predictions_drate], pars_true, fill = False, line = True, linestyles = ['solid','--'], 
                 color = color_drate, probs = prob, zorder = 3, nvals = 20, smooth = 1)
if s1s2:
    plot2d_emcee(ax, [predictions_s1s2], pars_true, fill = False, line = True, linestyles = ['solid', '--'], 
                 color = color_s1s2, probs = prob, zorder = 4, nvals = 40)
ax.set_ylabel('Log$_{10}(\\sigma^{SI} \ $[cm$^{2}$])', fontsize = 12)
ax.set_xlabel('Log$_{10}(m_{\chi} $[GeV])', fontsize = 12)
ax.set_xlim([1, 3])
ax.set_ylim([-49.5, -43])

ax.scatter(emcee_pars[0,0], emcee_pars[0,1], marker = 'D', color = 'black', zorder = 4)
ax.scatter(emcee_pars[0,0], emcee_pars[0,1], marker = 'D', color = 'yellow', zorder = 5, s = 10)

ax = axes[1,1]
ax.cla()

if rate:
    ax.hist(rate_samples[:,1], color = 'grey', bins = 15, zorder = 0, histtype = 'step', orientation="horizontal", density = True)
    plot1d_emcee(ax, [predictions_rate], pars_true, par = 1, 
                 flip = True, fill = False, linestyles = ['solid', ':'], color = color_rate, fac = 70, probs = prob)
if drate:
    ax.hist(drate_samples[:,1], color = 'grey', bins = 15, zorder = 0, histtype = 'step', orientation="horizontal", density = True)
    plot1d_emcee(ax, [predictions_drate], pars_true, par = 1, 
                 flip = True, fill = False, linestyles = ['solid', ':'], color = color_drate, fac = 80, probs = prob)
if s1s2:
    plot1d_emcee(ax, [predictions_s1s2], pars_true, par = 1, 
                 flip = True, fill = False, linestyles = ['solid', ':'], color = color_s1s2, fac = 60, probs = prob)
    ax.hist(s1s2_samples[:,1], color = 'grey', bins = 15, zorder = 0, histtype = 'step', orientation="horizontal", density = True)
    #hist = np.histogram(s1s2_samples[:,1], bins = 30)
    #ax.barplot(hist[0], hist[1][:-1], color = 'grey', zorder = 0)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title('')
#ax.set_xlim([-49.5, -43])
ax.set_xticks([])
ax.set_yticks([])

custom_lines = []

labels = ['cS1-cS2']
markers = ['solid']
colors = [color_s1s2]
for i in range(len(labels)):
    custom_lines.append( Line2D([0],[0], linestyle = markers[i], color = colors[i], 
            label = labels[i], lw = 2) )

custom_lines.append( Patch(facecolor='gainsboro', edgecolor='gainsboro',
                         label='MCMC cS1-cS2') )
ax.legend(handles = custom_lines, frameon = False, loc = 'lower left', bbox_to_anchor=(-0.1,1.07), fontsize = 12)
ax.text(0.3,1.35, '$\\mathcal{O}_{1}$', fontsize = 14, transform = ax.transAxes)

plt.savefig('../graph/SWYFT_BILBY_s1s2_comparison_O1_m_{:.2f}_s_{:.2f}_t_{:.2f}.pdf'.format(emcee_pars[0,0],emcee_pars[0,1],emcee_pars[0,2]), bbox_inches='tight')
# -

# # Other

res, xedges, yedges, _ = stats.binned_statistic_2d(pars[:,1], pars[:,2], rate_raw[:,1], 'mean', bins=50)

rate_raw[:,1]

X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
plt.contourf(X + (xedges[1] - xedges[0]) / 2, Y + (yedges[1] - yedges[0]) / 2, res.T, levels = [0,1,10,100,1000,1000000], colors = ['black','green','red','blue','magenta'])
#plt.xscale('log')
plt.colorbar()


