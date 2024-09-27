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
color_comb = 'purple'

# Check if gpu is available
if torch.cuda.is_available():
    device = 'gpu'
    print('Using GPU')
else:
    device = 'cpu'
    print('Using CPU')


# # Custom Functions

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
    
    s1s2_slices = s1s2_WIMP_slices #+ s1s2_er_slices + s1s2_ac_slices + s1s2_cevns_SM_slices + s1s2_radio_slices + s1s2_wall_slices
    rate_slices = np.sum(s1s2_slices, axis = 1) # Just to have the same as on the other notebooks. This already includes the backgrounds
    s1s2_slices = s1s2_slices.reshape(nobs_slices, 97, 97)

    diff_rate_slices = diff_rate_WIMP #+ diff_rate_er + diff_rate_ac + diff_rate_cevns_SM + diff_rate_radio + diff_rate_wall
    
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
    s_values = np.logspace(-49., -43.1, nvals)
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


# # Let's load the data

# !ls ../data/andresData/SI-run0and1/SI-run01

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
        # #%diff_rate_er       = np.loadtxt(folder + 'diff_rate_er.txt')
        # #%diff_rate_ac       = np.loadtxt(folder + 'diff_rate_ac.txt')
        # #%diff_rate_cevns_SM = np.loadtxt(folder + 'diff_rate_CEVNS-SM.txt')
        # #%diff_rate_radio    = np.loadtxt(folder + 'diff_rate_radiogenics.txt')
        # #%diff_rate_wall     = np.loadtxt(folder + 'diff_rate_wall.txt')
        
        s1s2_WIMP     = np.loadtxt(folder + 's1s2_WIMP.txt')
        # #%s1s2_er       = np.loadtxt(folder + 's1s2_er.txt')
        # #%s1s2_ac       = np.loadtxt(folder + 's1s2_ac.txt')
        # #%s1s2_cevns_SM = np.loadtxt(folder + 's1s2_CEVNS-SM.txt')
        # #%s1s2_radio    = np.loadtxt(folder + 's1s2_radiogenics.txt')
        # #%s1s2_wall     = np.loadtxt(folder + 's1s2_wall.txt')
    else:
        pars      = np.vstack((pars, np.loadtxt(folder + 'pars.txt'))) # pars[:,0] = mass ; pars[:,1] = cross-section ; pars[:,2] = theta
        rate_raw  = np.vstack((rate_raw, np.loadtxt(folder + 'rate.txt'))) # rate[:,0] = total expected events ; rate[:,1] = expected signal ; rate[:,2] = # events pseudo-experiment ; rate[:,3] = # signal events pseudo-experiment 
        
        diff_rate_WIMP     = np.vstack(( diff_rate_WIMP, np.loadtxt(folder + 'diff_rate_WIMP.txt')))
        # #%diff_rate_er       = np.vstack(( diff_rate_er, np.loadtxt(folder + 'diff_rate_er.txt')))
        # #%diff_rate_ac       = np.vstack(( diff_rate_ac, np.loadtxt(folder + 'diff_rate_ac.txt')))
        # #%diff_rate_cevns_SM = np.vstack(( diff_rate_cevns_SM, np.loadtxt(folder + 'diff_rate_CEVNS-SM.txt')))
        # #%diff_rate_radio    = np.vstack(( diff_rate_radio, np.loadtxt(folder + 'diff_rate_radiogenics.txt')))
        # #%diff_rate_wall     = np.vstack(( diff_rate_wall, np.loadtxt(folder + 'diff_rate_wall.txt')))
        
        s1s2_WIMP     = np.vstack((s1s2_WIMP, np.loadtxt(folder + 's1s2_WIMP.txt')))
        # #%s1s2_er       = np.vstack((s1s2_er, np.loadtxt(folder + 's1s2_er.txt')))
        # #%s1s2_ac       = np.vstack((s1s2_ac, np.loadtxt(folder + 's1s2_ac.txt')))
        # #%s1s2_cevns_SM = np.vstack((s1s2_cevns_SM, np.loadtxt(folder + 's1s2_CEVNS-SM.txt')))
        # #%s1s2_radio    = np.vstack((s1s2_radio, np.loadtxt(folder + 's1s2_radiogenics.txt')))
        # #%s1s2_wall     = np.vstack((s1s2_wall, np.loadtxt(folder + 's1s2_wall.txt')))
        
    
nobs = len(pars) # Total number of observations
print('We have ' + str(nobs) + ' observations...')

diff_rate = diff_rate_WIMP #+ diff_rate_er + diff_rate_ac + diff_rate_cevns_SM + diff_rate_radio + diff_rate_wall

s1s2 = s1s2_WIMP #+ s1s2_er + s1s2_ac + s1s2_cevns_SM + s1s2_radio + s1s2_wall
rate = np.sum(s1s2, axis = 1) # Just to have the same as on the other notebooks. This already includes the backgrounds
s1s2 = s1s2.reshape(nobs, 97, 97)

# Let's work with the log of the mass and cross-section

pars[:,0] = np.log10(pars[:,0])
pars[:,1] = np.log10(pars[:,1])

# Let's transform the diff_rate to counts per energy bin

#diff_rate = np.round(diff_rate * 362440)
# -

# This should be always zero
i = np.random.randint(nobs)
print(rate_raw[i,2] - rate[i])
print(rate_raw[i,2] - np.sum(diff_rate[i,:]))

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

# ## Ibarra

# +
ibarra_solid = np.loadtxt('../data/ibarra_cp1_solid.csv', skiprows = 1, delimiter = ',')
ibarra_dashed = np.loadtxt('../data/ibarra_cp1_dashed.csv', skiprows = 1, delimiter = ',')
ibarra_dotted = np.loadtxt('../data/ibarra_cp1_dotted.csv', skiprows = 1, delimiter = ',')

mu = ibarra_solid[:,0] * 1 / (ibarra_solid[:,0] + 1)
ibarra_solid[:,1] = ( 4 * ibarra_solid[:,1]**2 / (80**4) ) * (long_planck**2) * (masa_planck**2) * (mu**2) / np.pi
#ibarra_solid[:,1] = (2 * fac * ibarra_solid[:,1])**2 * (mu**2) / np.pi

mu = ibarra_dashed[:,0] * 1 / (ibarra_dashed[:,0] + 1)
ibarra_dashed[:,1] = ( 4 * ibarra_dashed[:,1]**2 / (80**4) ) * (long_planck**2) * (masa_planck**2) * (mu**2) / np.pi
#ibarra_dashed[:,1] = ibarra_dashed[:,1] / (0.08**2)
#ibarra_dashed[:,1] = (2 * fac * ibarra_dashed[:,1])**2 * (mu**2) / np.pi

mu = ibarra_dotted[:,0] * 1 / (ibarra_dotted[:,0] + 1)
ibarra_dotted[:,1] = ( 4 * ibarra_dotted[:,1]**2 / (80**4) ) * (long_planck**2) * (masa_planck**2) * (mu**2) / np.pi
#ibarra_dotted[:,1] = ibarra_dotted[:,1] / (0.08**2)
#ibarra_dotted[:,1] = (2 * fac * ibarra_dotted[:,1])**2 * (mu**2) / np.pi
# -

# ## Neutrino Fog

neutrino_fog = np.loadtxt('../data/neutrino_fog.csv', skiprows = 1, delimiter = ',')

neutrino_fog.shape

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
# -

# # Let's play with SWYFT

# ## Using only the total rate with background 

# ### Training

x_rate = rate_trainset # Observable. Input data.

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
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O1_noBKG_rate_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
trainer_rate = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 2000, precision = 64, callbacks=[early_stopping_callback, checkpoint_callback, cb])
network_rate = Network_rate()

# +
x_test_rate = rate_testset
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
    checkpoint_callback.to_yaml("./logs/O1_rate_noBKG.yaml") 
    ckpt_path = swyft.best_from_yaml("./logs/O1_rate_noBKG.yaml")
    email('Termino de entrenar rate O1')
    
else:
    ckpt_path = swyft.best_from_yaml("./logs/O1_rate_noBKG.yaml")

# ---------------------------------------------- 
# It converges to val_loss = -1.18 at epoch ~50
# ---------------------------------------------- 

# +
x_test_rate = rate_testset
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

if fit:
    val_loss = []
    train_loss = []
    for i in range(1, len(cb.collection)):
        train_loss.append( np.asarray(cb.train_loss[i].cpu()) )
        val_loss.append( np.asarray(cb.val_loss[i].cpu()) )
    
    plt.plot(train_loss, label = 'Train Loss')
    plt.plot(val_loss, label = 'Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../graph/O1_noBKG_loss_rate.pdf')

if fit:
    pars_prior    = np.random.uniform(low = 0, high = 1, size = (100_000, 3))
    prior_samples = swyft.Samples(z = pars_prior)
    
    coverage_samples = trainer_rate.test_coverage(network_rate, samples_test_rate[:50], prior_samples)
    
    fix, axes = plt.subplots(1, 3, figsize = (12, 4))
    for i in range(3):
        swyft.plot_zz(coverage_samples, "pars_norm[%i]"%i, ax = axes[i])
    plt.tight_layout()
    plt.savefig('../graph/O1_noBKG_Coverage_rate.pdf')

# ## Only using the total diff_rate

# ### Training

x_drate = diff_rate_trainset # Observable. Input data. 

# +
# Let's normalize everything between 0 and 1

pars_min = np.min(pars_trainset, axis = 0)
pars_max = np.max(pars_trainset, axis = 0)

pars_norm = (pars_trainset - pars_min) / (pars_max - pars_min)

x_min_drate = np.min(x_drate, axis = 0)
x_max_drate = np.max(x_drate, axis = 0)

x_norm_drate = (x_drate - x_min_drate) / (x_max_drate - x_min_drate)
# -

x_norm_drate = np.nan_to_num(x_norm_drate)

# +
fig,ax = plt.subplots(2,2, gridspec_kw = {'hspace':0.5, 'wspace':0.5})


for i in range(50):
    ax[0,0].plot(x_norm_drate[i])
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
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O1_noBKG_drate_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
trainer_drate = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 2000, precision = 64, callbacks=[early_stopping_callback, checkpoint_callback, cb])
network_drate = Network()


# +
x_test_drate = diff_rate_testset
x_norm_test_drate = (x_test_drate - x_min_drate) / (x_max_drate - x_min_drate)
x_norm_test_drate = np.nan_to_num(x_norm_test_drate)

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
    checkpoint_callback.to_yaml("./logs/O1_noBKG_drate.yaml") 
    ckpt_path = swyft.best_from_yaml("./logs/O1_noBKG_drate.yaml")
    email('Termino el entramiento del drate para O1_noBKG')
else:
    ckpt_path = swyft.best_from_yaml("./logs/O1_noBKG_drate.yaml")

# ---------------------------------------------- 
# It converges to val_loss = -1.8 @ epoch 20
# ---------------------------------------------- 

# +
x_test_drate = diff_rate_testset
x_norm_test_drate = (x_test_drate - x_min_drate) / (x_max_drate - x_min_drate)
x_norm_test_drate = np.nan_to_num(x_norm_test_drate)

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

if fit:
    val_loss = []
    train_loss = []
    for i in range(1, len(cb.collection)):
        train_loss.append( np.asarray(cb.train_loss[i].cpu()) )
        val_loss.append( np.asarray(cb.val_loss[i].cpu()) )

    plt.plot(val_loss, label = 'Val Loss')
    plt.plot(train_loss, label = 'Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../graph/O1_noBKG_loss_drate.pdf')

if fit:
    pars_prior    = np.random.uniform(low = 0, high = 1, size = (100_000, 3))
    prior_samples = swyft.Samples(z = pars_prior)
    
    coverage_samples = trainer_drate.test_coverage(network_drate, samples_test_drate[:50], prior_samples)
    
    fix, axes = plt.subplots(1, 3, figsize = (12, 4))
    for i in range(3):
        swyft.plot_zz(coverage_samples, "pars_norm[%i]"%i, ax = axes[i])
    plt.tight_layout()
    plt.savefig('../graph/O1_noBKG_Coverage_drate.pdf')

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
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O1_noBKG_s1s2_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
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

# +
fit = False
if fit:
    trainer_s1s2.fit(network_s1s2, dm_s1s2)
    checkpoint_callback.to_yaml("./logs/O1_noBKG_s1s2.yaml") 
    ckpt_path = swyft.best_from_yaml("./logs/O1_noBKG_s1s2.yaml")
else:
    ckpt_path = swyft.best_from_yaml("./logs/O1_noBKG_s1s2.yaml")

# ---------------------------------------
# Min val loss value at 48 epochs. -3.31
# ---------------------------------------
# -


trainer_s1s2.test(network_s1s2, dm_test_s1s2, ckpt_path = ckpt_path)

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

if fit:
    val_loss = []
    train_loss = []
    for i in range(1, len(cb.collection)):
        train_loss.append( np.asarray(cb.train_loss[i].cpu()) )
        val_loss.append( np.asarray(cb.val_loss[i].cpu()) )

    plt.plot(val_loss, label = 'Val Loss')
    plt.plot(train_loss, label = 'Train Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('../graph/O1_noBKG_loss_s1s2_temp.pdf')

if fit:
    pars_prior    = np.random.uniform(low = 0, high = 1, size = (100_000, 3))
    prior_samples = swyft.Samples(z = pars_prior)
    
    coverage_samples = trainer_s1s2.test_coverage(network_s1s2, samples_test_s1s2[:50], prior_samples)
    
    fix, axes = plt.subplots(1, 3, figsize = (12, 4))
    for i in range(3):
        swyft.plot_zz(coverage_samples, "pars_norm[%i]"%i, ax = axes[i])
    plt.tight_layout()
    plt.savefig('../graph/O1_noBKG_Coverage_s1s2.pdf')

# ## Combine everything

# ### Let's make the contour plot

# !ls ../data/andresData/O1-slices-5vecescadatheta/theta-0/SI-slices01-0/cross_sec_int_prob_sup_rate*

pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice(['../data/andresData/O1-slices-5vecescadatheta/theta-0/SI-slices01-0/'])

m_vals = np.logspace(np.min(pars_slices[:,0]), np.max(pars_slices[:,0]),30)
cross_vals = np.logspace(np.min(pars_slices[:,1]), np.max(pars_slices[:,1]),30)

x_rate = rate_slices
x_norm_rate = (x_rate - x_min_rate) / (x_max_rate - x_min_rate)
x_norm_rate = x_norm_rate.reshape(len(x_norm_rate), 1)

x_rate[490:510]

itest = 503
x_obs_rate = x_norm_rate[itest, :]
obs_rate = swyft.Sample(x = x_obs_rate)
predictions_rate = trainer_rate.infer(network_rate, obs_rate, prior_samples)


x_rate[itest]

x_norm_rate[itest,:]

# +
par = 1
parameter = np.asarray(predictions_rate[0].params[:,par,0]) * (pars_max[par] - pars_min[par]) + pars_min[par]
ratios = np.asarray(predictions_rate[0].logratios[:,par])
    
ratios = np.exp(ratios)
    
ind_sort  = np.argsort(parameter)
ratios    = ratios[ind_sort]
parameter = parameter[ind_sort]

plt.plot(parameter, np.exp(np.asarray(predictions_rate[0].logratios[:,par])[ind_sort]), color =color_rate, label = 'rate')
#plt.plot(parameter0, ratios0, ls=':')
# -

x_drate = diff_rate_slices
x_norm_drate = (x_drate - x_min_drate) / (x_max_drate - x_min_drate)
x_norm_drate = np.nan_to_num(x_norm_drate)

itest = 503
x_obs_drate = x_norm_drate[itest, :]
obs_drate = swyft.Sample(x = x_obs_drate)
predictions_drate = trainer_drate.infer(network_drate, obs_drate, prior_samples)


# +
par = 1
parameter = np.asarray(predictions_drate[0].params[:,par,0]) * (pars_max[par] - pars_min[par]) + pars_min[par]
ratios = np.asarray(predictions_drate[0].logratios[:,par])
    
ratios = np.exp(ratios)
    
ind_sort  = np.argsort(parameter)
ratios    = ratios[ind_sort]
parameter = parameter[ind_sort]

plt.plot(parameter, np.exp(np.asarray(predictions_drate[0].logratios[:,par])[ind_sort]), color =color_rate, label = 'rate')
#plt.plot(parameter0, ratios0, ls=':')

# +
plotMass = False
plotCross = False
iplot = [342,343,344]

cross_section_th = -49 # Cross-section threshold for 1d analysis 
m_min_th = 1 # Min Mass for 1d analysis
m_max_th = 2.6 # Max Mass for 1d analysis

rate  = True # Flag to use the information of the rate analysis
drate = True # Flag to use the information of the drate analysis
s1s2  = True # Flag to use the information of the s1s2 analysis

if rate: 
    flag = 'rate_T'
else:
    flag = 'rate_F'

if drate: 
    flag = flag + '_drate_T'
else:
    flag = flag + '_drate_F'

if s1s2: 
    flag = flag + '_s1s2_T'
else:
    flag = flag + '_s1s2_F'

flag = flag + '_noBKG'
force = True # Flag to force to compute everything again although it was pre-computed

thetas = ['0', 'minuspidiv2', 'minuspidiv4', 'pluspidiv2', 'pluspidiv4']
cross_sec_int_prob_sup_aux    = []
cross_sec_int_prob_sup_aux_sd = []
masses_int_prob_sup_aux       = []
masses_int_prob_sup_aux_sd    = []
masses_prob_sup_aux           = []
masses_prob_sup_aux_sd        = []
masses_prob_inf_aux           = []
masses_prob_inf_aux_sd        = []

# we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
pars_prior    = np.random.uniform(low = 0, high = 1, size = (10_000, 3))
prior_samples = swyft.Samples(z = pars_prior)
for theta in thetas:
    print('\n')
    print('\n')
    print('Analyzing theta = ' + theta)
    print('\n')
    print('\n')
    folders = ['../data/andresData/O1-slices-5vecescadatheta/theta-' + theta + '/SI-slices01-' + theta + '/',
               '../data/andresData/O1-slices-5vecescadatheta/theta-' + theta + '/SI-slices01-' + theta + '-v2/',
               '../data/andresData/O1-slices-5vecescadatheta/theta-' + theta + '/SI-slices01-' + theta + '-v3/',
               '../data/andresData/O1-slices-5vecescadatheta/theta-' + theta + '/SI-slices01-' + theta + '-v4/',
               '../data/andresData/O1-slices-5vecescadatheta/theta-' + theta + '/SI-slices01-' + theta + '-v5/'
             ]
    
    cross_sec_int_prob_sup_full = []
    
    masses_int_prob_sup_full = []
    masses_prob_sup_full     = []
    masses_prob_inf_full     = []

    for folder in folders:
        pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice([folder])
        
        if (
            os.path.exists(folder + 'cross_sec_int_prob_sup_' + flag + '.txt') &
            os.path.exists(folder + 'masses_int_prob_sup_' + flag + '.txt') &
            os.path.exists(folder + 'masses_prob_sup_' + flag + '.txt') &
            os.path.exists(folder + 'masses_prob_inf_' + flag + '.txt') 
           ) == False or force == True:
            # Let's normalize testset between 0 and 1
            
            pars_norm = (pars_slices - pars_min) / (pars_max - pars_min)
            
            x_norm_s1s2 = x_s1s2 = s1s2_slices[:,:-1,:-1]
               
            x_drate = diff_rate_slices
            x_norm_drate = (x_drate - x_min_drate) / (x_max_drate - x_min_drate)
            x_norm_drate = np.nan_to_num(x_norm_drate)
               
            x_rate = rate_slices
            x_norm_rate = (x_rate - x_min_rate) / (x_max_rate - x_min_rate)
            x_norm_rate = x_norm_rate.reshape(len(x_norm_rate), 1)
        
            cross_sec_int_prob_sup = np.ones(len(pars_norm)) * -99
            masses_int_prob_sup = np.ones(len(pars_norm)) * -99
            masses_prob_sup     = np.ones(len(pars_norm)) * -99
            masses_prob_inf     = np.ones(len(pars_norm)) * -99
               
            for itest in tqdm(range(len(pars_norm))):
                if s1s2:
                    x_obs_s1s2 = x_norm_s1s2[itest, :,:]
                    obs_s1s2 = swyft.Sample(x = x_obs_s1s2.reshape(1,96,96))
                    predictions_s1s2 = trainer_s1s2.infer(network_s1s2, obs_s1s2, prior_samples)

                if drate:
                    x_obs_drate = x_norm_drate[itest, :]
                    obs_drate = swyft.Sample(x = x_obs_drate)
                    predictions_drate = trainer_drate.infer(network_drate, obs_drate, prior_samples)

                if rate:
                    x_obs_rate = x_norm_rate[itest, :]
                    obs_rate = swyft.Sample(x = x_obs_rate)
                    predictions_rate = trainer_rate.infer(network_rate, obs_rate, prior_samples)
        
                # Cross-section
                par = 1 # 0 = mass, 1 = cross-section, 2 = theta
                parameter = np.asarray(predictions_rate[0].params[:,par,0]) * (pars_max[par] - pars_min[par]) + pars_min[par]
                ratios = np.zeros_like(np.asarray(predictions_rate[0].logratios[:,par]))
                if rate:  ratios = ratios + np.asarray(predictions_rate[0].logratios[:,par])
                if drate: ratios = ratios + np.asarray(predictions_drate[0].logratios[:,par])
                if s1s2:  ratios = ratios + np.asarray(predictions_s1s2[0].logratios[:,par])
                    
                ratios = np.exp(ratios)
                    
                ind_sort  = np.argsort(parameter)
                ratios    = ratios[ind_sort]
                parameter = parameter[ind_sort]
                
                # Let's compute the integrated probability for different threshold
                cr_th = np.argmin(np.abs(parameter - cross_section_th))
                cross_sec_int_prob_sup[itest] = trapezoid(ratios[cr_th:], parameter[cr_th:]) / trapezoid(ratios, parameter)

                
                if plotCross & (itest in iplot):
                    plt.plot(parameter, np.exp(np.asarray(predictions_rate[0].logratios[:,par])[ind_sort]), color =color_rate, label = 'rate')
                    plt.plot(parameter, np.exp(np.asarray(predictions_drate[0].logratios[:,par])[ind_sort]), color =color_drate, label = 'drate')
                    plt.plot(parameter, np.exp(np.asarray(predictions_s1s2[0].logratios[:,par])[ind_sort]), color =color_s1s2, label = 's1s2')
                    
                    plt.plot(parameter, ratios, color ='black', ls = '--')
                    
                    plt.xlabel('$\\sigma$')
                    plt.ylabel('$P$')
                    plt.legend()
                    plt.savefig('../graph/O1_CrossPosteriorsComb_' + flag + '_theta_' + theta + '_obs_' + str(itest) + '_folder_' + folder[-2:-1] + '.pdf')
                    plt.clf()
                #-----------------------------------------------------------------------------------------------------------
                
                # Mass
                par = 0 # 0 = mass, 1 = cross-section, 2 = theta
                parameter = np.asarray(predictions_rate[0].params[:,par,0]) * (pars_max[par] - pars_min[par]) + pars_min[par]
                ratios = np.zeros_like(np.asarray(predictions_rate[0].logratios[:,par]))
                if rate:  ratios = ratios + np.asarray(predictions_rate[0].logratios[:,par])
                if drate: ratios = ratios + np.asarray(predictions_drate[0].logratios[:,par])
                if s1s2:  ratios = ratios + np.asarray(predictions_s1s2[0].logratios[:,par])
                    
                ratios = np.exp(ratios)
                
                ind_sort  = np.argsort(parameter)
                ratios    = ratios[ind_sort]
                parameter = parameter[ind_sort]
                
                # Let's compute the integrated probability for different threshold            
                m_min = np.argmin(np.abs(parameter - m_min_th))
                m_max = np.argmin(np.abs(parameter - m_max_th))
                
                masses_int_prob_sup[itest] = trapezoid(ratios[m_min:m_max], parameter[m_min:m_max]) / trapezoid(ratios, parameter)
                masses_prob_sup[itest] = trapezoid(ratios[m_min:], parameter[m_min:]) / trapezoid(ratios, parameter)
                masses_prob_inf[itest] = trapezoid(ratios[:m_max], parameter[:m_max]) / trapezoid(ratios, parameter)

                if plotMass & (itest in iplot):
                    plt.plot(parameter, np.exp(np.asarray(predictions_rate[0].logratios[:,par])[ind_sort]), color =color_rate, label = 'rate')
                    plt.plot(parameter, np.exp(np.asarray(predictions_drate[0].logratios[:,par])[ind_sort]), color =color_drate, label = 'drate')
                    plt.plot(parameter, np.exp(np.asarray(predictions_s1s2[0].logratios[:,par])[ind_sort]), color =color_s1s2, label = 's1s2')
                    
                    plt.plot(parameter, ratios, color ='black', ls = '--')
                    
                    plt.xlabel('$log10(mass)$')
                    plt.ylabel('$P$')
                    plt.legend()
                    plt.savefig('../graph/O1_MassPosteriorsComb_' + flag + '_theta_' + theta + '_obs_' + str(itest) + '_folder_' + folder[-2:-1] + '.pdf')
                    plt.clf()
                    
                #-----------------------------------------------------------------------------------------------------------
                
            cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
            masses_int_prob_sup_full.append(masses_int_prob_sup)
            masses_prob_sup_full.append(masses_prob_sup)
            masses_prob_inf_full.append(masses_prob_inf)
                
            np.savetxt(folder + 'cross_sec_int_prob_sup_' + flag + '.txt', cross_sec_int_prob_sup)
            np.savetxt(folder + 'masses_int_prob_sup_' + flag + '.txt', masses_int_prob_sup)
            np.savetxt(folder + 'masses_prob_sup_' + flag + '.txt', masses_prob_sup)
            np.savetxt(folder + 'masses_prob_inf_' + flag + '.txt', masses_prob_inf)
        else:
            print('pre-computed')
            cross_sec_int_prob_sup = np.loadtxt(folder + 'cross_sec_int_prob_sup_' + flag + '.txt')
            masses_int_prob_sup    = np.loadtxt(folder + 'masses_int_prob_sup_' + flag + '.txt')
            masses_prob_sup        = np.loadtxt(folder + 'masses_prob_sup_' + flag + '.txt')
            masses_prob_inf        = np.loadtxt(folder + 'masses_prob_inf_' + flag + '.txt')
    
            cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
            masses_int_prob_sup_full.append(masses_int_prob_sup)
            masses_prob_sup_full.append(masses_prob_sup)
            masses_prob_inf_full.append(masses_prob_inf)
    
    
    cross_sec_int_prob_sup_aux.append( np.mean(np.asarray(cross_sec_int_prob_sup_full), axis = 0) )
    cross_sec_int_prob_sup_aux_sd.append( np.std(np.asarray(cross_sec_int_prob_sup_full), axis = 0) )
    masses_int_prob_sup_aux.append( np.mean(np.asarray(masses_int_prob_sup_full), axis = 0) )
    masses_int_prob_sup_aux_sd.append( np.std(np.asarray(masses_int_prob_sup_full), axis = 0) )
    masses_prob_sup_aux.append( np.mean(np.asarray(masses_prob_sup_full), axis = 0) )
    masses_prob_sup_aux_sd.append( np.std(np.asarray(masses_prob_sup_full), axis = 0) )
    masses_prob_inf_aux.append( np.mean(np.asarray(masses_prob_inf_full), axis = 0) )
    masses_prob_inf_aux_sd.append( np.std(np.asarray(masses_prob_inf_full), axis = 0) )

# +
fig, ax = plt.subplots(2,2)

for i in range(len(thetas)):
    sbn.kdeplot(cross_sec_int_prob_sup_aux[i], label = '$\\theta = $' + thetas[i], ax = ax[0,0])    
    sbn.kdeplot(masses_int_prob_sup_aux[i], label = '$\\theta = $' + thetas[i], ax = ax[0,1])
    sbn.kdeplot(masses_prob_sup_aux[i], label = '$\\theta = $' + thetas[i], ax = ax[1,0])
    sbn.kdeplot(masses_prob_inf_aux[i], label = '$\\theta = $' + thetas[i], ax = ax[1,1])
ax[0,0].legend()
ax[0,0].set_xlabel('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')
ax[0,0].set_title('s1s2')

ax[0,1].legend()
ax[0,1].set_xlabel('$\int_{m_{min}}^{m_{max}} P(m_{DM}|x)$')
ax[0,1].set_title('s1s2')

ax[1,0].legend()
ax[1,0].set_xlabel('$\int_{m_{min}}^{\inf} P(m_{DM}|x)$')

ax[1,1].legend()
ax[1,1].set_xlabel('$\int_{0}^{m_{max}} P(m_{DM}|x)$')

#plt.savefig('../graph/O1_int_prob_distribution_s1s2_noBKG.pdf')



# +
CR_int_prob_sup_comb = []
M_int_prob_sup_comb = []
M_prob_sup_comb = []
M_prob_inf_comb = []

sigma = 1.1
for i in range(len(thetas)):
    CR_int_prob_sup_comb.append( gaussian_filter(cross_sec_int_prob_sup_aux[i], sigma) )
    M_int_prob_sup_comb.append( gaussian_filter(masses_int_prob_sup_aux[i], 1.5) )
    M_prob_sup_comb.append( gaussian_filter(masses_prob_sup_aux[i], sigma) )
    M_prob_inf_comb.append( gaussian_filter(masses_prob_inf_aux[i], sigma) )
# -


np.max(rate_slices)

# +
levels = [0.67, 0.76, 0.84, 0.9, 1]

color_rate  = "#d55e00"
color_drate = "#0072b2"
color_s1s2  = "#009e73"
color_comb = "#009e73"

fig, ax = plt.subplots(1,3, sharex = True, sharey = True, figsize = (12,5))
fig.subplots_adjust(hspace = 0, wspace = 0)

for i, theta in enumerate([3,4,0]):
    pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice(['../data/andresData/O1-slices-5vecescadatheta/theta-' + thetas[theta] + '/SI-slices01-' + thetas[theta] + '/'])
    ax[i].contour(m_vals, cross_vals, rate_slices.reshape(30,30).T, levels = [10,100,1000,10000], colors = ['purple','purple','purple'], linestyles = ['solid','--',':','-.'])
    
    ax[i].contour(m_vals, cross_vals, CR_int_prob_sup_comb[theta].reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_comb)
    ax[i].contour(m_vals, cross_vals, M_int_prob_sup_comb[theta].reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = ':', colors = color_comb)
    ax[i].contour(m_vals, cross_vals, M_prob_sup_comb[theta].reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_comb)

    ax[i].contour(m_vals, cross_vals, CR_int_prob_sup_rate[theta].reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_rate)
    #ax[i].contour(m_vals, cross_vals, M_int_prob_sup_rate[theta].reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_rate)
    #ax[i].contour(m_vals, cross_vals, M_prob_sup_rate[theta].reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)

    ax[i].contour(m_vals, cross_vals, CR_int_prob_sup_drate[theta].reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_drate)
    #ax[i].contour(m_vals, cross_vals, M_int_prob_sup_drate[theta].reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_drate)
    #ax[i].contour(m_vals, cross_vals, M_prob_sup_drate[theta].reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
    
    #ax[i].contour(m_vals, cross_vals, CR_int_prob_sup_s1s2[theta].reshape(30,30).T, levels = [0.9], linewidths = 2, colors = 'magenta')
    #ax[i].contour(m_vals, cross_vals, M_prob_sup_s1s2[theta].reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = 'magenta')

    #ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_rate)
    #ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_drate)

# #%ax[0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]', linestyle = ':')
ax[0].fill_between(neutrino_fog[:,0], neutrino_fog[:,1], -50, color = "none", edgecolor='black', label = '$\\nu$ fog', alpha = 0.8, hatch = '///')
ax[0].plot(masses, s1s2_90_CL_pi2[2,:], color = 'black', linestyle = ':', label = 'Bin. Lik. [90%]')
ax[0].fill_between(masses, s1s2_current_pi2[2,:], 1e-43, color = 'black', alpha = 0.2, label = 'Excluded', zorder = 1)

ax[0].set_yscale('log')
ax[0].set_xscale('log')
#ax[0].grid(which='both')
ax[0].text(3e2, 2e-44, '$\\theta = \pi/2$')
ax[0].legend(loc = 'lower left')

ax[1].plot(masses, s1s2_90_CL_pi4[2,:], color = 'black', linestyle = ':')
ax[1].fill_between(masses, s1s2_current_pi4[2,:], 1e-43, color = 'black', alpha = 0.2)

#ax[1].grid(which='both')
ax[1].text(3e2, 2e-44, '$\\theta = \pi/4$')

ax[2].plot(masses, s1s2_90_CL_0[2,:], color = 'black', linestyle = ':')
ax[2].fill_between(masses, s1s2_current_0[2,:], 1e-43, color = 'black', alpha = 0.2, label = 'Excluded')
ax[2].legend(loc = 'lower right')

#ax[2].grid(which='both')
#ax[2].text(3e2, 2e-44, '$\\theta = 0$')

ax[0].set_ylabel('$\sigma \ [cm^{2}]$')
ax[0].set_xlabel('m [GeV]')
ax[1].set_xlabel('m [GeV]')
ax[2].set_xlabel('m [GeV]')

ax[0].set_ylim(1e-49, 5e-44)
ax[0].set_xlim(6, 9.8e2)

fig.subplots_adjust(right=0.8)

custom_lines = []
labels = ['Rate', 'Rate + Dif. Rate', 'Rate + Dif. Rate + S1S2']
markers = ['solid','solid', 'solid']
colors = [color_rate, color_drate, color_comb]
for i in range(3):
    custom_lines.append( Line2D([0],[0], linestyle = markers[i], color = colors[i], 
            label = labels[i]) )
    
ax[1].legend(handles = custom_lines, loc = 'lower left')

custom_lines = []
#labels = ['$\\sigma$', '$M_{DM}$']
labels = ['$\\mathcal{P}_{\\sigma}$', '$\\mathcal{P}_{M_{DM}}$']
markers = ['solid','--']
for i in range(2):
    custom_lines.append( Line2D([0],[0], linestyle = markers[i], color = 'black', 
            label = labels[i]) )
    
ax[2].legend(handles = custom_lines, loc = 'lower left')

plt.savefig('../graph/O1_contours_all_int_prob_sup_COMB_noBKG.pdf')
# -



