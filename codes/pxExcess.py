import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import time
from scipy import stats
import seaborn as sbn
import pandas as pd
import h5py
import seaborn as sns
pallete = np.flip(sns.color_palette("tab20c", 8), axis = 0)
cross_section_th = -50

# It is usefull to print the versions of the package that we are using
print('numpy version:', np.__version__)
print('matplotlib version:', mpl.__version__)


def read_slice(datFolder):
    nobs_slices = 0
    for i, folder in enumerate(datFolder):
        print(i)
        if i == 0:
            pars_slices      = np.loadtxt(folder + 'pars.txt') # pars[:,0] = mass ; pars[:,1] = cross-section ; pars[:,2] = theta
            rate_raw_slices  = np.loadtxt(folder + 'rate.txt') # rate[:,0] = total expected events ; rate[:,1] = expected signal ; rate[:,2] = # events pseudo-experiment ; rate[:,3] = # signal events pseudo-experiment 
            diff_rate_slices = np.loadtxt(folder + 'diff_rate.txt')
            
            s1s2_WIMP_slices     = np.loadtxt(folder + 's1s2_WIMP.txt')
            s1s2_er_slices       = np.loadtxt(folder + 's1s2_er.txt')
            s1s2_ac_slices       = np.loadtxt(folder + 's1s2_ac.txt')
            s1s2_cevns_SM_slices = np.loadtxt(folder + 's1s2_CEVNS-SM.txt')
            s1s2_radio_slices    = np.loadtxt(folder + 's1s2_radiogenics.txt')
            s1s2_wall_slices     = np.loadtxt(folder + 's1s2_wall.txt')
        else:
            pars_slices      = np.vstack((pars_slices, np.loadtxt(folder + 'pars.txt'))) # pars[:,0] = mass ; pars[:,1] = cross-section ; pars[:,2] = theta
            rate_raw_slices  = np.vstack((rate_raw_slices, np.loadtxt(folder + 'rate.txt'))) # rate[:,0] = total expected events ; rate[:,1] = expected signal ; rate[:,2] = # events pseudo-experiment ; rate[:,3] = # signal events pseudo-experiment 
            diff_rate_slices = np.vstack((diff_rate_slices, np.loadtxt(folder + 'diff_rate.txt')))
            
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
    
    # Let's work with the log of the mass and cross-section
    
    pars_slices[:,0] = np.log10(pars_slices[:,0])
    pars_slices[:,1] = np.log10(pars_slices[:,1])
    
    # Let's transform the diff_rate to counts per energy bin
    
    diff_rate_slices = np.round(diff_rate_slices * 362440)
    return pars_slices, rate_slices, diff_rate_slices, s1s2_slices

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
        diff_rate = np.loadtxt(folder + 'diff_rate.txt')
        
        s1s2_WIMP     = np.loadtxt(folder + 's1s2_WIMP.txt')
        s1s2_er       = np.loadtxt(folder + 's1s2_er.txt')
        s1s2_ac       = np.loadtxt(folder + 's1s2_ac.txt')
        s1s2_cevns_SM = np.loadtxt(folder + 's1s2_CEVNS-SM.txt')
        s1s2_radio    = np.loadtxt(folder + 's1s2_radiogenics.txt')
        s1s2_wall     = np.loadtxt(folder + 's1s2_wall.txt')
    else:
        pars      = np.vstack((pars, np.loadtxt(folder + 'pars.txt'))) # pars[:,0] = mass ; pars[:,1] = cross-section ; pars[:,2] = theta
        rate_raw  = np.vstack((rate_raw, np.loadtxt(folder + 'rate.txt'))) # rate[:,0] = total expected events ; rate[:,1] = expected signal ; rate[:,2] = # events pseudo-experiment ; rate[:,3] = # signal events pseudo-experiment 
        diff_rate = np.vstack((diff_rate, np.loadtxt(folder + 'diff_rate.txt')))
        
        s1s2_WIMP     = np.vstack((s1s2_WIMP, np.loadtxt(folder + 's1s2_WIMP.txt')))
        s1s2_er       = np.vstack((s1s2_er, np.loadtxt(folder + 's1s2_er.txt')))
        s1s2_ac       = np.vstack((s1s2_ac, np.loadtxt(folder + 's1s2_ac.txt')))
        s1s2_cevns_SM = np.vstack((s1s2_cevns_SM, np.loadtxt(folder + 's1s2_CEVNS-SM.txt')))
        s1s2_radio    = np.vstack((s1s2_radio, np.loadtxt(folder + 's1s2_radiogenics.txt')))
        s1s2_wall     = np.vstack((s1s2_wall, np.loadtxt(folder + 's1s2_wall.txt')))
        
    
nobs = len(pars) # Total number of observations
print('We have ' + str(nobs) + ' observations...')

s1s2 = s1s2_WIMP + s1s2_er + s1s2_ac + s1s2_cevns_SM + s1s2_radio + s1s2_wall
s1s2_bck = s1s2_er + s1s2_ac + s1s2_radio + s1s2_wall + s1s2_cevns_SM
rate = np.sum(s1s2, axis = 1) # Just to have the same as on the other notebooks. This already includes the backgrounds
s1s2 = s1s2.reshape(nobs, 97, 97)
s1s2_bck = s1s2_bck.reshape(nobs, 97, 97)

# Let's work with the log of the mass and cross-section

pars[:,0] = np.log10(pars[:,0])
pars[:,1] = np.log10(pars[:,1])

# Let's transform the diff_rate to counts per energy bin

diff_rate = np.round(diff_rate * 362440)

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

s1s2_bck_mean = np.mean(s1s2_bck[train_ind, :,:], axis = 0)
s1s2_bck_std = np.std(s1s2_bck[train_ind, :,:], axis = 0)

# +
fig, ax = plt.subplots(2,2)

ax[0,0].imshow(s1s2_bck_mean.T, origin = 'lower')
ax[0,1].imshow(s1s2_bck_mean.T + s1s2_bck_std.T, origin = 'lower')
ax[1,0].imshow(s1s2_bck_mean.T + 2 * s1s2_bck_std.T, origin = 'lower')
ax[1,1].imshow(s1s2_bck_mean.T + 5 * s1s2_bck_std.T, origin = 'lower')
# -

# ## Xenon nt 

xenon_nt_5s   = np.loadtxt('../data/xenon_nt_5sigma.csv', skiprows = 1, delimiter = ',')
xenon_nt_3s   = np.loadtxt('../data/xenon_nt_3sigma.csv', skiprows = 1, delimiter = ',')
xenon_nt_90cl = np.loadtxt('../data/xenon_nt_90cl.csv', skiprows = 1, delimiter = ',')

# ## Analysis

# +
t1_bck = np.ones(len(val_ind)) * -99
t3_bck = np.ones(len(val_ind)) * -99
t5_bck = np.ones(len(val_ind)) * -99

t1 = np.ones(len(val_ind)) * -99
t3 = np.ones(len(val_ind)) * -99
t5 = np.ones(len(val_ind)) * -99

for i in range(len(val_ind)):
    img = s1s2_bck[val_ind[i],:,:]
    
    rest  = img - (s1s2_bck_mean + s1s2_bck_std)
    ind0  = np.where(rest > 0)[0]
    ind1  = np.where(rest > 0)[1]
    t1_bck[i] = np.log10(np.sum(rest[ind0, ind1]))
    
    rest  = img - (s1s2_bck_mean + 3 * s1s2_bck_std)
    ind0  = np.where(rest > 0)[0]
    ind1  = np.where(rest > 0)[1]
    t3_bck[i] = np.log10(np.sum(rest[ind0, ind1]))
    
    rest  = img - (s1s2_bck_mean + 5 * s1s2_bck_std)
    ind0  = np.where(rest > 0)[0]
    ind1  = np.where(rest > 0)[1]
    t5_bck[i] = np.log10(np.sum(rest[ind0, ind1]))
    
    rest  = s1s2_valset[i] - (s1s2_bck_mean + s1s2_bck_std)
    ind0  = np.where(rest > 0)[0]
    ind1  = np.where(rest > 0)[1]
    t1[i] = np.log10(np.sum(rest[ind0, ind1]))
    
    rest  = s1s2_valset[i] - (s1s2_bck_mean + 3 * s1s2_bck_std)
    ind0  = np.where(rest > 0)[0]
    ind1  = np.where(rest > 0)[1]
    t3[i] = np.log10(np.sum(rest[ind0, ind1]))
    
    rest  = s1s2_valset[i] - (s1s2_bck_mean + 5 * s1s2_bck_std)
    ind0  = np.where(rest > 0)[0]
    ind1  = np.where(rest > 0)[1]
    t5[i] = np.log10(np.sum(rest[ind0, ind1]))

# +
fig, ax = plt.subplots(1,3, figsize = (10,5))

sns.kdeplot(t1_bck, ax = ax[0], label = 'Bck')
sns.kdeplot(t3_bck, ax = ax[1])
sns.kdeplot(t5_bck, ax = ax[2])

sns.kdeplot(t1, ax = ax[0], color = 'red', label = 'WIMP')
sns.kdeplot(t3, ax = ax[1], color = 'red')
sns.kdeplot(t5, ax = ax[2], color = 'red')

ax[0].legend()
ax[0].set_xlabel('t1')
ax[1].set_xlabel('t3')
ax[2].set_xlabel('t5')
# -

nval = len(t1_bck)
pval1 = np.ones(len(val_ind)) * -99
pval3 = np.ones(len(val_ind)) * -99
pval5 = np.ones(len(val_ind)) * -99
for i in range(len(val_ind)):
    pval1[i] = len(np.where(t1_bck > t1[i])[0]) / nval
    pval3[i] = len(np.where(t3_bck > t3[i])[0]) / nval
    pval5[i] = len(np.where(t5_bck > t5[i])[0]) / nval

# +
fig, ax = plt.subplots(1,3, figsize = (10,5))

ax[0].hist(pval1)
ax[1].hist(pval3)
ax[2].hist(pval5)
# -

m_vals = np.logspace(np.min(pars_slices[:,0]), np.max(pars_slices[:,0]),30)
cross_vals = np.logspace(np.min(pars_slices[:,1]), np.max(pars_slices[:,1]),30)
pval_th = 0.1


folder = ['../data/andresData/SI-slices01-variostheta/SI-slices01-pluspidiv2/',
          '../data/andresData/SI-slices01-variostheta/SI-slices01-pluspidiv4/',
          '../data/andresData/SI-slices01-variostheta/SI-slices01-minuspidiv2/',
          '../data/andresData/SI-slices01-variostheta/SI-slices01-theta0/'
         ]
res_1sigma = []
res_3sigma = []
res_5sigma = []
for fol in folder:
    print(fol)
    pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice([fol])
    
    t1 = np.ones(len(pars_slices)) * -99
    t3 = np.ones(len(pars_slices)) * -99
    t5 = np.ones(len(pars_slices)) * -99
    pval1 = np.ones(len(pars_slices)) * -99
    pval3 = np.ones(len(pars_slices)) * -99
    pval5 = np.ones(len(pars_slices)) * -99
    
    for i in range(len(pars_slices)):
        rest  = s1s2_slices[i] - (s1s2_bck_mean + s1s2_bck_std)
        ind0  = np.where(rest > 0)[0]
        ind1  = np.where(rest > 0)[1]
        t1[i] = np.log10(np.sum(rest[ind0, ind1]))
        pval1[i] = len(np.where(t1_bck > t1[i])[0]) / nval
        
        rest  = s1s2_slices[i] - (s1s2_bck_mean + 3 * s1s2_bck_std)
        ind0  = np.where(rest > 0)[0]
        ind1  = np.where(rest > 0)[1]
        t3[i] = np.log10(np.sum(rest[ind0, ind1]))
        pval3[i] = len(np.where(t3_bck > t3[i])[0]) / nval
        
        rest  = s1s2_slices[i] - (s1s2_bck_mean + 5 * s1s2_bck_std)
        ind0  = np.where(rest > 0)[0]
        ind1  = np.where(rest > 0)[1]
        t5[i] = np.log10(np.sum(rest[ind0, ind1]))    
        pval5[i] = len(np.where(t5_bck > t5[i])[0]) / nval
        
    
    res_1sigma_aux = np.ones(len(pars_slices)) * -99
    res_3sigma_aux = np.ones(len(pars_slices)) * -99
    res_5sigma_aux = np.ones(len(pars_slices)) * -99
    
    res_1sigma_aux[np.where(pval1 < pval_th)[0]] = 1    
    res_3sigma_aux[np.where(pval3 < pval_th)[0]] = 1    
    res_5sigma_aux[np.where(pval5 < pval_th)[0]] = 1

    res_1sigma.append(res_1sigma_aux)
    res_3sigma.append(res_3sigma_aux)
    res_5sigma.append(res_5sigma_aux)

# +
fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))

ax[0,0].contour(m_vals, cross_vals, res_1sigma[0].reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[0,0].contour(m_vals, cross_vals, res_3sigma[0].reshape(30,30).T, levels=[0], linestyles = ':')
ax[0,0].contourf(m_vals, cross_vals, res_5sigma[0].reshape(30,30).T, levels=[-100, 0, 1], alpha = 0.6, zorder = 1, colors = ['white','blue'])
ax[0,0].contour(m_vals, cross_vals, res_5sigma[0].reshape(30,30).T, levels=[0])

ax[0,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
ax[0,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
ax[0,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-44, '$\\theta = \pi/2$')
#ax[0,0].legend(loc = 'lower right')

ax[0,1].contour(m_vals, cross_vals, res_1sigma[1].reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[0,1].contour(m_vals, cross_vals, res_3sigma[1].reshape(30,30).T, levels=[0], linestyles = ':')
ax[0,1].contourf(m_vals, cross_vals, res_5sigma[1].reshape(30,30).T, levels=[-100, 0, 1], alpha = 0.6, zorder = 1, colors = ['white','blue'])
ax[0,1].contour(m_vals, cross_vals, res_5sigma[1].reshape(30,30).T, levels=[0])

ax[0,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--', label = 'XENON nT [$3\sigma$]')
ax[0,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':', label = 'XENON nT [$5\sigma$]')
ax[0,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-44, '$\\theta = \pi/4$')
ax[0,1].legend(loc = 'lower right')

ax[1,0].contour(m_vals, cross_vals, res_1sigma[2].reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[1,0].contour(m_vals, cross_vals, res_3sigma[2].reshape(30,30).T, levels=[0], linestyles = ':')
ax[1,0].contourf(m_vals, cross_vals, res_5sigma[2].reshape(30,30).T, levels=[-100, 0, 1], alpha = 0.6, zorder = 1, colors = ['white','blue'])
ax[1,0].contour(m_vals, cross_vals, res_5sigma[2].reshape(30,30).T, levels=[0])

ax[1,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
ax[1,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
ax[1,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-44, '$\\theta = -\pi/2$')

ax[1,1].contour(m_vals, cross_vals, res_1sigma[3].reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[1,1].contour(m_vals, cross_vals, res_3sigma[3].reshape(30,30).T, levels=[0], linestyles = ':')
ax[1,1].contourf(m_vals, cross_vals, res_5sigma[3].reshape(30,30).T, levels=[-100, 0, 1], alpha = 0.6, zorder = 1, colors = ['white','blue'])
ax[1,1].contour(m_vals, cross_vals, res_5sigma[3].reshape(30,30).T, levels=[0])

ax[1,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
ax[1,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
ax[1,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-44, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma$ []')
ax[1,0].set_ylabel('$\sigma$ []')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')
# -


