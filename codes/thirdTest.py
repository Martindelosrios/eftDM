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
import torch
import torchist
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import seaborn as sns
pallete = np.flip(sns.color_palette("tab20c", 8), axis = 0)

from torchsummary import summary

# It is usefull to print the versions of the package that we are using
print('swyft version:', swyft.__version__)
print('numpy version:', np.__version__)
print('matplotlib version:', mpl.__version__)
print('torch version:', torch.__version__)

# Check if gpu is available
if torch.cuda.is_available():
    device = 'gpu'
    print('Using GPU')
else:
    device = 'cpu'
    print('Using CPU')

# # Let's load the data

# !ls ../data/andresData/SI-run0and1/

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
# -

s1s2 = s1s2_WIMP + s1s2_er + s1s2_ac + s1s2_cevns_SM + s1s2_radio + s1s2_wall
rate = np.sum(s1s2, axis = 1) # Just to have the same as on the other notebooks. This already includes the backgrounds
s1s2 = s1s2.reshape(nobs, 97, 97)

# This should be always zero
i = np.random.randint(nobs)
rate_raw[i,2] - rate[i]

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

# +
# Let's work with the log of the mass and cross-section

pars[:,0] = np.log10(pars[:,0])
pars[:,1] = np.log10(pars[:,1])

# +
# Let's transform the diff_rate to counts per energy bin

diff_rate = np.round(diff_rate * 362440)
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

# ## Let's make some exploratory plots

sbn.pairplot(pd.DataFrame(np.hstack((pars,np.log10(rate).reshape(nobs,1))), columns = ['$m_{\chi}$','$\sigma$', '$\\theta$', '#']))

# +
fig, ax = plt.subplots(1,3, figsize = (10,5))

ax[0].hist(pars[:,0], histtype = 'step')
ax[0].set_xlabel('$\log_{10}$(m [GeV?] )')
#ax[0].set_xscale('log')

ax[1].hist(pars[:,1], histtype = 'step')
ax[1].set_xlabel('$\log_{10}{\sigma}$ [?]')
#ax[1].set_xscale('log')

ax[2].hist(pars[:,2], histtype = 'step')
ax[2].set_xlabel('$\\theta$')

# +
i = np.random.randint(len(pars))
print(i)
fig, ax = plt.subplots(1,2, figsize = (10,5))

ax[0].plot(diff_rate[i,:], c = 'black')
ax[0].set_xlabel('$E_{r}$ [keV]' )
ax[0].set_ylabel('$dR/E_{r}$' )
ax[0].text(0.5, 0.8,  '$\log_{10} $' + 'm = {:.2f} [?]'.format(pars[i,0]), transform = ax[0].transAxes)
ax[0].text(0.5, 0.7,  '$\log_{10}\sigma$' + ' = {:.2f} [?]'.format(pars[i,1]), transform = ax[0].transAxes)
ax[0].text(0.5, 0.6, '$\\theta$ = {:.2f}'.format(pars[i,2]), transform = ax[0].transAxes)
ax[0].text(0.5, 0.5, 'Total Rate = {:.2e}'.format(rate[i]), transform = ax[0].transAxes)
#ax[0].set_yscale('log')

ax[1].imshow(s1s2[i].T, origin = 'lower')
ax[1].set_xlabel('s1')
ax[1].set_ylabel('s2')
# -

# # Let's play with SWYFT

# ## Using only the total rate with gaussian background ($\mu = 7 ; \sigma = 1$)

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
# Let's configure, instantiate and traint the network
torch.manual_seed(28890)
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta = 0., patience=30, verbose=False, mode='min')
trainer_rate = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 200, precision = 64, callbacks=[early_stopping_callback])
network_rate = Network_rate()
trainer_rate.fit(network_rate, dm_rate)

# ---------------------------------------------- 
# It converges to val_loss = -1.18 at epoch 100
# ---------------------------------------------- 
# -

# ### Let's make some inference

# +
# Let's normalize testset between 0 and 1

pars_norm = (pars_testset - pars_min) / (pars_max - pars_min)

x_rate = np.log10(rate_testset)
x_norm_rate = (x_rate - x_min_rate) / (x_max_rate - x_min_rate)
x_norm_rate = x_norm_rate.reshape(len(x_norm_rate), 1)

# +
# First let's create some observation from some "true" theta parameters
i = np.random.randint(ntest)
print(i)
pars_true = pars_norm[i,:]
x_obs     = x_norm_rate[i,:]

print('"Normalized Observed" x value : {}'.format(x_obs))
real_val = 10**(x_obs * (x_max_rate - x_min_rate) + x_min_rate)
print('"Observed" x value : {}'.format(real_val))

if real_val < 7: 
    flag = 'exc'
else:
    flag = 'disc'
print(flag)


# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
pars_prior    = np.random.uniform(low = 0, high = 1, size = (1_000_000, 3))
prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions_rate = trainer_rate.infer(network_rate, obs, prior_samples)
# -

# Let's plot the results
swyft.corner(predictions_rate, ('pars_norm[0]', 'pars_norm[1]', 'pars_norm[2]'), bins = 200, smooth = 3)
if flag == 'exc':
    plt.savefig('../graph/cornerplot_rate_exc.pdf')
else:
    plt.savefig('../graph/cornerplot_rate.pdf')

# +
bins = 50
logratios_rate = predictions_rate[0].logratios[:,1]
v              = predictions_rate[0].params[:,1,0]
low, upp = v.min(), v.max()
weights  = torch.exp(logratios_rate) / torch.exp(logratios_rate).mean(axis = 0)
h1       = torchist.histogramdd(predictions_rate[0].params[:,1,:], bins, weights = weights, low=low, upp=upp)
h1      /= len(predictions_rate[0].params[:,1,:]) * (upp - low) / bins
h1       = np.array(h1)

edges = torch.linspace(v.min(), v.max(), bins + 1)
x     = np.array((edges[1:] + edges[:-1]) / 2) * (pars_max[1] - pars_min[1]) + pars_min[1]
#x     = 10**(x)

# +
vals = sorted(swyft.plot.plot2.get_HDI_thresholds(h1, cred_level=[0.68268, 0.95450, 0.99730]))

low_1sigma = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
up_1sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])

low_2sigma = np.min(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
up_2sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[1]))[0]])

low_3sigma = np.min(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
up_3sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[0]))[0]])

if low_1sigma > -47.8: print('Distinguish at 1 $\sigma$')
if low_2sigma > -47.8: print('Distinguish at 2 $\sigma$')
if low_3sigma > -47.8: print('Distinguish at 3 $\sigma$')


# +
plt.plot(x, h1, c = 'blue')

#y0 = 0 #-1.0 * x.max()
#y1 = 5.0# * x.max()
#plt.fill_between(x, y0, y1, where = h1 > vals[0], color='red', alpha=0.1)
#plt.fill_between(x, y0, y1, where = h1 > vals[1], color='red', alpha=0.2)
#plt.fill_between(x, y0, y1, where = h1 > vals[2], color='red', alpha=0.3)

if low_1sigma > -47.8: plt.axvline(low_1sigma, c = 'green')
if up_1sigma > -47.8: plt.axvline(up_1sigma, c = 'green')

if low_2sigma > -47.8: plt.axvline(low_2sigma, c = 'green', linestyle = '--')
if up_2sigma > -47.8: plt.axvline(up_2sigma, c = 'green', linestyle = '--')

if low_3sigma > -47.8: plt.axvline(low_3sigma, c = 'green', linestyle = ':')
if up_3sigma > -47.8: plt.axvline(up_3sigma, c = 'green', linestyle = ':')
#plt.ylim(0,4.5)
#plt.xscale('log')

# +
swyft.plot_1d(predictions_rate, "pars_norm[1]", bins = 50, smooth = 1)
plt.plot(x, h1, c = 'blue')
#plt.plot(zm, v, c = 'magenta', linestyle = '--')

#swyft.plot.plot2.contour1d(x, h1, vals, linestyle = ':', alpha = 0.2, color = 'magenta')
# -

parameters_rate = np.asarray(predictions_rate[0].params[:,:,0])
parameters_rate = parameters_rate * (pars_max - pars_min) + pars_min
parameters_rate.shape

# +
fig,ax = plt.subplots(1,3, sharey=True)

ax[0].plot(parameters_rate[:,0], predictions_rate[0].logratios[:,0], 'o', rasterized = True)
ax[0].set_xlabel(r'$m$')
ax[0].set_ylabel(r'log ratio')
ax[0].axvline(x = pars_testset[i,0])

ax[1].plot(parameters_rate[:,1], predictions_rate[0].logratios[:,1], 'o', rasterized = True)
ax[1].set_xlabel(r'$\sigma$')
ax[1].axvline(x = pars_testset[i,1])

ax[2].plot(parameters_rate[:,2], predictions_rate[0].logratios[:,2], 'o', rasterized = True)
ax[2].set_xlabel(r'$g$')
ax[2].axvline(x = pars_testset[i,2])

if flag == 'exc':
    plt.savefig('../graph/loglikratio_rate_exc.pdf')
else:
    plt.savefig('../graph/loglikratio_rate.pdf')
# -

10**(pars_true * (pars_max - pars_min) + pars_min)

results_pars_rate = np.asarray(predictions_rate[1].params)
results_rate      = np.asarray(predictions_rate[1].logratios)

# +
fig, ax = plt.subplots(1,3, gridspec_kw = {'hspace':0.7, 'wspace':0.4}, figsize = (12,4))

#  -------------------------------- MAX  ----------------------------------------

# M vs Sigma

m_results     = 10**(results_pars_rate[:,0,0] * (pars_max[0] - pars_min[0]) + pars_min[0])
m_true        = 10**(pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0])
sigma_results = 10**(results_pars_rate[:,0,1] * (pars_max[1] - pars_min[1]) + pars_min[1])
sigma_true    = 10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1])

val, xaux, yaux,_ = stats.binned_statistic_2d(m_results, sigma_results, results_rate[:,0], 'max', bins = [np.logspace(0.81, 3, 15), np.logspace(-48.2, -41, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im20 = ax[0].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im20, ax = ax[0])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,0]), 10**(pars[:,1]), np.log10(rate + 7), 'min', bins = [np.logspace(0.81, 3, 10), np.logspace(-48.2, -41, 10)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0].contour(x_centers, y_centers, val.T, levels = [0, 1, 2, 3], cmap = 'inferno')
ax[0].axhline(y = sigma_true, c = 'red')
ax[0].axvline(x = m_true, c = 'red')
ax[0].set_xlabel('m')
ax[0].set_ylabel('$\sigma$')
ax[0].set_xscale('log')
ax[0].set_yscale('log')

# M vs theta

m_results     = 10**(results_pars_rate[:,1,0] * (pars_max[0] - pars_min[0]) + pars_min[0])
m_true        = 10**(pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0])
theta_results = results_pars_rate[:,1,1] * (pars_max[2] - pars_min[2]) + pars_min[2]
theta_true    = pars_true[2] * (pars_max[2] - pars_min[2]) + pars_min[2]

val, xaux, yaux,_ = stats.binned_statistic_2d(m_results, theta_results, results_rate[:,1], 'max', bins = [np.logspace(0.81, 3, 15), np.linspace(-1.6, 1.6, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im21 = ax[1].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im21, ax = ax[1])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,0]), pars[:,2], np.log10(rate + 7), 'min', bins = [np.logspace(0.81, 3, 10), np.linspace(-1.6, 1.6, 10)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1].contour(x_centers, y_centers, val.T, levels = [0, 1, 2, 3], cmap = 'inferno')
ax[1].axhline(y = theta_true, c = 'red')
ax[1].axvline(x = m_true, c = 'red')
ax[1].set_xlabel('m')
ax[1].set_ylabel('$\\theta$')
ax[1].set_xscale('log')

# Sigma vs theta

sigma_results = 10**(results_pars_rate[:,2,0] * (pars_max[1] - pars_min[1]) + pars_min[1])
sigma_true    = 10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1])
theta_results = results_pars_rate[:,2,1] * (pars_max[2] - pars_min[2]) + pars_min[2]
theta_true    = pars_true[2] * (pars_max[2] - pars_min[2]) + pars_min[2]

val, xaux, yaux,_ = stats.binned_statistic_2d(sigma_results, theta_results, results_rate[:,2], 'max', bins = [np.logspace(-48.2, -41, 15), np.linspace(-1.6, 1.6, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im22 = ax[2].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im22, ax = ax[2])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,1]), pars[:,2], np.log10(rate + 7), 'min', bins = [np.logspace(-48.2, -41, 10), np.linspace(-1.6, 1.6, 10)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2].contour(x_centers, y_centers, val.T, levels = [0, 1, 2, 3], cmap = 'inferno')
ax[2].axhline(y = theta_true, c = 'red')
ax[2].axvline(x = sigma_true, c = 'red')
ax[2].set_xlabel('$\sigma$')
ax[2].set_ylabel('$\\theta$')
ax[2].set_xscale('log')

if flag == 'exc':
    plt.savefig('../graph/pars_rate_exc.pdf')
else:
    plt.savefig('../graph/pars_rate.pdf')

# +
# Let's normalize testset between 0 and 1

pars_norm = (pars_valset - pars_min) / (pars_max - pars_min)

x_rate = np.log10(rate_valset)
x_norm_rate = (x_rate - x_min_rate) / (x_max_rate - x_min_rate)
x_norm_rate = x_norm_rate.reshape(len(x_norm_rate), 1)

res_1sigma = np.ones(len(x_rate)) * -99
res_2sigma = np.ones(len(x_rate)) * -99
res_3sigma = np.ones(len(x_rate)) * -99

for itest in tqdm(range(len(x_rate))):
    x_obs = x_norm_rate[itest, :]
    
    # We have to put this "observation" into a swyft.Sample object
    obs = swyft.Sample(x = x_obs)
    
    # Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
    pars_prior    = np.random.uniform(low = 0, high = 1, size = (100_000, 3))
    prior_samples = swyft.Samples(z = pars_prior)
    
    # Finally we make the inference
    predictions_rate = trainer_rate.infer(network_rate, obs, prior_samples)

    bins = 50
    logratios_rate = predictions_rate[0].logratios[:,1]
    v              = predictions_rate[0].params[:,1,0]
    low, upp = v.min(), v.max()
    weights  = torch.exp(logratios_rate) / torch.exp(logratios_rate).mean(axis = 0)
    h1       = torchist.histogramdd(predictions_rate[0].params[:,1,:], bins, weights = weights, low=low, upp=upp)
    h1      /= len(predictions_rate[0].params[:,1,:]) * (upp - low) / bins
    h1       = np.array(h1)
    
    edges = torch.linspace(v.min(), v.max(), bins + 1)
    x     = np.array((edges[1:] + edges[:-1]) / 2) * (pars_max[1] - pars_min[1]) + pars_min[1]

    vals = sorted(swyft.plot.plot2.get_HDI_thresholds(h1, cred_level=[0.68268, 0.95450, 0.99730]))
    
    low_1sigma = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
    up_1sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
    
    low_2sigma = np.min(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
    up_2sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
    
    low_3sigma = np.min(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
    up_3sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
    
    if low_1sigma > -47.8: res_1sigma[itest] = 1
    if low_2sigma > -47.8: res_2sigma[itest] = 1
    if low_3sigma > -47.8: res_3sigma[itest] = 1

# +
val, x, y,_ = stats.binned_statistic_2d(pars_valset[:,0], pars_valset[:,1], res_1sigma, 'max', bins = 10)
        
xbin = x[1] - x[0]
x_centers = x[:-1] + xbin

ybin = y[1] - y[0]
y_centers = y[:-1] + ybin

cs = plt.contourf(x_centers, y_centers, val.T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
plt.contour(x_centers, y_centers, val.T, levels=[0], linewidths = 2, zorder = 4)
plt.scatter(pars_valset[:-2,0], pars_valset[:-2,1])
# -

# ## Only using the total diff_rate (without background)

x_drate = np.log10(diff_rate_trainset + 1) # Observable. Input data. 

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

for i in range(50):
    ax[0,0].plot(x_norm_drate[i])
ax[0,0].set_xlabel('$E_{r}$')

ax[1,0].hist(pars_norm[:,0])
ax[1,0].set_xlabel('$M_{DM}$')

ax[0,1].hist(pars_norm[:,1])
ax[0,1].set_xlabel('$\sigma$')

ax[1,1].hist(pars_norm[:,2])
ax[1,1].set_xlabel('$\\theta$')
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
    def __init__(self):
        super().__init__()
        marginals = ((0, 1), (0, 2), (1, 2))
        self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = 56, num_params = 3, varnames = 'pars_norm')
        self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features = 56, marginals = marginals, varnames = 'pars_norm')

    def forward(self, A, B):
        logratios1 = self.logratios1(A['x'], B['z'])
        logratios2 = self.logratios2(A['x'], B['z'])
        return logratios1, logratios2


# +
# Let's configure, instantiate and traint the network
torch.manual_seed(28890)
trainer_drate = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 200, precision = 64)
network_drate = Network()
trainer_drate.fit(network_drate, dm_drate)

# --------------------------------------------
# Converge to val_loss = -2.59 at 105 epochs
# --------------------------------------------
# -

# ### Let's make some inference

# +
# Let's normalize testset between 0 and 1

pars_norm = (pars_testset - pars_min) / (pars_max - pars_min)

x_drate = np.log10(diff_rate_testset + 1)
x_norm_drate = (x_drate - x_min_drate) / (x_max_drate - x_min_drate)

# +
# First let's create some observation from some "true" theta parameters
#i = np.random.randint(ntest)
print(i)
pars_true = pars_norm[i,:]
x_obs     = x_norm_drate[i,:]

plt.plot(x_obs)
# -

pars_true * (pars_max - pars_min) + pars_min

# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
pars_prior = np.random.uniform(low = 0, high = 1, size = (1_000_000, 3))

prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions_drate = trainer_drate.infer(network_drate, obs, prior_samples)

# +
# Let's plot the results
swyft.corner(predictions_drate, ('pars_norm[0]', 'pars_norm[1]', 'pars_norm[2]'), bins = 200, smooth = 3)

if flag == 'exc':
    plt.savefig('../graph/cornerplot_drate_exc.pdf')
else:
    plt.savefig('../graph/cornerplot_drate.pdf')

# +
bins = 50
logratios_drate = predictions_drate[0].logratios[:,1]
v               = predictions_drate[0].params[:,1,0]
low, upp = v.min(), v.max()
weights  = torch.exp(logratios_drate) / torch.exp(logratios_drate).mean(axis = 0)
h1       = torchist.histogramdd(predictions_drate[0].params[:,1,:], bins, weights = weights, low=low, upp=upp)
h1      /= len(predictions_drate[0].params[:,1,:]) * (upp - low) / bins
h1       = np.array(h1)

edges = torch.linspace(v.min(), v.max(), bins + 1)
x     = np.array((edges[1:] + edges[:-1]) / 2) * (pars_max[1] - pars_min[1]) + pars_min[1]

# +
vals = sorted(swyft.plot.plot2.get_HDI_thresholds(h1, cred_level=[0.68268, 0.95450, 0.99730]))

low_1sigma = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
up_1sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])

low_2sigma = np.min(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
up_2sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[1]))[0]])

low_3sigma = np.min(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
up_3sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[0]))[0]])

if low_1sigma > -47.8: print('Distinguish at 1 $\sigma$')
if low_2sigma > -47.8: print('Distinguish at 2 $\sigma$')
if low_3sigma > -47.8: print('Distinguish at 3 $\sigma$')

# +
plt.plot(x, h1, c = 'blue')

#y0 = 0 #-1.0 * x.max()
#y1 = 5.0# * x.max()
#plt.fill_between(x, y0, y1, where = h1 > vals[0], color='red', alpha=0.1)
#plt.fill_between(x, y0, y1, where = h1 > vals[1], color='red', alpha=0.2)
#plt.fill_between(x, y0, y1, where = h1 > vals[2], color='red', alpha=0.3)

if low_1sigma > -47.8: plt.axvline(low_1sigma, c = 'green')
if up_1sigma > -47.8: plt.axvline(up_1sigma, c = 'green')

if low_2sigma > -47.8: plt.axvline(low_2sigma, c = 'green', linestyle = '--')
if up_2sigma > -47.8: plt.axvline(up_2sigma, c = 'green', linestyle = '--')

if low_3sigma > -47.8: plt.axvline(low_3sigma, c = 'green', linestyle = ':')
if up_3sigma > -47.8: plt.axvline(up_3sigma, c = 'green', linestyle = ':')
# -

swyft.plot_1d(predictions_drate, "pars_norm[1]", bins = 50, smooth = 1)
plt.plot(x, h1, c = 'blue')

parameters_drate = np.asarray(predictions_drate[0].params[:,:,0])
parameters_drate = parameters_drate * (pars_max - pars_min) + pars_min
parameters_drate.shape

# +
fig,ax = plt.subplots(1,3, sharey=True)

ax[0].plot(parameters_drate[:,0], predictions_drate[0].logratios[:,0], 'o', rasterized = True)
ax[0].set_xlabel(r'$m$')
ax[0].set_ylabel(r'log ratio')
ax[0].axvline(x = pars_testset[i,0])

ax[1].plot(parameters_drate[:,1], predictions_drate[0].logratios[:,1], 'o', rasterized = True)
ax[1].set_xlabel(r'$\sigma$')
ax[1].axvline(x = pars_testset[i,1])

ax[2].plot(parameters_drate[:,2], predictions_drate[0].logratios[:,2], 'o', rasterized = True)
ax[2].set_xlabel(r'$g$')
ax[2].axvline(x = pars_testset[i,2])

if flag == 'exc':
    plt.savefig('../graph/loglikratio_drate_exc.pdf')
else:
    plt.savefig('../graph/loglikratio_drate.pdf')
# -

results_pars_drate = np.asarray(predictions_drate[1].params)
results_drate = np.asarray(predictions_drate[1].logratios)

# +
fig, ax = plt.subplots(1,3, gridspec_kw = {'hspace':0.7, 'wspace':0.4}, figsize = (12,4))

#  -------------------------------- MAX  ----------------------------------------

# M vs Sigma

m_results     = 10**(results_pars_drate[:,0,0] * (pars_max[0] - pars_min[0]) + pars_min[0])
m_true        = 10**(pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0])
sigma_results = 10**(results_pars_drate[:,0,1] * (pars_max[1] - pars_min[1]) + pars_min[1])
sigma_true    = 10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1])

val, xaux, yaux,_ = stats.binned_statistic_2d(m_results, sigma_results, results_drate[:,0], 'max', bins = [np.logspace(0.81, 3, 15), np.logspace(-48.2, -41, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im20 = ax[0].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im20, ax = ax[0])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,0]), 10**(pars[:,1]), np.log10(rate + 7), 'min', bins = [np.logspace(0.81, 3, 10), np.logspace(-48.2, -41, 10)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0].contour(x_centers, y_centers, val.T, levels = [0, 1, 2, 3], cmap = 'inferno')
ax[0].axhline(y = sigma_true, c = 'red')
ax[0].axvline(x = m_true, c = 'red')
ax[0].set_xlabel('m')
ax[0].set_ylabel('$\sigma$')
ax[0].set_xscale('log')
ax[0].set_yscale('log')

# M vs theta

m_results     = 10**(results_pars_drate[:,1,0] * (pars_max[0] - pars_min[0]) + pars_min[0])
m_true        = 10**(pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0])
theta_results = results_pars_drate[:,1,1] * (pars_max[2] - pars_min[2]) + pars_min[2]
theta_true    = pars_true[2] * (pars_max[2] - pars_min[2]) + pars_min[2]

val, xaux, yaux,_ = stats.binned_statistic_2d(m_results, theta_results, results_drate[:,1], 'max', bins = [np.logspace(0.81, 3, 15), np.linspace(-1.6, 1.6, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im21 = ax[1].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im21, ax = ax[1])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,0]), pars[:,2], np.log10(rate + 7), 'min', bins = [np.logspace(0.81, 3, 10), np.linspace(-1.6, 1.6, 10)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1].contour(x_centers, y_centers, val.T, levels = [0, 1, 2, 3], cmap = 'inferno')
ax[1].axhline(y = theta_true, c = 'red')
ax[1].axvline(x = m_true, c = 'red')
ax[1].set_xlabel('m')
ax[1].set_ylabel('$\\theta$')
ax[1].set_xscale('log')

# Sigma vs theta

sigma_results = 10**(results_pars_drate[:,2,0] * (pars_max[1] - pars_min[1]) + pars_min[1])
sigma_true    = 10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1])
theta_results = results_pars_drate[:,2,1] * (pars_max[2] - pars_min[2]) + pars_min[2]
theta_true    = pars_true[2] * (pars_max[2] - pars_min[2]) + pars_min[2]

val, xaux, yaux,_ = stats.binned_statistic_2d(sigma_results, theta_results, results_drate[:,2], 'max', bins = [np.logspace(-48.2, -41, 15), np.linspace(-1.6, 1.6, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im22 = ax[2].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im22, ax = ax[2])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,1]), pars[:,2], np.log10(rate + 7), 'min', bins = [np.logspace(-48.2, -41, 10), np.linspace(-1.6, 1.6, 10)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2].contour(x_centers, y_centers, val.T, levels = [0, 1, 2, 3], cmap = 'inferno')
ax[2].axhline(y = theta_true, c = 'red')
ax[2].axvline(x = sigma_true, c = 'red')
ax[2].set_xlabel('$\sigma$')
ax[2].set_ylabel('$\\theta$')
ax[2].set_xscale('log')

if flag == 'exc':
    plt.savefig('../graph/pars_drate_exc.pdf')
else:
    plt.savefig('../graph/pars_drate.pdf')
# -

# ## Using s1s2

x_s1s2 = s1s2_trainset[:,:-1,:-1] # Observable. Input data. I am cutting a bit the images to have 64x64

# +
# Let's normalize everything between 0 and 1

pars_min = np.min(pars_trainset, axis = 0)
pars_max = np.max(pars_trainset, axis = 0)

pars_norm = (pars_trainset - pars_min) / (pars_max - pars_min)

x_min_s1s2 = np.min(x_s1s2, axis = 0)
x_max_s1s2 = np.max(x_s1s2, axis = 0)

x_norm_s1s2 = x_s1s2#(x - x_min) / (x_max - x_min)

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
from pytorch_lightning.callbacks import Callback

class MetricTracker(Callback):

    def __init__(self):
        self.collection = []
    
    def on_validation_batch_end(trainer, module, outputs):
        vacc = outputs['val_loss'] # you can access them here
        self.collection.append(vacc) # track them
    
    def on_validation_epoch_end(trainer, module):
        elogs = trainer.logged_metrics # access it here
        self.collection.append(elogs)
    # do whatever is needed


# +
# Let's configure, instantiate and traint the network
torch.manual_seed(28891)
cb = MetricTracker()
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta = 0., patience=35, verbose=False, mode='min')
trainer_s1s2 = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 250, precision = 64, callbacks=[early_stopping_callback])
network_s1s2 = Network()
story = trainer_s1s2.fit(network_s1s2, dm_s1s2)

# Do not converge at 100 epochs. Val_loss increasing!!!
# -

# ### Let's make some inference

# +
# Let's normalize testset between 0 and 1

pars_norm = (pars_testset - pars_min) / (pars_max - pars_min)

x_norm_s1s2 = x_s1s2 = s1s2_testset[:,:-1,:-1]

# +
# First let's create some observation from some "true" theta parameters
#i = np.random.randint(ntest)
print(i)

pars_true = pars_norm[i,:]
x_obs     = x_norm_s1s2[i,:].reshape(1,96,96)

plt.imshow(x_obs[0].T, origin = 'lower')
# -

pars_true * (pars_max - pars_min) + pars_min

# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
pars_prior = np.random.uniform(low = 0, high = 1, size = (1_000_000, 3))

prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions_s1s2 = trainer_s1s2.infer(network_s1s2, obs, prior_samples)

# +
# Let's plot the results
swyft.corner(predictions_s1s2, ('pars_norm[0]', 'pars_norm[1]', 'pars_norm[2]'), bins = 200, smooth = 3)

if flag == 'exc':
    plt.savefig('../graph/cornerplot_s1s2_exc.pdf')
else:
    plt.savefig('../graph/cornerplot_s1s2.pdf')

# +
bins = 50
logratios_s1s2 = predictions_s1s2[0].logratios[:,1]
v              = predictions_s1s2[0].params[:,1,0]
low, upp = v.min(), v.max()
weights  = torch.exp(logratios_s1s2) / torch.exp(logratios_s1s2).mean(axis = 0)
h1       = torchist.histogramdd(predictions_s1s2[0].params[:,1,:], bins, weights = weights, low=low, upp=upp)
h1      /= len(predictions_s1s2[0].params[:,1,:]) * (upp - low) / bins
h1       = np.array(h1)

edges = torch.linspace(v.min(), v.max(), bins + 1)
x     = np.array((edges[1:] + edges[:-1]) / 2) * (pars_max[1] - pars_min[1]) + pars_min[1]

# +
vals = sorted(swyft.plot.plot2.get_HDI_thresholds(h1, cred_level=[0.68268, 0.95450, 0.99730]))

low_1sigma = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
up_1sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])

low_2sigma = np.min(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
up_2sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[1]))[0]])

low_3sigma = np.min(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
up_3sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[0]))[0]])

if low_1sigma > -47.8: print('Distinguish at 1 $\sigma$')
if low_2sigma > -47.8: print('Distinguish at 2 $\sigma$')
if low_3sigma > -47.8: print('Distinguish at 3 $\sigma$')

# +
plt.plot(x, h1, c = 'blue')

#y0 = 0 #-1.0 * x.max()
#y1 = 5.0# * x.max()
#plt.fill_between(x, y0, y1, where = h1 > vals[0], color='red', alpha=0.1)
#plt.fill_between(x, y0, y1, where = h1 > vals[1], color='red', alpha=0.2)
#plt.fill_between(x, y0, y1, where = h1 > vals[2], color='red', alpha=0.3)

if low_1sigma > -47.8: plt.axvline(low_1sigma, c = 'green')
if up_1sigma > -47.8: plt.axvline(up_1sigma, c = 'green')

if low_2sigma > -47.8: plt.axvline(low_2sigma, c = 'green', linestyle = '--')
if up_2sigma > -47.8: plt.axvline(up_2sigma, c = 'green', linestyle = '--')

if low_3sigma > -47.8: plt.axvline(low_3sigma, c = 'green', linestyle = ':')
if up_3sigma > -47.8: plt.axvline(up_3sigma, c = 'green', linestyle = ':')
# -

swyft.plot_1d(predictions_s1s2, "pars_norm[1]", bins = 50, smooth = 1)
plt.plot(x, h1, c = 'blue')

parameters_s1s2 = np.asarray(predictions_s1s2[0].params[:,:,0])
parameters_s1s2 = parameters_s1s2 * (pars_max - pars_min) + pars_min
parameters_s1s2.shape

# +
fig,ax = plt.subplots(1,3, sharey=True)

ax[0].plot(parameters_s1s2[:,0], predictions_s1s2[0].logratios[:,0], 'o', rasterized = True)
ax[0].set_xlabel(r'$m$')
ax[0].set_ylabel(r'log ratio')
ax[0].axvline(x = pars_testset[i,0])

ax[1].plot(parameters_s1s2[:,1], predictions_s1s2[0].logratios[:,1], 'o', rasterized = True)
ax[1].set_xlabel(r'$\sigma$')
ax[1].axvline(x = pars_testset[i,1])

ax[2].plot(parameters_s1s2[:,2], predictions_s1s2[0].logratios[:,2], 'o', rasterized = True)
ax[2].set_xlabel(r'$g$')
ax[2].axvline(x = pars_testset[i,2])

if flag == 'exc':
    plt.savefig('../graph/loglikratio_s1s2_exc.pdf')
else:
    plt.savefig('../graph/loglikratio_s1s2.pdf')
# -

results_pars_s1s2 = np.asarray(predictions_s1s2[1].params)
results_s1s2      = np.asarray(predictions_s1s2[1].logratios)

# +
fig, ax = plt.subplots(1,3, gridspec_kw = {'hspace':0.7, 'wspace':0.4}, figsize = (12,4))

#  -------------------------------- MAX  ----------------------------------------

# M vs Sigma

m_results     = 10**(results_pars_s1s2[:,0,0] * (pars_max[0] - pars_min[0]) + pars_min[0])
m_true        = 10**(pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0])
sigma_results = 10**(results_pars_s1s2[:,0,1] * (pars_max[1] - pars_min[1]) + pars_min[1])
sigma_true    = 10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1])

val, xaux, yaux,_ = stats.binned_statistic_2d(m_results, sigma_results, results_s1s2[:,0], 'max', bins = [np.logspace(0.81, 3, 15), np.logspace(-48.2, -41, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im20 = ax[0].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im20, ax = ax[0])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,0]), 10**(pars[:,1]), np.log10(rate + 7), 'min', bins = [np.logspace(0.81, 3, 10), np.logspace(-48.2, -41, 10)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0].contour(x_centers, y_centers, val.T, levels = [0, 1, 2, 3], cmap = 'inferno')
ax[0].axhline(y = sigma_true, c = 'red')
ax[0].axvline(x = m_true, c = 'red')
ax[0].set_xlabel('m')
ax[0].set_ylabel('$\sigma$')
ax[0].set_xscale('log')
ax[0].set_yscale('log')

# M vs theta

m_results     = 10**(results_pars_s1s2[:,1,0] * (pars_max[0] - pars_min[0]) + pars_min[0])
m_true        = 10**(pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0])
theta_results = results_pars_s1s2[:,1,1] * (pars_max[2] - pars_min[2]) + pars_min[2]
theta_true    = pars_true[2] * (pars_max[2] - pars_min[2]) + pars_min[2]

val, xaux, yaux,_ = stats.binned_statistic_2d(m_results, theta_results, results_s1s2[:,1], 'max', bins = [np.logspace(0.81, 3, 15), np.linspace(-1.6, 1.6, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im21 = ax[1].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im21, ax = ax[1])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,0]), pars[:,2], np.log10(rate + 7), 'min', bins = [np.logspace(0.81, 3, 10), np.linspace(-1.6, 1.6, 10)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1].contour(x_centers, y_centers, val.T, levels = [0, 1, 2, 3], cmap = 'inferno')
ax[1].axhline(y = theta_true, c = 'red')
ax[1].axvline(x = m_true, c = 'red')
ax[1].set_xlabel('m')
ax[1].set_ylabel('$\\theta$')
ax[1].set_xscale('log')

# Sigma vs theta

sigma_results = 10**(results_pars_s1s2[:,2,0] * (pars_max[1] - pars_min[1]) + pars_min[1])
sigma_true    = 10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1])
theta_results = results_pars_s1s2[:,2,1] * (pars_max[2] - pars_min[2]) + pars_min[2]
theta_true    = pars_true[2] * (pars_max[2] - pars_min[2]) + pars_min[2]

val, xaux, yaux,_ = stats.binned_statistic_2d(sigma_results, theta_results, results_s1s2[:,2], 'max', bins = [np.logspace(-48.2, -41, 15), np.linspace(-1.6, 1.6, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im22 = ax[2].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im22, ax = ax[2])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,1]), pars[:,2], np.log10(rate + 7), 'min', bins = [np.logspace(-48.2, -41, 10), np.linspace(-1.6, 1.6, 10)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2].contour(x_centers, y_centers, val.T, levels = [0, 1, 2, 3], cmap = 'inferno')
ax[2].axhline(y = theta_true, c = 'red')
ax[2].axvline(x = sigma_true, c = 'red')
ax[2].set_xlabel('$\sigma$')
ax[2].set_ylabel('$\\theta$')
ax[2].set_xscale('log')

if flag == 'exc':
    plt.savefig('../graph/pars_s1s2_exc.pdf')
else:
    plt.savefig('../graph/pars_s1s2.pdf')
# -



