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

pars      = np.load('../data/andresData/data10krandom/pars.npy')
diff_rate = np.load('../data/andresData/data10krandom/diff_rate.npy')
rate      = np.load('../data/andresData/data10krandom/rate.npy')
s1s2      = np.load('../data/andresData/data10krandom/s1s2.npy')

print(pars.shape)
print(rate.shape)
print(diff_rate.shape)
print(s1s2.shape)

# ## Let's make some exploratory plots

# +
fig, ax = plt.subplots(1,3, figsize = (10,5))

ax[0].hist(np.log(pars[:,0]), histtype = 'step')
ax[0].set_xlabel('m [GeV?]')
#ax[0].set_xscale('log')

ax[1].hist(np.log(pars[:,1]), histtype = 'step')
ax[1].set_xlabel('$\sigma$ [?]')
#ax[1].set_xscale('log')

ax[2].hist(pars[:,2], histtype = 'step')
ax[2].set_xlabel('$\\theta$')

# +
i = np.random.randint(len(pars))

fig, ax = plt.subplots(1,2, figsize = (10,5))

ax[0].plot(diff_rate[i,:])
ax[0].set_xlabel('$E_{r}$ [keV]' )
ax[0].set_ylabel('$dR/E_{r}$' )
ax[0].text(0.5, 0.8,  'm = {:.2f} [?]'.format(pars[i,0]), transform = ax[0].transAxes)
ax[0].text(0.5, 0.7,  '$\sigma$ = {:.2e} [?]'.format(pars[i,1]), transform = ax[0].transAxes)
ax[0].text(0.5, 0.6, '$\\theta$ = {:.2f}'.format(pars[i,2]), transform = ax[0].transAxes)
ax[0].text(0.5, 0.5, 'Total Rate = {:.2e}'.format(rate[i]), transform = ax[0].transAxes)

ax[1].imshow(s1s2[i], origin = 'lower')
ax[1].set_xlabel('s1')
ax[1].set_ylabel('s2')
# -

# # Let's play with SWYFT

# ## Only using the total rate

np.max(rate)

x = np.log10(rate + 7) # Observable. Input data. I am adding 7 backgorund events to everything

# +
# Let's normalize everything between 0 and 1

pars_min = np.min(pars, axis = 0)
pars_max = np.max(pars, axis = 0)

pars_norm = (pars - pars_min) / (pars_max - pars_min)

x_min = np.min(x, axis = 0)
x_max = np.max(x, axis = 0)

x_norm = (x - x_min) / (x_max - x_min)

# +
fig,ax = plt.subplots(2,2, gridspec_kw = {'hspace':0.5, 'wspace':0.5})

ax[0,0].hist(x_norm)
ax[0,0].set_xlabel('# Events')

ax[1,0].hist(pars_norm[:,0])
ax[1,0].set_xlabel('$M_{DM}$')

ax[0,1].hist(pars_norm[:,1])
ax[0,1].set_xlabel('$\sigma$')

ax[1,1].hist(pars_norm[:,2])
ax[1,1].set_xlabel('$\\theta$')

# -

x_norm = x_norm.reshape(len(x_norm), 1)
print(x_norm.shape)
print(pars_norm.shape)

# +
# We have to build a swyft.Samples object that will handle the data
samples = swyft.Samples(x = x_norm, z = pars_norm)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm = swyft.SwyftDataModule(samples, fractions = [0.7, 0.25, 0.05])


# -

# Now let's define a network that estimates all the 1D and 2D marginal posteriors
class Network(swyft.SwyftModule):
    def __init__(self):
        super().__init__()
        marginals = ((0, 1), (0, 2), (1, 2))
        self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = 1, num_params = 3, varnames = 'pars_norm')
        self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features = 1, marginals = marginals, varnames = 'pars_norm')

    def forward(self, A, B):
        logratios1 = self.logratios1(A['x'], B['z'])
        logratios2 = self.logratios2(A['x'], B['z'])
        return logratios1, logratios2


# Let's configure, instantiate and traint the network
trainer = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 10, precision = 64)
network = Network()
trainer.fit(network, dm)

# ### Let's make some inference

# +
# First let's create some observation from some "true" theta parameters
i = 1000
#theta_true = theta_norm[i,:]
pars_true = pars_norm[i,:]
x_obs = x_norm[i,:]

print('"Observed" x value : {}'.format(x_obs))

# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
pars_prior = np.random.uniform(low = 0, high = 1, size = (10000, 3))
#theta_prior = np.random.uniform(low = [0, -50, -2], high = [1000, -40, 2], size = (1000000, 3))
prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions = trainer.infer(network, obs, prior_samples)
# -

# Let's plot the results
swyft.corner(predictions, ('pars_norm[0]', 'pars_norm[1]', 'pars_norm[2]'), bins = 200, smooth = 3)

parameters = np.asarray(predictions[0].params[:,:,0])
parameters = parameters * (pars_max - pars_min) + pars_min
parameters.shape

# +
fig,ax = plt.subplots(1,3, sharey=True)

ax[0].scatter(parameters[:,0], predictions[0].logratios[:,0])
ax[0].set_xlabel(r'$m$')
ax[0].set_ylabel(r'log ratio')
ax[0].axvline(x = pars[i,0])

ax[1].scatter(parameters[:,1], predictions[0].logratios[:,1])
ax[1].set_xlabel(r'$\sigma$')
ax[1].axvline(x = pars[i,1])

ax[2].scatter(parameters[:,2], predictions[0].logratios[:,2])
ax[2].set_xlabel(r'$g$')
ax[2].axvline(x = pars[i,2])
# -

results_pars = np.asarray(predictions[1].params)
results = np.asarray(predictions[1].logratios)

# +
fig, ax = plt.subplots(3,3, gridspec_kw = {'hspace':0.5, 'wspace':0.5}, figsize = (10,10))

# ------------------------- MIN ----------------------------------------

# M vs Sigma
val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,0,0], results_pars[:,0,1], results[:,0], 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im00 = ax[0,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im00, ax = ax[0,0])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,1], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[0,0].axhline(y = pars_true[1], c = 'red')
ax[0,0].axvline(x = pars_true[0], c = 'red')
ax[0,0].set_xlabel('m')
ax[0,0].set_ylabel('$\sigma$')

# M vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,1,0], results_pars[:,1,1], results[:,1], 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im01 = ax[0,1].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im01, ax = ax[0,1])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,2], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[0,1].axhline(y = pars_true[2], c = 'red')
ax[0,1].axvline(x = pars_true[0], c = 'red')
ax[0,1].set_xlabel('m')
ax[0,1].set_ylabel('$\\theta$')

# Sigma vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,2,0], results_pars[:,2,1], results[:,2], 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im02 = ax[0,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im02, ax = ax[0,2])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,1], pars_norm[:,2], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[0,2].axhline(y = pars_true[2], c = 'red')
ax[0,2].axvline(x = pars_true[1], c = 'red')
ax[0,2].set_xlabel('$\sigma$')
ax[0,2].set_ylabel('$\\theta$')

# ------------------------------ MEAN ------------------------------

# M vs Sigma

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,0,0], results_pars[:,0,1], results[:,0], 'mean', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im10 = ax[1,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im10, ax = ax[1,0])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,1], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[1,0].axhline(y = pars_true[1], c = 'red')
ax[1,0].axvline(x = pars_true[0], c = 'red')
ax[1,0].set_xlabel('m')
ax[1,0].set_ylabel('$\sigma$')

# M vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,1,0], results_pars[:,1,1], results[:,1], 'mean', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im11 = ax[1,1].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im11, ax = ax[1,1])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,2], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[1,1].axhline(y = pars_true[2], c = 'red')
ax[1,1].axvline(x = pars_true[0], c = 'red')
ax[1,1].set_xlabel('m')
ax[1,1].set_ylabel('$\\theta$')

# Sigma vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,2,0], results_pars[:,2,1], results[:,2], 'mean', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im12 = ax[1,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im12, ax = ax[1,2])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,1], pars_norm[:,2], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[1,2].axhline(y = pars_true[2], c = 'red')
ax[1,2].axvline(x = pars_true[1], c = 'red')
ax[1,2].set_xlabel('$\sigma$')
ax[1,2].set_ylabel('$\\theta$')

#  -------------------------------- MAX  ----------------------------------------

# M vs Sigma

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,0,0], results_pars[:,0,1], results[:,0], 'max', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im20 = ax[2,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im20, ax = ax[2,0])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,1], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[2,0].axhline(y = pars_true[1], c = 'red')
ax[2,0].axvline(x = pars_true[0], c = 'red')
ax[2,0].set_xlabel('m')
ax[2,0].set_ylabel('$\sigma$')

# M vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,1,0], results_pars[:,1,1], results[:,1], 'max', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im21 = ax[2,1].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im21, ax = ax[2,1])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,2], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[2,1].axhline(y = pars_true[2], c = 'red')
ax[2,1].axvline(x = pars_true[0], c = 'red')
ax[2,1].set_xlabel('m')
ax[2,1].set_ylabel('$\\theta$')

# Sigma vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,2,0], results_pars[:,2,1], results[:,2], 'max', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im22 = ax[2,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im22, ax = ax[2,2])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,1], pars_norm[:,2], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[2,2].axhline(y = pars_true[2], c = 'red')
ax[2,2].axvline(x = pars_true[1], c = 'red')
ax[2,2].set_xlabel('$\sigma$')
ax[2,2].set_ylabel('$\\theta$')
# -

# ## Only using the total diff_rate

x = diff_rate # Observable. Input data. I am cutting a bit the images to have 64x64

# +
# Let's normalize everything between 0 and 1

pars_min = np.min(pars, axis = 0)
pars_max = np.max(pars, axis = 0)

pars_norm = (pars - pars_min) / (pars_max - pars_min)

x_min = np.min(x, axis = 0)
x_max = np.max(x, axis = 0)

x_norm = (x - x_min) / (x_max - x_min)

# +
fig,ax = plt.subplots(2,2, gridspec_kw = {'hspace':0.5, 'wspace':0.5})

for i in range(50):
    ax[0,0].plot(x_norm[i])
ax[0,0].set_xlabel('$E_{r}$')

ax[1,0].hist(pars_norm[:,0])
ax[1,0].set_xlabel('$M_{DM}$')

ax[0,1].hist(pars_norm[:,1])
ax[0,1].set_xlabel('$\sigma$')

ax[1,1].hist(pars_norm[:,2])
ax[1,1].set_xlabel('$\\theta$')
# -

print(x_norm.shape)
print(pars_norm.shape)

# +
# We have to build a swyft.Samples object that will handle the data
samples = swyft.Samples(x = x_norm, z = pars_norm)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm = swyft.SwyftDataModule(samples, fractions = [0.7, 0.25, 0.05], batch_size = 32)


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


# Let's configure, instantiate and traint the network
trainer = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 5, precision = 64)
network = Network()
trainer.fit(network, dm)

# ### Let's make some inference

# +
# First let's create some observation from some "true" theta parameters
i = 1000
#theta_true = theta_norm[i,:]
pars_true = pars_norm[i,:]
x_obs = x_norm[i,:]

plt.plot(x_obs)

# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
pars_prior = np.random.uniform(low = 0, high = 1, size = (10000, 3))
#theta_prior = np.random.uniform(low = [0, -50, -2], high = [1000, -40, 2], size = (1000000, 3))
prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions = trainer.infer(network, obs, prior_samples)
# -

# Let's plot the results
swyft.corner(predictions, ('pars_norm[0]', 'pars_norm[1]', 'pars_norm[2]'), bins = 200, smooth = 3);

parameters = np.asarray(predictions[0].params[:,:,0])
parameters = parameters * (pars_max - pars_min) + pars_min
parameters.shape

# +
fig,ax = plt.subplots(1,3, sharey=True)

ax[0].scatter(parameters[:,0], predictions[0].logratios[:,0])
ax[0].set_xlabel(r'$m$')
ax[0].set_ylabel(r'log ratio')
ax[0].axvline(x = pars[i,0])

ax[1].scatter(parameters[:,1], predictions[0].logratios[:,1])
ax[1].set_xlabel(r'$\sigma$')
ax[1].axvline(x = pars[i,1])

ax[2].scatter(parameters[:,2], predictions[0].logratios[:,2])
ax[2].set_xlabel(r'$g$')
ax[2].axvline(x = pars[i,2])
# -

results_pars = np.asarray(predictions[1].params)
results = np.asarray(predictions[1].logratios)

# +
fig, ax = plt.subplots(3,3, gridspec_kw = {'hspace':0.5, 'wspace':0.5}, figsize = (10,10))

# ------------------------- MIN ----------------------------------------

# M vs Sigma
val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,0,0], results_pars[:,0,1], results[:,0], 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im00 = ax[0,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im00, ax = ax[0,0])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,1], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[0,0].axhline(y = pars_true[1], c = 'red')
ax[0,0].axvline(x = pars_true[0], c = 'red')
ax[0,0].set_xlabel('m')
ax[0,0].set_ylabel('$\sigma$')

# M vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,1,0], results_pars[:,1,1], results[:,1], 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im01 = ax[0,1].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im01, ax = ax[0,1])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,2], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[0,1].axhline(y = pars_true[2], c = 'red')
ax[0,1].axvline(x = pars_true[0], c = 'red')
ax[0,1].set_xlabel('m')
ax[0,1].set_ylabel('$\\theta$')

# Sigma vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,2,0], results_pars[:,2,1], results[:,2], 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im02 = ax[0,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im02, ax = ax[0,2])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,1], pars_norm[:,2], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[0,2].axhline(y = pars_true[2], c = 'red')
ax[0,2].axvline(x = pars_true[1], c = 'red')
ax[0,2].set_xlabel('$\sigma$')
ax[0,2].set_ylabel('$\\theta$')

# ------------------------------ MEAN ------------------------------

# M vs Sigma

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,0,0], results_pars[:,0,1], results[:,0], 'mean', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im10 = ax[1,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im10, ax = ax[1,0])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,1], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[1,0].axhline(y = pars_true[1], c = 'red')
ax[1,0].axvline(x = pars_true[0], c = 'red')
ax[1,0].set_xlabel('m')
ax[1,0].set_ylabel('$\sigma$')

# M vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,1,0], results_pars[:,1,1], results[:,1], 'mean', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im11 = ax[1,1].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im11, ax = ax[1,1])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,2], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[1,1].axhline(y = pars_true[2], c = 'red')
ax[1,1].axvline(x = pars_true[0], c = 'red')
ax[1,1].set_xlabel('m')
ax[1,1].set_ylabel('$\\theta$')

# Sigma vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,2,0], results_pars[:,2,1], results[:,2], 'mean', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im12 = ax[1,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im12, ax = ax[1,2])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,1], pars_norm[:,2], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[1,2].axhline(y = pars_true[2], c = 'red')
ax[1,2].axvline(x = pars_true[1], c = 'red')
ax[1,2].set_xlabel('$\sigma$')
ax[1,2].set_ylabel('$\\theta$')

#  -------------------------------- MAX  ----------------------------------------

# M vs Sigma

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,0,0], results_pars[:,0,1], results[:,0], 'max', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im20 = ax[2,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im20, ax = ax[2,0])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,1], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[2,0].axhline(y = pars_true[1], c = 'red')
ax[2,0].axvline(x = pars_true[0], c = 'red')
ax[2,0].set_xlabel('m')
ax[2,0].set_ylabel('$\sigma$')

# M vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,1,0], results_pars[:,1,1], results[:,1], 'max', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im21 = ax[2,1].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im21, ax = ax[2,1])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,2], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[2,1].axhline(y = pars_true[2], c = 'red')
ax[2,1].axvline(x = pars_true[0], c = 'red')
ax[2,1].set_xlabel('m')
ax[2,1].set_ylabel('$\\theta$')

# Sigma vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,2,0], results_pars[:,2,1], results[:,2], 'max', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im22 = ax[2,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im22, ax = ax[2,2])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,1], pars_norm[:,2], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[2,2].axhline(y = pars_true[2], c = 'red')
ax[2,2].axvline(x = pars_true[1], c = 'red')
ax[2,2].set_xlabel('$\sigma$')
ax[2,2].set_ylabel('$\\theta$')
# -

# ## Only using s1s2

x = s1s2[:,3:-3,1:-2] # Observable. Input data. I am cutting a bit the images to have 64x64

# +
# Let's normalize everything between 0 and 1

pars_min = np.min(pars, axis = 0)
pars_max = np.max(pars, axis = 0)

pars_norm = (pars - pars_min) / (pars_max - pars_min)

x_min = np.min(x, axis = 0)
x_max = np.max(x, axis = 0)

x_norm = x#(x - x_min) / (x_max - x_min)

# +
fig,ax = plt.subplots(2,2, gridspec_kw = {'hspace':0.5, 'wspace':0.5})

ax[0,0].hist(x_norm[:,50,30])
ax[0,0].set_xlabel('# Events')

ax[1,0].hist(pars_norm[:,0])
ax[1,0].set_xlabel('$M_{DM}$')

ax[0,1].hist(pars_norm[:,1])
ax[0,1].set_xlabel('$\sigma$')

ax[1,1].hist(pars_norm[:,2])
ax[1,1].set_xlabel('$\\theta$')

# -

x_norm = x_norm.reshape(len(x_norm), 1, 64, 64) # The shape need to be (#obs, #channels, dim, dim)
print(x_norm.shape)
print(pars_norm.shape)

# +
# We have to build a swyft.Samples object that will handle the data
samples = swyft.Samples(x = x_norm, z = pars_norm)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm = swyft.SwyftDataModule(samples, fractions = [0.7, 0.25, 0.05], batch_size = 32)


# -

# Now let's define a network that estimates all the 1D and 2D marginal posteriors
class Network(swyft.SwyftModule):
    def __init__(self, lr = 1e-3, gamma = 1.):
        super().__init__()
        self.optimizer_init = swyft.OptimizerInit(torch.optim.Adam, dict(lr = lr),
              torch.optim.lr_scheduler.ExponentialLR, dict(gamma = gamma))
        self.net = torch.nn.Sequential(
          torch.nn.Conv2d(1, 10, kernel_size=5),
          torch.nn.MaxPool2d(2),
          torch.nn.ReLU(),
          torch.nn.Conv2d(10, 20, kernel_size=5, padding=2),
          torch.nn.MaxPool2d(2),
          torch.nn.ReLU(),
          torch.nn.Flatten(),
          torch.nn.Linear(4500, 50),
          torch.nn.ReLU(),
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


# Let's configure, instantiate and traint the network
trainer = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 5, precision = 64)
network = Network()
trainer.fit(network, dm)

# ### Let's make some inference

# +
# First let's create some observation from some "true" theta parameters
i = 1000
#theta_true = theta_norm[i,:]
pars_true = pars_norm[i,:]
x_obs = x_norm[i,:]

plt.imshow(x_obs[0], origin = 'lower')

# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
pars_prior = np.random.uniform(low = 0, high = 1, size = (10000, 3))
#theta_prior = np.random.uniform(low = [0, -50, -2], high = [1000, -40, 2], size = (1000000, 3))
prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions = trainer.infer(network, obs, prior_samples)
# -

# Let's plot the results
swyft.corner(predictions, ('pars_norm[0]', 'pars_norm[1]', 'pars_norm[2]'), bins = 200, smooth = 3);

parameters = np.asarray(predictions[0].params[:,:,0])
parameters = parameters * (pars_max - pars_min) + pars_min
parameters.shape

predictions[0].logratios[:,0].shape

# +
fig,ax = plt.subplots(1,3, sharey=True)

ax[0].scatter(parameters[:,0], predictions[0].logratios[:,0])
ax[0].set_xlabel(r'$m$')
ax[0].set_ylabel(r'log ratio')
ax[0].axvline(x = pars[i,0])

ax[1].scatter(parameters[:,1], predictions[0].logratios[:,1])
ax[1].set_xlabel(r'$\sigma$')
ax[1].axvline(x = pars[i,1])

ax[2].scatter(parameters[:,2], predictions[0].logratios[:,2])
ax[2].set_xlabel(r'$g$')
ax[2].axvline(x = pars[i,2])

# -

results_pars = np.asarray(predictions[1].params)
results = np.asarray(predictions[1].logratios)

results[:,0].shape

# +
fig, ax = plt.subplots(3,3, gridspec_kw = {'hspace':0.5, 'wspace':0.5}, figsize = (10,10))

# ------------------------- MIN ----------------------------------------

# M vs Sigma
val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,0,0], results_pars[:,0,1], results[:,0], 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im00 = ax[0,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im00, ax = ax[0,0])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,1], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[0,0].axhline(y = pars_true[1], c = 'red')
ax[0,0].axvline(x = pars_true[0], c = 'red')
ax[0,0].set_xlabel('m')
ax[0,0].set_ylabel('$\sigma$')

# M vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,1,0], results_pars[:,1,1], results[:,1], 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im01 = ax[0,1].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im01, ax = ax[0,1])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,2], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[0,1].axhline(y = pars_true[2], c = 'red')
ax[0,1].axvline(x = pars_true[0], c = 'red')
ax[0,1].set_xlabel('m')
ax[0,1].set_ylabel('$\\theta$')

# Sigma vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,2,0], results_pars[:,2,1], results[:,2], 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im02 = ax[0,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im02, ax = ax[0,2])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,1], pars_norm[:,2], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[0,2].axhline(y = pars_true[2], c = 'red')
ax[0,2].axvline(x = pars_true[1], c = 'red')
ax[0,2].set_xlabel('$\sigma$')
ax[0,2].set_ylabel('$\\theta$')

# ------------------------------ MEAN ------------------------------

# M vs Sigma

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,0,0], results_pars[:,0,1], results[:,0], 'mean', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im10 = ax[1,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im10, ax = ax[1,0])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,1], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[1,0].axhline(y = pars_true[1], c = 'red')
ax[1,0].axvline(x = pars_true[0], c = 'red')
ax[1,0].set_xlabel('m')
ax[1,0].set_ylabel('$\sigma$')

# M vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,1,0], results_pars[:,1,1], results[:,1], 'mean', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im11 = ax[1,1].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im11, ax = ax[1,1])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,2], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[1,1].axhline(y = pars_true[2], c = 'red')
ax[1,1].axvline(x = pars_true[0], c = 'red')
ax[1,1].set_xlabel('m')
ax[1,1].set_ylabel('$\\theta$')

# Sigma vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,2,0], results_pars[:,2,1], results[:,2], 'mean', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im12 = ax[1,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im12, ax = ax[1,2])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,1], pars_norm[:,2], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[1,2].axhline(y = pars_true[2], c = 'red')
ax[1,2].axvline(x = pars_true[1], c = 'red')
ax[1,2].set_xlabel('$\sigma$')
ax[1,2].set_ylabel('$\\theta$')

#  -------------------------------- MAX  ----------------------------------------

# M vs Sigma

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,0,0], results_pars[:,0,1], results[:,0], 'max', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im20 = ax[2,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im20, ax = ax[2,0])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,1], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[2,0].axhline(y = pars_true[1], c = 'red')
ax[2,0].axvline(x = pars_true[0], c = 'red')
ax[2,0].set_xlabel('m')
ax[2,0].set_ylabel('$\sigma$')

# M vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,1,0], results_pars[:,1,1], results[:,1], 'max', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im21 = ax[2,1].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im21, ax = ax[2,1])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,0], pars_norm[:,2], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[2,1].axhline(y = pars_true[2], c = 'red')
ax[2,1].axvline(x = pars_true[0], c = 'red')
ax[2,1].set_xlabel('m')
ax[2,1].set_ylabel('$\\theta$')

# Sigma vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,2,0], results_pars[:,2,1], results[:,2], 'max', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im22 = ax[2,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im22, ax = ax[2,2])

val, xaux, yaux,_ = stats.binned_statistic_2d(pars_norm[:,1], pars_norm[:,2], rate, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[2,2].axhline(y = pars_true[2], c = 'red')
ax[2,2].axvline(x = pars_true[1], c = 'red')
ax[2,2].set_xlabel('$\sigma$')
ax[2,2].set_ylabel('$\\theta$')