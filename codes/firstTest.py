import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import swyft
from tqdm import tqdm
import time
from scipy import stats
import seaborn as sbn
import pandas as pd

# It is usefull to print the versions of the package that we are using
print('swyft version:', swyft.__version__)
print('numpy version:', np.__version__)
print('matplotlib version:', mpl.__version__)

# # Let's load the data

data = np.loadtxt('../data/andresData/primerTest.dat')

data.shape

# # Let's play with SWYFT

x = np.log10(data[:,3] + 7) # Observable. Input data. I am adding 7 backgorund events to everything
theta = data[:,0:3] # Pars

theta_aux = theta[np.where(np.abs(theta[:,2] - 1) < 0.04)[0],:]

x_aux = x[np.where(np.abs(theta[:,2] - 1) < 0.04)[0]]

np.min(x_aux)

# +
# Let's normalize everything between 0 and 1

theta_min = np.min(theta, axis = 0)
theta_max = np.max(theta, axis = 0)

theta_norm = (theta - theta_min) / (theta_max - theta_min)

x_min = np.min(x, axis = 0)
x_max = np.max(x, axis = 0)

x_norm = (x - x_min) / (x_max - x_min)

# +
fig,ax = plt.subplots(2,2, gridspec_kw = {'hspace':0.5, 'wspace':0.5})

ax[0,0].hist(x_norm)
ax[0,0].set_xlabel('# Events')

ax[1,0].hist(theta_norm[:,0])
ax[1,0].set_xlabel('$M_{DM}$')

ax[0,1].hist(theta_norm[:,1])
ax[0,1].set_xlabel('$\sigma$')

ax[1,1].hist(theta_norm[:,2])
ax[1,1].set_xlabel('$g$')

# -

np.log10(4.449 * np.sqrt(7) + 7)

# +
fig, ax = plt.subplots(3,3, gridspec_kw = {'hspace':0.5, 'wspace':0.5}, figsize = (10,10))

# MIN 
val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,0], theta_norm[:,1], x, 'min', bins = 10)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im00 = ax[0,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = 0, vmax = 8)
ax[0,0].contour(x_centers, y_centers, val.T, levels = [0., 1.27])
plt.colorbar(im00, ax = ax[0,0])
ax[0,0].set_xlabel('m')
ax[0,0].set_ylabel('$\sigma$')

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,0], theta_norm[:,2], x, 'min', bins = 10)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im01 = ax[0,1].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = 0, vmax = 8)
ax[0,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
plt.colorbar(im01, ax = ax[0,1])
ax[0,1].set_xlabel('m')
ax[0,1].set_ylabel('$g$')

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,1], theta_norm[:,2], x, 'min', bins = 10)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im02 = ax[0,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = 0, vmax = 8)
ax[0,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
plt.colorbar(im02, ax = ax[0,2])
ax[0,2].set_xlabel('$\sigma$')
ax[0,2].set_ylabel('$g$')

# MEAN
val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,0], theta_norm[:,1], x, 'mean', bins = 10)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im10 = ax[1,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = 0, vmax = 8)
ax[1,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
plt.colorbar(im10, ax = ax[1,0])
ax[1,0].set_xlabel('m')
ax[1,0].set_ylabel('$\sigma$')

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,0], theta_norm[:,2], x, 'mean', bins = 10)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im11 = ax[1,1].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = 0, vmax = 8)
ax[1,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
plt.colorbar(im11, ax = ax[1,1])
ax[1,1].set_xlabel('m')
ax[1,1].set_ylabel('$g$')

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,1], theta_norm[:,2], x, 'mean', bins = 10)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im12 = ax[1,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = 0, vmax = 8)
ax[1,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
plt.colorbar(im12, ax = ax[1,2])
ax[1,2].set_xlabel('$\sigma$')
ax[1,2].set_ylabel('$g$')

# MAX 

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,0], theta_norm[:,1], x, 'max', bins = 10)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im20 = ax[2,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = 0, vmax = 8)
ax[2,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
plt.colorbar(im20, ax = ax[2,0])
ax[2,0].set_xlabel('m')
ax[2,0].set_ylabel('$\sigma$')

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,0], theta_norm[:,2], x, 'max', bins = 10)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im21 = ax[2,1].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = 0, vmax = 8)
ax[2,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
plt.colorbar(im21, ax = ax[2,1])
ax[2,1].set_xlabel('m')
ax[2,1].set_ylabel('$g$')

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,1], theta_norm[:,2], x, 'max', bins = 10)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im22 = ax[2,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = 0, vmax = 8)
ax[2,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
plt.colorbar(im22, ax = ax[2,2])
ax[2,2].set_xlabel('$\sigma$')
ax[2,2].set_ylabel('$g$')
# -

x_norm = x_norm.reshape(len(x_norm), 1)
print(x_norm.shape)
print(theta_norm.shape)

# +
# We have to build a swyft.Samples object that will handle the data
samples = swyft.Samples(x = x_norm, z = theta_norm)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm = swyft.SwyftDataModule(samples, fractions = [0.7, 0.25, 0.05])


# -

# Now let's define a network that estimates all the 1D and 2D marginal posteriors
class Network(swyft.SwyftModule):
    def __init__(self):
        super().__init__()
        marginals = ((0, 1), (0, 2), (1, 2))
        self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = 1, num_params = 3, varnames = 'theta_norm')
        self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features = 1, marginals = marginals, varnames = 'theta_norm')

    def forward(self, A, B):
        logratios1 = self.logratios1(A['x'], B['z'])
        logratios2 = self.logratios2(A['x'], B['z'])
        return logratios1, logratios2


# Let's configure, instantiate and traint the network
trainer = swyft.SwyftTrainer(accelerator = 'gpu', devices=1, max_epochs = 50, precision = 64)
network = Network()
trainer.fit(network, dm)

# # Let's make some inference

# +
# First let's create some observation from some "true" theta parameters
i = 21000
#theta_true = theta_norm[i,:]
theta_true = theta_norm[i,:]
x_obs = x_norm[i,:]

print('"Observed" x value : {}'.format(10**x[i]))

# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
theta_prior = np.random.uniform(low = 0, high = 1, size = (1000000, 3))
#theta_prior = np.random.uniform(low = [0, -50, -2], high = [1000, -40, 2], size = (1000000, 3))
prior_samples = swyft.Samples(z = theta_prior)

# Finally we make the inference
predictions = trainer.infer(network, obs, prior_samples)
# -

# Let's plot the results
swyft.corner(predictions, ('theta_norm[0]', 'theta_norm[1]', 'theta_norm[2]'), bins = 200, smooth = 3);

pars = np.asarray(predictions[0].params[:,:,0])
pars = pars * (theta_max - theta_min) + theta_min
pars.shape

# +
fig,ax = plt.subplots(1,3, sharey=True)

ax[0].scatter(pars[:,0], predictions[0].logratios[:,0])
ax[0].set_xlabel(r'$m$')
ax[0].set_ylabel(r'log ratio')
ax[0].axvline(x = theta[i,0])

ax[1].scatter(pars[:,1], predictions[0].logratios[:,1])
ax[1].set_xlabel(r'$\sigma$')
ax[1].axvline(x = theta[i,1])

ax[2].scatter(pars[:,2], predictions[0].logratios[:,2])
ax[2].set_xlabel(r'$g$')
ax[2].axvline(x = theta[i,2])

# -

results_pars = np.asarray(predictions[1].params)
results = np.asarray(predictions[1].logratios)

results[:,0]

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

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,0], theta_norm[:,1], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[0,0].axhline(y = theta_true[1], c = 'red')
ax[0,0].axvline(x = theta_true[0], c = 'red')
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

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,0], theta_norm[:,2], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[0,1].axhline(y = theta_true[2], c = 'red')
ax[0,1].axvline(x = theta_true[0], c = 'red')
ax[0,1].set_xlabel('m')
ax[0,1].set_ylabel('$g$')

# Sigma vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,2,0], results_pars[:,2,1], results[:,2], 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im02 = ax[0,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im02, ax = ax[0,2])

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,1], theta_norm[:,2], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[0,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[0,2].axhline(y = theta_true[2], c = 'red')
ax[0,2].axvline(x = theta_true[1], c = 'red')
ax[0,2].set_xlabel('$\sigma$')
ax[0,2].set_ylabel('$g$')

# ------------------------------ MEAN ------------------------------

# M vs Sigma

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,0,0], results_pars[:,0,1], results[:,0], 'mean', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im10 = ax[1,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im10, ax = ax[1,0])

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,0], theta_norm[:,1], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[1,0].axhline(y = theta_true[1], c = 'red')
ax[1,0].axvline(x = theta_true[0], c = 'red')
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

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,0], theta_norm[:,2], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[1,1].axhline(y = theta_true[2], c = 'red')
ax[1,1].axvline(x = theta_true[0], c = 'red')
ax[1,1].set_xlabel('m')
ax[1,1].set_ylabel('$g$')

# Sigma vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,2,0], results_pars[:,2,1], results[:,2], 'mean', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im12 = ax[1,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im12, ax = ax[1,2])

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,1], theta_norm[:,2], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[1,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[1,2].axhline(y = theta_true[2], c = 'red')
ax[1,2].axvline(x = theta_true[1], c = 'red')
ax[1,2].set_xlabel('$\sigma$')
ax[1,2].set_ylabel('$g$')

#  -------------------------------- MAX  ----------------------------------------

# M vs Sigma

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,0,0], results_pars[:,0,1], results[:,0], 'max', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im20 = ax[2,0].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im20, ax = ax[2,0])

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,0], theta_norm[:,1], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2,0].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[2,0].axhline(y = theta_true[1], c = 'red')
ax[2,0].axvline(x = theta_true[0], c = 'red')
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

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,0], theta_norm[:,2], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2,1].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[2,1].axhline(y = theta_true[2], c = 'red')
ax[2,1].axvline(x = theta_true[0], c = 'red')
ax[2,1].set_xlabel('m')
ax[2,1].set_ylabel('$g$')

# Sigma vs g

val, xaux, yaux,_ = stats.binned_statistic_2d(results_pars[:,2,0], results_pars[:,2,1], results[:,2], 'max', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im22 = ax[2,2].contourf(x_centers, y_centers, val.T, alpha = 0.6, vmin = -15, vmax = 8)
plt.colorbar(im22, ax = ax[2,2])

val, xaux, yaux,_ = stats.binned_statistic_2d(theta_norm[:,1], theta_norm[:,2], x, 'min', bins = 15)
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin
ax[2,2].contour(x_centers, y_centers, val.T, levels = [0, 1.27])
ax[2,2].axhline(y = theta_true[2], c = 'red')
ax[2,2].axvline(x = theta_true[1], c = 'red')
ax[2,2].set_xlabel('$\sigma$')
ax[2,2].set_ylabel('$g$')
# -



