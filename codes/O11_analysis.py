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
from matplotlib.pyplot import contour, show
from matplotlib.lines import Line2D

import torch
import torchist
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import Callback

import seaborn as sns

torch.set_float32_matmul_precision('high')
pallete = np.flip(sns.color_palette("tab20c", 8), axis = 0)
cross_sec_th = -49

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


# # Let's load the data

# !ls ../data/andresData/O11-full/O11

# +
# where are your files?
datFolder = ['../data/andresData/O11-full/O11/O11-run01/']
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
print(rate_raw[i,2] - rate[i])
print(rate_raw[i,2] - np.sum(diff_rate[i,:]))

plt.scatter(rate_raw[:,2], np.sum(diff_rate, axis = 1))

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

# ## Neutrino Fog

neutrino_fog = np.loadtxt('../data/neutrino_fog.csv', skiprows = 1, delimiter = ',')

# ## Xenon data
#
# from https://arxiv.org/pdf/2007.08796.pdf (Figure 6)

xenon_nt_5s   = np.loadtxt('../data/xenon_nt_5sigma.csv', skiprows = 1, delimiter = ',')
xenon_nt_3s   = np.loadtxt('../data/xenon_nt_3sigma.csv', skiprows = 1, delimiter = ',')
xenon_nt_90cl = np.loadtxt('../data/xenon_nt_90cl.csv', skiprows = 1, delimiter = ',')

# !ls ../data/andresData/BL-constraints-PARAO11/BL-constraints/

# +
masses = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/masses.txt')[:30]

rate_90_CL_pi2  = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-rate-thetapi2.txt')
rate_90_CL_pi4  = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-rate-thetapi4.txt')
rate_90_CL_0    = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-rate-theta0.txt')
rate_90_CL_mpi2 = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-rate-thetampi2.txt')
rate_90_CL_mpi4 = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-rate-thetampi4.txt')

rate_current_pi2  = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-rate-thetapi2-current.txt')
rate_current_pi4  = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-rate-thetapi4-current.txt')
rate_current_0    = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-rate-theta0-current.txt')
rate_current_mpi2 = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-rate-thetampi2-current.txt')
rate_current_mpi4 = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-rate-thetampi4-current.txt')

s1s2_90_CL_pi2  = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-s1s2-thetapi2.txt')
s1s2_90_CL_pi4  = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-s1s2-thetapi4.txt')
s1s2_90_CL_0    = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-s1s2-theta0.txt')
s1s2_90_CL_mpi2 = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-s1s2-thetampi2.txt')
s1s2_90_CL_mpi4 = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-s1s2-thetampi4.txt')

s1s2_current_pi2  = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-s1s2-thetapi2-current.txt')
s1s2_current_pi4  = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-s1s2-thetapi4-current.txt')
s1s2_current_0    = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-s1s2-theta0-current.txt')
s1s2_current_mpi2 = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-s1s2-thetampi2-current.txt')
s1s2_current_mpi4 = np.loadtxt('../data/andresData/BL-constraints-PARAO11/BL-constraints/BL-s1s2-thetampi4-current.txt')
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
i = np.random.randint(len(pars_testset))
print(i)
fig, ax = plt.subplots(1,2, figsize = (10,5))

ax[0].plot(diff_rate_testset[i,:], c = 'black')
ax[0].plot(diff_rate_WIMP[test_ind[i],:], c = 'black', linestyle = ':')
ax[0].set_xlabel('$E_{r}$ [keV]' )
ax[0].set_ylabel('$dR/E_{r}$' )
ax[0].text(0.5, 0.8,  '$\log_{10} $' + 'm = {:.2f} [?]'.format(pars_testset[i,0]), transform = ax[0].transAxes)
ax[0].text(0.5, 0.7,  '$\log_{10}\sigma$' + ' = {:.2f} [?]'.format(pars_testset[i,1]), transform = ax[0].transAxes)
ax[0].text(0.5, 0.6, '$\\theta$ = {:.2f}'.format(pars_testset[i,2]), transform = ax[0].transAxes)
ax[0].text(0.5, 0.5, 'Total Rate = {:.3f}'.format(rate_testset[i]), transform = ax[0].transAxes)
#ax[0].set_yscale('log')

ax[1].imshow(s1s2_testset[i].T, origin = 'lower')
ax[1].set_xlabel('s1')
ax[1].set_ylabel('s2')
# -

# # Let's play with SWYFT

# ## Using only the total rate with background 

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
# -

pars_min

pars_max

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
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O11_rate_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
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
    checkpoint_callback.to_yaml("./logs/O11_rate.yaml") 
    ckpt_path = swyft.best_from_yaml("./logs/O11_rate.yaml")
else:
    ckpt_path = swyft.best_from_yaml("./logs/O11_rate.yaml")

# ---------------------------------------------- 
# It converges to val_loss =  at epoch ~50
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
# It converges to val_loss ~ -0.9 in testset
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
    plt.savefig('../graph/O11_loss_rate.pdf')

email('Termino el training de O11')

# ### Let's make some inference (NOT IMPLEMENTED) 

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
pars_prior    = np.random.uniform(low = 0, high = 1, size = (10_000, 3))
#pars_prior = np.random.uniform(low = pars_min, high = pars_max, size = (100_000, 3))
#pars_prior = np.random.uniform(low = [6, 1e-50, -1.57], high = [1000, 1e-45, 1.57], size = (100_000, 3))
#pars_prior[:,0] = np.log10(pars_prior[:,0])
#pars_prior[:,1] = np.log10(pars_prior[:,1])
#pars_prior = (pars_prior - pars_min) / (pars_max - pars_min)

prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions_rate = trainer_rate.infer(network_rate, obs, prior_samples)
# -

# Let's plot the results
swyft.corner(predictions_rate, ('pars_norm[0]', 'pars_norm[1]', 'pars_norm[2]'), bins = 200, smooth = 3)
if flag == 'exc':
    plt.savefig('../graph/011_cornerplot_rate_exc.pdf')
else:
    plt.savefig('../graph/011_cornerplot_rate.pdf')

# +
cross_sec = np.asarray(predictions_rate[0].params[:,1,0]) * B + A
ratios = np.exp(np.asarray(predictions_rate[0].logratios[:,1]))

ind_sort = np.argsort(cross_sec)
ratios = ratios[ind_sort]
cross_sec = cross_sec[ind_sort]
# -

plt.plot(cross_sec, ratios, c = 'blue')
plt.xlabel('$\log_{10}(\sigma)$')
plt.ylabel('$P(\sigma|x)\ /\ P(\sigma)$')
#plt.yscale('log')
#plt.xscale('log')

trapezoid(ratios, cross_sec)

# +
cr_th = np.argmin(np.abs(cross_sec + 45))

trapezoid(ratios[cr_th:], cross_sec[cr_th:]) / trapezoid(ratios, cross_sec)

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
cross_section_th = -45
vals = sorted(swyft.plot.plot2.get_HDI_thresholds(h1, cred_level=[0.68268, 0.95450, 0.99730]))

low_1sigma = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
up_1sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])

low_2sigma = np.min(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
up_2sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[1]))[0]])

low_3sigma = np.min(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
up_3sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[0]))[0]])

if low_1sigma > cross_section_th: print('Distinguish at 1 $\sigma$')
if low_2sigma > cross_section_th: print('Distinguish at 2 $\sigma$')
if low_3sigma > cross_section_th: print('Distinguish at 3 $\sigma$')


# +
plt.plot(x, h1, c = 'black')
plt.axvline(x = pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1], c = 'orange')

#y0 = 0 #-1.0 * x.max()
#y1 = 5.0# * x.max()
#plt.fill_between(x, y0, y1, where = h1 > vals[0], color='red', alpha=0.1)
#plt.fill_between(x, y0, y1, where = h1 > vals[1], color='red', alpha=0.2)
#plt.fill_between(x, y0, y1, where = h1 > vals[2], color='red', alpha=0.3)

if low_1sigma > cross_section_th: plt.axvline(low_1sigma, c = 'black')
if up_1sigma > cross_section_th: plt.axvline(up_1sigma, c = 'black')

if low_2sigma > cross_section_th: plt.axvline(low_2sigma, c = 'black', linestyle = '--')
if up_2sigma > cross_section_th: plt.axvline(up_2sigma, c = 'black', linestyle = '--')

if low_3sigma > cross_section_th: plt.axvline(low_3sigma, c = 'black', linestyle = ':')
if up_3sigma > cross_section_th: plt.axvline(up_3sigma, c = 'black', linestyle = ':')
#plt.ylim(0,4.5)
#plt.xscale('log')
plt.xlabel('$log(\sigma)$')
plt.ylabel('$P(\sigma|x)$')
plt.text(-43,3, '$m = {:.2e}$'.format(10**(pars_true[0])))
plt.text(-43,2.8, '$\sigma = {:.2e}$'.format(10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1])))
plt.text(-43,2.5, '$\\theta = {:.2f}$'.format(pars_true[0]))
plt.plot(cross_sec, ratios, c = 'blue')

if flag == 'exc':
    plt.savefig('../graph/011_1Dposterior_rate_exc_' + str(i) + '.pdf')
else:
    plt.savefig('../graph/011_1Dposterior_rate_' + str(i) + '.pdf')
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
    plt.savefig('../graph/011_loglikratio_rate_exc.pdf')
else:
    plt.savefig('../graph/011_loglikratio_rate.pdf')
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

val, xaux, yaux,_ = stats.binned_statistic_2d(m_results, sigma_results, results_rate[:,0], 'max', bins = [np.logspace(0.81, 3, 15), np.logspace(-43.2, -35, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im20 = ax[0].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im20, ax = ax[0])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,0]), 10**(pars[:,1]), np.log10(rate + 7), 'min', bins = [np.logspace(0.81, 3, 10), np.logspace(-43.2, -35, 10)])
    
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

val, xaux, yaux,_ = stats.binned_statistic_2d(sigma_results, theta_results, results_rate[:,2], 'max', bins = [np.logspace(-43.2, -35, 15), np.linspace(-1.6, 1.6, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im22 = ax[2].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im22, ax = ax[2])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,1]), pars[:,2], np.log10(rate + 7), 'min', bins = [np.logspace(-43.2, -35, 10), np.linspace(-1.6, 1.6, 10)])
    
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
    plt.savefig('../graph/011_pars_rate_exc.pdf')
else:
    plt.savefig('../graph/011_pars_rate.pdf')
# -

# ### Let's make the contour plot

# !ls ../data/andresData/O11-full/O11/theta-minuspidiv2

pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice(['../data/andresData/O11-full/O11/theta-minuspidiv2/O11-slices01-minuspidiv2-v4/'])

m_vals = np.logspace(np.min(pars_slices[:,0]), np.max(pars_slices[:,0]),30)
cross_vals = np.logspace(np.min(pars_slices[:,1]), np.max(pars_slices[:,1]),30)

# +
#'../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0/'
#'../data/andresData/O11-full/O11/theta-minuspidiv2/O11-slices01-minuspidiv2-v5/'

# +
force = False
folders = [#'../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0/',
           '../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0-v2/',
           '../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0-v3/',
           '../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0-v4/',
           '../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0-v5/'
         ]


cross_sec_sigmas_full       = []
cross_sec_int_prob_full     = []
cross_sec_int_prob_sup_full = []

masses_int_prob_sup_full = []

for folder in folders:
    pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice([folder])

    if (os.path.exists(folder + 'cross_sec_sigmas_rate.txt') &
        os.path.exists(folder + 'cross_sec_int_prob_rate.txt') &
        os.path.exists(folder + 'cross_sec_int_prob_sup_rate.txt') &
        os.path.exists(folder + 'masses_int_prob_sup_rate.txt')
       ) == False or force == True:
        # Let's normalize testset between 0 and 1

        pars_norm = (pars_slices - pars_min) / (pars_max - pars_min)

        x_rate = np.log10(rate_slices)
        x_norm_rate = (x_rate - x_min_rate) / (x_max_rate - x_min_rate)
        x_norm_rate = x_norm_rate.reshape(len(x_norm_rate), 1)

        cross_sec_sigmas = np.ones((len(pars_slices), 6))

        cross_sec_int_prob     = np.ones(len(pars_norm)) * -99
        cross_sec_int_prob_sup = np.ones(len(pars_norm)) * -99
        masses_int_prob_sup    = np.ones(len(pars_norm)) * -99

        for itest in tqdm(range(len(pars_norm))):
            x_obs = x_norm_rate[itest, :]

            # We have to put this "observation" into a swyft.Sample object
            obs = swyft.Sample(x = x_obs)

            # Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
            pars_prior    = np.random.uniform(low = 0, high = 1, size = (10_000, 3))
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

            cross_sec_sigmas[itest,0] = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
            cross_sec_sigmas[itest,3] = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])

            cross_sec_sigmas[itest,1] = np.min(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
            cross_sec_sigmas[itest,4] = np.max(x[np.where(np.array(h1) > np.array(vals[1]))[0]])

            cross_sec_sigmas[itest,2] = np.min(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
            cross_sec_sigmas[itest,5] = np.max(x[np.where(np.array(h1) > np.array(vals[0]))[0]])

            cr_th = np.argmin(np.abs(x - (-45)))
            cross_sec_int_prob[itest]     = trapezoid(h1[:cr_th], x[:cr_th]) / trapezoid(h1, x)
            cross_sec_int_prob_sup[itest] = trapezoid(h1[cr_th:], x[cr_th:]) / trapezoid(h1, x)

            ratios_rate = np.exp(np.asarray(predictions_rate[0].logratios[:,0]))
            masses_pred = np.asarray(predictions_rate[0].params[:,0,0]) * (pars_max[0] - pars_min[0]) + pars_min[0]
            ind_sort    = np.argsort(masses_pred)
            ratios_rate = ratios_rate[ind_sort]
            masses_pred = masses_pred[ind_sort]
            m_min = np.argmin(np.abs(masses_pred - 1))
            m_max = np.argmin(np.abs(masses_pred - 2.6))
            masses_int_prob_sup[itest] = trapezoid(ratios_rate[m_min:m_max], masses_pred[m_min:m_max]) / trapezoid(ratios_rate, masses_pred)

        cross_sec_sigmas_full.append(cross_sec_sigmas)
        cross_sec_int_prob_full.append(cross_sec_int_prob)
        cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
        masses_int_prob_sup_full.append(masses_int_prob_sup)

        np.savetxt(folder + 'cross_sec_sigmas_rate.txt', cross_sec_sigmas)
        np.savetxt(folder + 'cross_sec_int_prob_rate.txt', cross_sec_int_prob)
        np.savetxt(folder + 'cross_sec_int_prob_sup_rate.txt', cross_sec_int_prob_sup)
        np.savetxt(folder + 'masses_int_prob_sup_rate.txt', masses_int_prob_sup)
    else:
        print('pre-computed')

        cross_sec_sigmas = np.loadtxt(folder + 'cross_sec_sigmas_rate.txt')
        cross_sec_int_prob = np.loadtxt(folder + 'cross_sec_int_prob_rate.txt')
        cross_sec_int_prob_sup = np.loadtxt(folder + 'cross_sec_int_prob_sup_rate.txt')
        masses_int_prob_sup = np.loadtxt(folder + 'masses_int_prob_sup_rate.txt')

        cross_sec_sigmas_full.append(cross_sec_sigmas)
        cross_sec_int_prob_full.append(cross_sec_int_prob)
        cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
        masses_int_prob_sup_full.append(masses_int_prob_sup)
email('Termino analisis slices para O11')

# +
cross_section_th = -45

if len(cross_sec_int_prob_full) > 1:
    cross_sec_int_prob_rate_0        = np.mean(np.asarray(cross_sec_int_prob_full), axis = 0)
    cross_sec_int_prob_sup_rate_0    = np.mean(np.asarray(cross_sec_int_prob_sup_full), axis = 0)
    cross_sec_int_prob_sup_rate_0_sd = np.std(np.asarray(cross_sec_int_prob_sup_full), axis = 0)
    masses_int_prob_sup_rate_0       = np.mean(np.asarray(masses_int_prob_sup_full), axis = 0)
    masses_int_prob_sup_rate_0_sd    = np.std(np.asarray(masses_int_prob_sup_full), axis = 0)
    cross_sec_sigmas = np.mean(np.asarray(cross_sec_sigmas_full), axis = 0)
else:
    cross_sec_int_prob_rate_0     = cross_sec_int_prob
    cross_sec_int_prob_sup_rate_0 = cross_sec_int_prob_sup
    masses_int_prob_sup_rate_0    = masses_int_prob_sup

rate_1sigma_0 = np.ones(900) * -99
rate_2sigma_0 = np.ones(900) * -99
rate_3sigma_0 = np.ones(900) * -99

rate_1sigma_0[np.where(cross_sec_sigmas[:,0] > cross_section_th)[0]] = 1
rate_2sigma_0[np.where(cross_sec_sigmas[:,1] > cross_section_th)[0]] = 1
rate_3sigma_0[np.where(cross_sec_sigmas[:,2] > cross_section_th)[0]] = 1

# +
fig, ax = plt.subplots(1,2)

sbn.kdeplot(cross_sec_int_prob_sup_rate_0, label = '$\\theta = 0$', ax = ax[0])
sbn.kdeplot(cross_sec_int_prob_sup_rate_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[0])
sbn.kdeplot(cross_sec_int_prob_sup_rate_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[0])
sbn.kdeplot(cross_sec_int_prob_sup_rate_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[0])
sbn.kdeplot(cross_sec_int_prob_sup_rate_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[0])
ax[0].legend()
ax[0].set_xlabel('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')
ax[0].set_title('Total Rate')

sbn.kdeplot(masses_int_prob_sup_rate_0, label = '$\\theta = 0$', ax = ax[1])
sbn.kdeplot(masses_int_prob_sup_rate_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[1])
sbn.kdeplot(masses_int_prob_sup_rate_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[1])
sbn.kdeplot(masses_int_prob_sup_rate_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[1])
sbn.kdeplot(masses_int_prob_sup_rate_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[1])
ax[1].legend()
ax[1].set_xlabel('$\int_{m_{min}}^{m_{max}} P(m_{DM}|x)$')
ax[1].set_title('Total Rate')

#plt.savefig('../graph/O11_int_prob_distribution_rate.pdf')

# +
sigma = 0.2 # this depends on how noisy your data is, play with it!

rate_1sigma_0_g = gaussian_filter(rate_1sigma_0, sigma)
rate_1sigma_pi_2_g = gaussian_filter(rate_1sigma_pi_2, sigma)
rate_1sigma_pi_4_g = gaussian_filter(rate_1sigma_pi_4, sigma)
rate_1sigma_mpi_2_g = gaussian_filter(rate_1sigma_mpi_2, sigma)
rate_1sigma_mpi_4_g = gaussian_filter(rate_1sigma_mpi_4, sigma)

rate_2sigma_0_g = gaussian_filter(rate_2sigma_0, sigma)
rate_2sigma_pi_2_g = gaussian_filter(rate_2sigma_pi_2, sigma)
rate_2sigma_pi_4_g = gaussian_filter(rate_2sigma_pi_4, sigma)
rate_2sigma_mpi_2_g = gaussian_filter(rate_2sigma_mpi_2, sigma)
rate_2sigma_mpi_4_g = gaussian_filter(rate_2sigma_mpi_4, sigma)

rate_3sigma_0_g = gaussian_filter(rate_3sigma_0, sigma)
rate_3sigma_pi_2_g = gaussian_filter(rate_3sigma_pi_2, sigma)
rate_3sigma_pi_4_g = gaussian_filter(rate_3sigma_pi_4, sigma)
rate_3sigma_mpi_2_g = gaussian_filter(rate_3sigma_mpi_2, sigma)
rate_3sigma_mpi_4_g = gaussian_filter(rate_3sigma_mpi_4, sigma)

fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))

ax[0,0].contour(m_vals, cross_vals, rate_1sigma_pi_2_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[0,0].contour(m_vals, cross_vals, rate_2sigma_pi_2_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[0,0].contourf(m_vals, cross_vals, rate_3sigma_pi_2_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, rate_3sigma_pi_2_g.reshape(30,30).T, levels=[0])

ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-44, '$\\theta = \pi/2$')
#ax[0,0].legend(loc = 'lower right')

ax[0,1].contour(m_vals, cross_vals, rate_1sigma_pi_4_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[0,1].contour(m_vals, cross_vals, rate_2sigma_pi_4_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[0,1].contourf(m_vals, cross_vals, rate_3sigma_pi_4_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, rate_3sigma_pi_4_g.reshape(30,30).T, levels=[0])

ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-44, '$\\theta = \pi/4$')
ax[0,1].legend(loc = 'lower right')

#ax[1,0].contour(m_vals, cross_vals, int_prob_0.reshape(30,30).T, levels=10, linewidths = 2, zorder = 4, linestyles = '--')
ax[1,0].contour(m_vals, cross_vals, rate_1sigma_mpi_2_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[1,0].contour(m_vals, cross_vals, rate_2sigma_mpi_2_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[1,0].contourf(m_vals, cross_vals, rate_3sigma_mpi_2_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, rate_3sigma_mpi_2_g.reshape(30,30).T, levels=[0])

ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-44, '$\\theta = -\pi/2$')

ax[1,1].contour(m_vals, cross_vals, rate_1sigma_0_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[1,1].contour(m_vals, cross_vals, rate_2sigma_0_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[1,1].contourf(m_vals, cross_vals, rate_3sigma_0_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, rate_3sigma_0_g.reshape(30,30).T, levels=[0])

ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-44, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma$ []')
ax[1,0].set_ylabel('$\sigma$ []')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(1e-46, 1e-40)

plt.savefig('../graph/O11_contours_rate.pdf')

# +
levels = 5#[0, 0.1, 0.16, 0.24, 0.32]

sigma = 1.81 # this depends on how noisy your data is, play with it!

int_prob_0_g = gaussian_filter(cross_sec_int_prob_rate_0, sigma)
int_prob_pi_2_g = gaussian_filter(cross_sec_int_prob_rate_pi_2, sigma)
int_prob_pi_4_g = gaussian_filter(cross_sec_int_prob_rate_pi_4, sigma)
int_prob_mpi_2_g = gaussian_filter(cross_sec_int_prob_rate_mpi_2, sigma)
int_prob_mpi_4_g = gaussian_filter(cross_sec_int_prob_rate_mpi_4, sigma)

fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))

fig00 = ax[0,0].contourf(m_vals, cross_vals, int_prob_pi_2_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, int_prob_pi_2_g.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)

ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-44, '$\\theta = \pi/2$')
ax[0,0].plot(masses, rate_90_CL_pi2[2,:], color = 'black', linestyle = '-.', label = 'Bin. Lik. [90%]')
ax[0,0].legend(loc = 'lower left')

ax[0,1].contourf(m_vals, cross_vals, int_prob_pi_4_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, int_prob_pi_4_g.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)

ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-44, '$\\theta = \pi/4$')

ax[1,0].contourf(m_vals, cross_vals, int_prob_mpi_2_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, int_prob_mpi_2_g.reshape(30,30).T, levels=levels)

ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-44, '$\\theta = -\pi/2$')

ax[1,1].contourf(m_vals, cross_vals, int_prob_0_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, int_prob_0_g.reshape(30,30).T, levels=levels)

ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-44, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(1e-46, 2e-41)
ax[0,0].set_xlim(7, 1e3)
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar    = fig.colorbar(fig00, cax=cbar_ax)
cbar.ax.set_title('$\int_{-\inf}^{\sigma_{th}} P(\sigma|x)$')

ax[0,1].plot(masses, rate_90_CL_pi4[2,:], color = 'black', linestyle = '-.')
ax[1,0].plot(masses, rate_90_CL_mpi2[2,:], color = 'black', linestyle = '-.')
ax[1,1].plot(masses, rate_90_CL_0[2,:], color = 'black', linestyle = '-.')

plt.savefig('../graph/O11_contours_rate_int_prob.pdf')

# +
sigma = 2.1 # this depends on how noisy your data is, play with it!

CR_int_prob_sup_0_rate          = gaussian_filter(cross_sec_int_prob_sup_rate_0, sigma)
CR_int_prob_sup_0_rate_max      = gaussian_filter(cross_sec_int_prob_sup_rate_0 + cross_sec_int_prob_sup_rate_0_sd, sigma)
CR_int_prob_sup_0_rate_min      = gaussian_filter(cross_sec_int_prob_sup_rate_0 - cross_sec_int_prob_sup_rate_0_sd, sigma)
CR_int_prob_sup_pi_2_rate       = gaussian_filter(cross_sec_int_prob_sup_rate_pi_2, sigma)
CR_int_prob_sup_pi_2_rate_max   = gaussian_filter(cross_sec_int_prob_sup_rate_pi_2 + cross_sec_int_prob_sup_rate_pi_2_sd, sigma)
CR_int_prob_sup_pi_2_rate_min   = gaussian_filter(cross_sec_int_prob_sup_rate_pi_2 - cross_sec_int_prob_sup_rate_pi_2_sd, sigma)
CR_int_prob_sup_pi_4_rate       = gaussian_filter(cross_sec_int_prob_sup_rate_pi_4, sigma)
CR_int_prob_sup_pi_4_rate_max   = gaussian_filter(cross_sec_int_prob_sup_rate_pi_4 + cross_sec_int_prob_sup_rate_pi_4_sd, sigma)
CR_int_prob_sup_pi_4_rate_min   = gaussian_filter(cross_sec_int_prob_sup_rate_pi_4 - cross_sec_int_prob_sup_rate_pi_4_sd, sigma)
CR_int_prob_sup_mpi_2_rate      = gaussian_filter(cross_sec_int_prob_sup_rate_mpi_2, sigma)
CR_int_prob_sup_mpi_2_rate_max  = gaussian_filter(cross_sec_int_prob_sup_rate_mpi_2 + cross_sec_int_prob_sup_rate_mpi_2_sd, sigma)
CR_int_prob_sup_mpi_2_rate_min  = gaussian_filter(cross_sec_int_prob_sup_rate_mpi_2 - cross_sec_int_prob_sup_rate_mpi_2_sd, sigma)
CR_int_prob_sup_mpi_4_rate      = gaussian_filter(cross_sec_int_prob_sup_rate_mpi_4, sigma)
CR_int_prob_sup_mpi_4_rate_max  = gaussian_filter(cross_sec_int_prob_sup_rate_mpi_4 + cross_sec_int_prob_sup_rate_mpi_4_sd, sigma)
CR_int_prob_sup_mpi_4_rate_min  = gaussian_filter(cross_sec_int_prob_sup_rate_mpi_4 - cross_sec_int_prob_sup_rate_mpi_4_sd, sigma)

M_int_prob_sup_0_rate         = gaussian_filter(masses_int_prob_sup_rate_0, sigma)
M_int_prob_sup_0_rate_max     = gaussian_filter(masses_int_prob_sup_rate_0 + masses_int_prob_sup_rate_0_sd, sigma)
M_int_prob_sup_0_rate_min     = gaussian_filter(masses_int_prob_sup_rate_0 - masses_int_prob_sup_rate_0_sd, sigma)
M_int_prob_sup_pi_2_rate      = gaussian_filter(masses_int_prob_sup_rate_pi_2, sigma)
M_int_prob_sup_pi_2_rate_max  = gaussian_filter(masses_int_prob_sup_rate_pi_2 + masses_int_prob_sup_rate_pi_2_sd, sigma)
M_int_prob_sup_pi_2_rate_min  = gaussian_filter(masses_int_prob_sup_rate_pi_2 - masses_int_prob_sup_rate_pi_2_sd, sigma)
M_int_prob_sup_pi_4_rate      = gaussian_filter(masses_int_prob_sup_rate_pi_4, sigma)
M_int_prob_sup_pi_4_rate_max  = gaussian_filter(masses_int_prob_sup_rate_pi_4 + masses_int_prob_sup_rate_pi_4_sd, sigma)
M_int_prob_sup_pi_4_rate_min  = gaussian_filter(masses_int_prob_sup_rate_pi_4 - masses_int_prob_sup_rate_pi_4_sd, sigma)
M_int_prob_sup_mpi_2_rate     = gaussian_filter(masses_int_prob_sup_rate_mpi_2, sigma)
M_int_prob_sup_mpi_2_rate_max = gaussian_filter(masses_int_prob_sup_rate_mpi_2 + masses_int_prob_sup_rate_mpi_2_sd, sigma)
M_int_prob_sup_mpi_2_rate_min = gaussian_filter(masses_int_prob_sup_rate_mpi_2 - masses_int_prob_sup_rate_mpi_2_sd, sigma)
M_int_prob_sup_mpi_4_rate     = gaussian_filter(masses_int_prob_sup_rate_mpi_4, sigma)
M_int_prob_sup_mpi_4_rate_max = gaussian_filter(masses_int_prob_sup_rate_mpi_4 + masses_int_prob_sup_rate_mpi_4_sd, sigma)
M_int_prob_sup_mpi_4_rate_min = gaussian_filter(masses_int_prob_sup_rate_mpi_4 - masses_int_prob_sup_rate_mpi_4_sd, sigma)

# +
levels = [0.46, 0.76, 0.84, 0.9, 1]

fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))
fig.subplots_adjust(hspace = 0, wspace = 0)

fig00 = ax[0,0].contourf(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)
#ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate_max.reshape(30,30).T, levels=[0.9], linewidths = 1, zorder = 4)
#ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate_min.reshape(30,30).T, levels=[0.9], linewidths = 1, zorder = 4)

ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-42, '$\\theta = \pi/2$')
ax[0,0].plot(masses, rate_90_CL_pi2[2,:], color = 'black', linestyle = '-.', label = 'Bin. Lik. [90%]')
ax[0,0].legend(loc = 'lower left')

ax[0,1].contourf(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)

ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-42, '$\\theta = \pi/4$')

ax[1,0].contourf(m_vals, cross_vals, CR_int_prob_sup_mpi_2_rate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, CR_int_prob_sup_mpi_2_rate.reshape(30,30).T, levels=levels)

ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-42, '$\\theta = -\pi/2$')

ax[1,1].contourf(m_vals, cross_vals, CR_int_prob_sup_0_rate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate.reshape(30,30).T, levels=levels)

ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-42, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(2e-46, 2e-41)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(fig00, cax=cbar_ax)
cbar.ax.set_title('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')

ax[0,1].plot(masses, rate_90_CL_pi4[2,:], color = 'black', linestyle = '-.')
ax[1,0].plot(masses, rate_90_CL_mpi2[2,:], color = 'black', linestyle = '-.')
ax[1,1].plot(masses, rate_90_CL_0[2,:], color = 'black', linestyle = '-.')

plt.savefig('../graph/O11_contours_rate_int_prob_sup.pdf')
# -

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
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O11_drate_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
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
    checkpoint_callback.to_yaml("./logs/O11_drate.yaml") 
    ckpt_path = swyft.best_from_yaml("./logs/O11_drate.yaml")
else:
    ckpt_path = swyft.best_from_yaml("./logs/O11_drate.yaml")

# ---------------------------------------------- 
# It converges to val_loss =  @ epoch 20
# ---------------------------------------------- 

# +
x_test_drate = diff_rate_testset
x_norm_test_drate = (x_test_drate - x_min_drate) / (x_max_drate - x_min_drate)

pars_norm_test = (pars_testset - pars_min) / (pars_max - pars_min)

# We have to build a swyft.Samples object that will handle the data
samples_test_drate = swyft.Samples(x = x_norm_test_drate, z = pars_norm_test)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_test_drate = swyft.SwyftDataModule(samples_test_drate, fractions = [0., 0., 1], batch_size = 32)
trainer_drate.test(network_drate, dm_test_drate, ckpt_path = ckpt_path)

# ---------------------------------------------- 
# It converges to  @ testset
# ---------------------------------------------- 
# -

if fit:
    email('termino el entrenamiento')
    
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
    plt.savefig('../graph/O11_loss_drate.pdf')

# ### Let's make some inference (NOT IMPLEMENTED)

# +
# Let's normalize testset between 0 and 1

pars_norm = (pars_testset - pars_min) / (pars_max - pars_min)

x_drate = diff_rate_testset
x_norm_drate = (x_drate - x_min_drate) / (x_max_drate - x_min_drate)

# +
# First let's create some observation from some "true" theta parameters
i = np.random.randint(ntest)
print(i)
pars_true = pars_norm[i,:]
x_obs     = x_norm_drate[i,:]

plt.plot(x_obs)
plt.text(5,0.5, str(np.sum(x_drate[i,:])))
if np.sum(diff_rate_WIMP[test_ind[i],:]) < 300: 
    flag = 'exc'
else:
    flag = 'disc'
print(np.sum(diff_rate_WIMP[test_ind[i],:]))
print(flag)
# -

pars_true * (pars_max - pars_min) + pars_min

# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
pars_prior = np.random.uniform(low = 0, high = 1, size = (100_000, 3))

prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions_drate = trainer_drate.infer(network_drate, obs, prior_samples)

# +
# Let's plot the results
swyft.corner(predictions_drate, ('pars_norm[0]', 'pars_norm[1]', 'pars_norm[2]'), bins = 200, smooth = 3)

if flag == 'exc':
    plt.savefig('../graph/O4_cornerplot_drate_exc.pdf')
else:
    plt.savefig('../graph/O4_cornerplot_drate.pdf')

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

if low_1sigma > -41: print('Distinguish at 1 $\sigma$')
if low_2sigma > -41: print('Distinguish at 2 $\sigma$')
if low_3sigma > -41: print('Distinguish at 3 $\sigma$')

# +
plt.plot(x, h1, c = 'blue')

#y0 = 0 #-1.0 * x.max()
#y1 = 5.0# * x.max()
#plt.fill_between(x, y0, y1, where = h1 > vals[0], color='red', alpha=0.1)
#plt.fill_between(x, y0, y1, where = h1 > vals[1], color='red', alpha=0.2)
#plt.fill_between(x, y0, y1, where = h1 > vals[2], color='red', alpha=0.3)

if low_1sigma > -41: plt.axvline(low_1sigma, c = 'green')
if up_1sigma > -41: plt.axvline(up_1sigma, c = 'green')

if low_2sigma > -41: plt.axvline(low_2sigma, c = 'green', linestyle = '--')
if up_2sigma > -41: plt.axvline(up_2sigma, c = 'green', linestyle = '--')

if low_3sigma > -41: plt.axvline(low_3sigma, c = 'green', linestyle = ':')
if up_3sigma > -41: plt.axvline(up_3sigma, c = 'green', linestyle = ':')

# +
plt.plot(x, h1, c = 'black')
plt.axvline(x = pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1], c = 'orange')

if low_1sigma > cross_section_th: plt.axvline(low_1sigma, c = 'black')
if up_1sigma > cross_section_th: plt.axvline(up_1sigma, c = 'black')

if low_2sigma > cross_section_th: plt.axvline(low_2sigma, c = 'black', linestyle = '--')
if up_2sigma > cross_section_th: plt.axvline(up_2sigma, c = 'black', linestyle = '--')

if low_3sigma > cross_section_th: plt.axvline(low_3sigma, c = 'black', linestyle = ':')
if up_3sigma > cross_section_th: plt.axvline(up_3sigma, c = 'black', linestyle = ':')
#plt.ylim(0,4.5)
#plt.xscale('log')
plt.xlabel('$log(\sigma)$')
plt.ylabel('$P(\sigma|x)$')
plt.text(-43,3, '$m = {:.2e}$'.format(10**(pars_true[0])))
plt.text(-43,2.8, '$\sigma = {:.2e}$'.format(10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1])))
plt.text(-43,2.5, '$\\theta = {:.2f}$'.format(pars_true[0]))
if flag == 'exc':
    plt.savefig('../graph/O4_1Dposterior_drate_exc_' + str(i) + '.pdf')
else:
    plt.savefig('../graph/O4_1Dposterior_drate_' + str(i) + '.pdf')
# -

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
    plt.savefig('../graph/O4_loglikratio_drate_exc.pdf')
else:
    plt.savefig('../graph/O4_loglikratio_drate.pdf')
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

val, xaux, yaux,_ = stats.binned_statistic_2d(m_results, sigma_results, results_drate[:,0], 'max', bins = [np.logspace(0.81, 3, 15), np.logspace(-43, -35, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im20 = ax[0].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im20, ax = ax[0])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,0]), 10**(pars[:,1]), np.log10(rate + 7), 'min', bins = [np.logspace(0.81, 3, 10), np.logspace(-43, -35, 10)])
    
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

val, xaux, yaux,_ = stats.binned_statistic_2d(sigma_results, theta_results, results_drate[:,2], 'max', bins = [np.logspace(-43, -35, 15), np.linspace(-1.6, 1.6, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im22 = ax[2].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im22, ax = ax[2])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,1]), pars[:,2], np.log10(rate + 7), 'min', bins = [np.logspace(-43, -35, 10), np.linspace(-1.6, 1.6, 10)])
    
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
    plt.savefig('../graph/O4_pars_drate_exc.pdf')
else:
    plt.savefig('../graph/O4_pars_drate.pdf')
# -

# ### Let's make the contour plot

# !ls ../data/andresData/O11-full/O11

pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice(['../data/andresData/O11-full/O11/theta-minuspidiv2/O11-slices01-minuspidiv2/'])

m_vals = np.logspace(np.min(pars_slices[:,0]), np.max(pars_slices[:,0]),30)
cross_vals = np.logspace(np.min(pars_slices[:,1]), np.max(pars_slices[:,1]),30)

# +
#'../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0/'
#'../data/andresData/O11-full/O11/theta-minuspidiv2/O11-slices01-minuspidiv2-v5/'

# +
force = False
folders = ['../data/andresData/O11-full/O11/theta-minuspidiv2/O11-slices01-minuspidiv2/',
           '../data/andresData/O11-full/O11/theta-minuspidiv2/O11-slices01-minuspidiv2-v2/',
           '../data/andresData/O11-full/O11/theta-minuspidiv2/O11-slices01-minuspidiv2-v3/',
           '../data/andresData/O11-full/O11/theta-minuspidiv2/O11-slices01-minuspidiv2-v4/'#,
           #'../data/andresData/O11-full/O11/theta-minuspidiv2/O11-slices01-minuspidiv2-v5/'
         ]


cross_sec_sigmas_full       = []
cross_sec_int_prob_full     = []
cross_sec_int_prob_sup_full = []

masses_int_prob_sup_full = []

for folder in folders:
    pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice([folder])
    
    if (os.path.exists(folder + 'cross_sec_sigmas_drate.txt') &
        os.path.exists(folder + 'cross_sec_int_prob_drate.txt') &
        os.path.exists(folder + 'cross_sec_int_prob_sup_drate.txt') &
        os.path.exists(folder + 'masses_int_prob_sup_drate.txt')
       ) == False or force == True:
        # Let's normalize testset between 0 and 1
        
        pars_norm = (pars_slices - pars_min) / (pars_max - pars_min)
        x_drate = diff_rate_slices
        x_norm_drate = (x_drate - x_min_drate) / (x_max_drate - x_min_drate)
        
        cross_sec_sigmas = np.ones((len(pars_slices), 6))
    
        cross_sec_int_prob     = np.ones(len(pars_norm)) * -99
        cross_sec_int_prob_sup = np.ones(len(pars_norm)) * -99
        masses_int_prob_sup    = np.ones(len(pars_norm)) * -99
           
        for itest in tqdm(range(len(pars_norm))):
            x_obs = x_norm_drate[itest, :]
            
            # We have to put this "observation" into a swyft.Sample object
            obs = swyft.Sample(x = x_obs)
            
            # Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
            pars_prior    = np.random.uniform(low = 0, high = 1, size = (10_000, 3))
            prior_samples = swyft.Samples(z = pars_prior)
            
            # Finally we make the inference
            predictions_drate = trainer_drate.infer(network_drate, obs, prior_samples)
        
            bins = 50
            logratios_drate = predictions_drate[0].logratios[:,1]
            v              = predictions_drate[0].params[:,1,0]
            low, upp = v.min(), v.max()
            weights  = torch.exp(logratios_drate) / torch.exp(logratios_drate).mean(axis = 0)
            h1       = torchist.histogramdd(predictions_drate[0].params[:,1,:], bins, weights = weights, low=low, upp=upp)
            h1      /= len(predictions_drate[0].params[:,1,:]) * (upp - low) / bins
            h1       = np.array(h1)
            
            edges = torch.linspace(v.min(), v.max(), bins + 1)
            x     = np.array((edges[1:] + edges[:-1]) / 2) * (pars_max[1] - pars_min[1]) + pars_min[1]
        
            vals = sorted(swyft.plot.plot2.get_HDI_thresholds(h1, cred_level=[0.68268, 0.95450, 0.99730]))
            try:
                cross_sec_sigmas[itest,0] = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
                cross_sec_sigmas[itest,3] = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
                
                cross_sec_sigmas[itest,1] = np.min(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
                cross_sec_sigmas[itest,4] = np.max(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
                
                cross_sec_sigmas[itest,2] = np.min(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
                cross_sec_sigmas[itest,5] = np.max(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
            except:
                pass
            cr_th = np.argmin(np.abs(x - (-45)))
            cross_sec_int_prob[itest]     = trapezoid(h1[:cr_th], x[:cr_th]) / trapezoid(h1, x)
            cross_sec_int_prob_sup[itest] = trapezoid(h1[cr_th:], x[cr_th:]) / trapezoid(h1, x)

            ratios_drate = np.exp(np.asarray(predictions_drate[0].logratios[:,0]))
            masses_pred = np.asarray(predictions_drate[0].params[:,0,0]) * (pars_max[0] - pars_min[0]) + pars_min[0]
            ind_sort    = np.argsort(masses_pred)
            ratios_drate = ratios_drate[ind_sort]
            masses_pred = masses_pred[ind_sort]
            m_min = np.argmin(np.abs(masses_pred - 1))
            m_max = np.argmin(np.abs(masses_pred - 2.6))
            masses_int_prob_sup[itest] = trapezoid(ratios_drate[m_min:m_max], masses_pred[m_min:m_max]) / trapezoid(ratios_drate, masses_pred)

        cross_sec_sigmas_full.append(cross_sec_sigmas)
        cross_sec_int_prob_full.append(cross_sec_int_prob)
        cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
        masses_int_prob_sup_full.append(masses_int_prob_sup)
            
        np.savetxt(folder + 'cross_sec_sigmas_drate.txt', cross_sec_sigmas)
        np.savetxt(folder + 'cross_sec_int_prob_drate.txt', cross_sec_int_prob)
        np.savetxt(folder + 'cross_sec_int_prob_sup_drate.txt', cross_sec_int_prob_sup)
        np.savetxt(folder + 'masses_int_prob_sup_drate.txt', masses_int_prob_sup)
    else:
        print('pre-computed')
                
        cross_sec_sigmas = np.loadtxt(folder + 'cross_sec_sigmas_drate.txt')
        cross_sec_int_prob = np.loadtxt(folder + 'cross_sec_int_prob_drate.txt')
        cross_sec_int_prob_sup = np.loadtxt(folder + 'cross_sec_int_prob_sup_drate.txt')
        masses_int_prob_sup = np.loadtxt(folder + 'masses_int_prob_sup_drate.txt')
        
        cross_sec_sigmas_full.append(cross_sec_sigmas)
        cross_sec_int_prob_full.append(cross_sec_int_prob)
        cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
        masses_int_prob_sup_full.append(masses_int_prob_sup)
#email('termino el analsisi slice drate O11')

# +
cross_section_th = -45

if len(cross_sec_int_prob_full) > 1:
    cross_sec_int_prob_drate_mpi_2        = np.mean(np.asarray(cross_sec_int_prob_full), axis = 0)
    cross_sec_int_prob_sup_drate_mpi_2    = np.mean(np.asarray(cross_sec_int_prob_sup_full), axis = 0)
    cross_sec_int_prob_sup_drate_mpi_2_sd = np.std(np.asarray(cross_sec_int_prob_sup_full), axis = 0)
    masses_int_prob_sup_drate_mpi_2       = np.mean(np.asarray(masses_int_prob_sup_full), axis = 0)
    masses_int_prob_sup_drate_mpi_2_sd    = np.std(np.asarray(masses_int_prob_sup_full), axis = 0)
    cross_sec_sigmas = np.mean(np.asarray(cross_sec_sigmas_full), axis = 0)
else:
    cross_sec_int_prob_drate_mpi_2     = cross_sec_int_prob
    cross_sec_int_prob_sup_drate_mpi_2 = cross_sec_int_prob_sup
    masses_int_prob_sup_drate_mpi_2    = masses_int_prob_sup

rate_1sigma_mpi_2 = np.ones(900) * -99
rate_2sigma_mpi_2 = np.ones(900) * -99
rate_3sigma_mpi_2 = np.ones(900) * -99

rate_1sigma_mpi_2[np.where(cross_sec_sigmas[:,0] > cross_section_th)[0]] = 1
rate_2sigma_mpi_2[np.where(cross_sec_sigmas[:,1] > cross_section_th)[0]] = 1
rate_3sigma_mpi_2[np.where(cross_sec_sigmas[:,2] > cross_section_th)[0]] = 1

# +
fig, ax = plt.subplots(1,2)

sbn.kdeplot(cross_sec_int_prob_sup_drate_0, label = '$\\theta = 0$', ax = ax[0])
sbn.kdeplot(cross_sec_int_prob_sup_drate_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[0])
sbn.kdeplot(cross_sec_int_prob_sup_drate_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[0])
sbn.kdeplot(cross_sec_int_prob_sup_drate_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[0])
sbn.kdeplot(cross_sec_int_prob_sup_drate_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[0])
ax[0].legend()
ax[0].set_xlabel('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')
ax[0].set_title('Diff. Rate')

sbn.kdeplot(masses_int_prob_sup_drate_0, label = '$\\theta = 0$', ax = ax[1])
sbn.kdeplot(masses_int_prob_sup_drate_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[1])
sbn.kdeplot(masses_int_prob_sup_drate_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[1])
sbn.kdeplot(masses_int_prob_sup_drate_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[1])
sbn.kdeplot(masses_int_prob_sup_drate_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[1])
ax[1].legend()
ax[1].set_xlabel('$\int_{m_{min}}^{m_{max}} P(m_{DM}|x)$')
ax[1].set_title('Diff. Rate')

#plt.savefig('../graph/O11_int_prob_distribution_drate.pdf')

# +
sigma = 2.1 # this depends on how noisy your data is, play with it!

CR_int_prob_sup_0_drate          = gaussian_filter(cross_sec_int_prob_sup_drate_0, sigma)
CR_int_prob_sup_0_drate_max      = gaussian_filter(cross_sec_int_prob_sup_drate_0 + cross_sec_int_prob_sup_drate_0_sd, sigma)
CR_int_prob_sup_0_drate_min      = gaussian_filter(cross_sec_int_prob_sup_drate_0 - cross_sec_int_prob_sup_drate_0_sd, sigma)
CR_int_prob_sup_pi_2_drate       = gaussian_filter(cross_sec_int_prob_sup_drate_pi_2, sigma)
CR_int_prob_sup_pi_2_drate_max   = gaussian_filter(cross_sec_int_prob_sup_drate_pi_2 + cross_sec_int_prob_sup_drate_pi_2_sd, sigma)
CR_int_prob_sup_pi_2_drate_min   = gaussian_filter(cross_sec_int_prob_sup_drate_pi_2 - cross_sec_int_prob_sup_drate_pi_2_sd, sigma)
CR_int_prob_sup_pi_4_drate       = gaussian_filter(cross_sec_int_prob_sup_drate_pi_4, sigma)
CR_int_prob_sup_pi_4_drate_max   = gaussian_filter(cross_sec_int_prob_sup_drate_pi_4 + cross_sec_int_prob_sup_drate_pi_4_sd, sigma)
CR_int_prob_sup_pi_4_drate_min   = gaussian_filter(cross_sec_int_prob_sup_drate_pi_4 - cross_sec_int_prob_sup_drate_pi_4_sd, sigma)
CR_int_prob_sup_mpi_2_drate      = gaussian_filter(cross_sec_int_prob_sup_drate_mpi_2, sigma)
CR_int_prob_sup_mpi_2_drate_max  = gaussian_filter(cross_sec_int_prob_sup_drate_mpi_2 + cross_sec_int_prob_sup_drate_mpi_2_sd, sigma)
CR_int_prob_sup_mpi_2_drate_min  = gaussian_filter(cross_sec_int_prob_sup_drate_mpi_2 - cross_sec_int_prob_sup_drate_mpi_2_sd, sigma)
CR_int_prob_sup_mpi_4_drate      = gaussian_filter(cross_sec_int_prob_sup_drate_mpi_4, sigma)
CR_int_prob_sup_mpi_4_drate_max  = gaussian_filter(cross_sec_int_prob_sup_drate_mpi_4 + cross_sec_int_prob_sup_drate_mpi_4_sd, sigma)
CR_int_prob_sup_mpi_4_drate_min  = gaussian_filter(cross_sec_int_prob_sup_drate_mpi_4 - cross_sec_int_prob_sup_drate_mpi_4_sd, sigma)

M_int_prob_sup_0_drate         = gaussian_filter(masses_int_prob_sup_drate_0, sigma)
M_int_prob_sup_0_drate_max     = gaussian_filter(masses_int_prob_sup_drate_0 + masses_int_prob_sup_drate_0_sd, sigma)
M_int_prob_sup_0_drate_min     = gaussian_filter(masses_int_prob_sup_drate_0 - masses_int_prob_sup_drate_0_sd, sigma)
M_int_prob_sup_pi_2_drate      = gaussian_filter(masses_int_prob_sup_drate_pi_2, sigma)
M_int_prob_sup_pi_2_drate_max  = gaussian_filter(masses_int_prob_sup_drate_pi_2 + masses_int_prob_sup_drate_pi_2_sd, sigma)
M_int_prob_sup_pi_2_drate_min  = gaussian_filter(masses_int_prob_sup_drate_pi_2 - masses_int_prob_sup_drate_pi_2_sd, sigma)
M_int_prob_sup_pi_4_drate      = gaussian_filter(masses_int_prob_sup_drate_pi_4, sigma)
M_int_prob_sup_pi_4_drate_max  = gaussian_filter(masses_int_prob_sup_drate_pi_4 + masses_int_prob_sup_drate_pi_4_sd, sigma)
M_int_prob_sup_pi_4_drate_min  = gaussian_filter(masses_int_prob_sup_drate_pi_4 - masses_int_prob_sup_drate_pi_4_sd, sigma)
M_int_prob_sup_mpi_2_drate     = gaussian_filter(masses_int_prob_sup_drate_mpi_2, sigma)
M_int_prob_sup_mpi_2_drate_max = gaussian_filter(masses_int_prob_sup_drate_mpi_2 + masses_int_prob_sup_drate_mpi_2_sd, sigma)
M_int_prob_sup_mpi_2_drate_min = gaussian_filter(masses_int_prob_sup_drate_mpi_2 - masses_int_prob_sup_drate_mpi_2_sd, sigma)
M_int_prob_sup_mpi_4_drate     = gaussian_filter(masses_int_prob_sup_drate_mpi_4, sigma)
M_int_prob_sup_mpi_4_drate_max = gaussian_filter(masses_int_prob_sup_drate_mpi_4 + masses_int_prob_sup_drate_mpi_4_sd, sigma)
M_int_prob_sup_mpi_4_drate_min = gaussian_filter(masses_int_prob_sup_drate_mpi_4 - masses_int_prob_sup_drate_mpi_4_sd, sigma)

# +
levels = [0.5, 0.76, 0.84, 0.9, 1]

fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))
fig.subplots_adjust(hspace = 0, wspace = 0)

fig00 = ax[0,0].contourf(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)
#ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate_max.reshape(30,30).T, levels=[0.9], linewidths = 1, zorder = 4)
#ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate_min.reshape(30,30).T, levels=[0.9], linewidths = 1, zorder = 4)

ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-41, '$\\theta = \pi/2$')

ax[0,1].contourf(m_vals, cross_vals, CR_int_prob_sup_pi_4_drate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_drate.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)

ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-41, '$\\theta = \pi/4$')

ax[1,0].contourf(m_vals, cross_vals, CR_int_prob_sup_mpi_2_drate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, CR_int_prob_sup_mpi_2_drate.reshape(30,30).T, levels=levels)

ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-41, '$\\theta = -\pi/2$')

ax[1,1].contourf(m_vals, cross_vals, CR_int_prob_sup_0_drate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_drate.reshape(30,30).T, levels=levels)

ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-41, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(1e-46, 4e-41)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(fig00, cax=cbar_ax)
cbar.ax.set_title('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')

ax[0,0].plot(masses, s1s2_90_CL_pi2[2,:], color = 'black', linestyle = '-.', label = 'Bin. Lik 90%')
ax[0,1].plot(masses, s1s2_90_CL_pi4[2,:], color = 'black', linestyle = '-.')
ax[1,0].plot(masses, s1s2_90_CL_mpi2[2,:], color = 'black', linestyle = '-.')
ax[1,1].plot(masses, s1s2_90_CL_0[2,:], color = 'black', linestyle = '-.')
ax[0,0].legend(loc = 'lower left')

plt.savefig('../graph/O11_contours_drate_int_prob_sup.pdf')
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
torch.manual_seed(28891)
cb = MetricTracker()
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta = 0., patience=25, verbose=False, mode='min')
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O11_s1s2_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
trainer_s1s2 = swyft.SwyftTrainer(accelerator = device, devices=1, max_epochs = 2500, precision = 64, callbacks=[early_stopping_callback, checkpoint_callback, cb])
network_s1s2 = Network()

# +
x_norm_test_s1s2 = s1s2_testset[:,:-1,:-1] # Observable. Input data. I am cutting a bit the images to have 96x96
x_norm_test_s1s2 = x_norm_test_s1s2.reshape(len(x_norm_test_s1s2), 1, 96, 96)
pars_norm_test = (pars_testset - pars_min) / (pars_max - pars_min)

# We have to build a swyft.Samples object that will handle the data
samples_test_s1s2 = swyft.Samples(x = x_norm_test_s1s2, z = pars_norm_test)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_test_s1s2 = swyft.SwyftDataModule(samples_test_s1s2, fractions = [0., 0., 1], batch_size = 32)
trainer_s1s2.test(network_s1s2, dm_test_s1s2)

# +
fit = True
if fit:
    trainer_s1s2.fit(network_s1s2, dm_s1s2)
    checkpoint_callback.to_yaml("./logs/O11_s1s2.yaml") 
    ckpt_path = swyft.best_from_yaml("./logs/O11_s1s2.yaml")
    email('Termino entrenamiento O11 s1s2')
else:
    ckpt_path = swyft.best_from_yaml("./logs/O11_s1s2.yaml")

# ---------------------------------------
# Min val loss value at  epochs. 
# ---------------------------------------


# +
x_norm_test_s1s2 = s1s2_testset[:,:-1,:-1] # Observable. Input data. I am cutting a bit the images to have 96x96
x_norm_test_s1s2 = x_norm_test_s1s2.reshape(len(x_norm_test_s1s2), 1, 96, 96)
pars_norm_test = (pars_testset - pars_min) / (pars_max - pars_min)

# We have to build a swyft.Samples object that will handle the data
samples_test_s1s2 = swyft.Samples(x = x_norm_test_s1s2, z = pars_norm_test)

# We have to build a swyft.SwyftDataModule object that will split the data into training, testing and validation sets
dm_test_s1s2 = swyft.SwyftDataModule(samples_test_s1s2, fractions = [0., 0., 1], batch_size = 32)
trainer_s1s2.test(network_s1s2, dm_test_s1s2, ckpt_path = ckpt_path)

# ---------------------------------------
# Min val loss value -1.42  @ testset
# ---------------------------------------

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
    plt.savefig('../graph/O11_loss_s1s2.pdf')

# ### Let's make some inference (NOT IMPLEMENTED)

# +
# Let's normalize testset between 0 and 1

pars_norm = (pars_testset - pars_min) / (pars_max - pars_min)

x_norm_s1s2 = x_s1s2 = s1s2_testset[:,:-1,:-1]

# +
# First let's create some observation from some "true" theta parameters
#i = 189 #np.random.randint(ntest) # 189 (disc) 455 (exc) 203 (middle)
print(i)

pars_true = pars_norm[i,:]
x_obs     = x_norm_s1s2[i,:].reshape(1,96,96)

plt.imshow(x_obs[0].T, origin = 'lower')

# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
pars_prior = np.random.uniform(low = 0, high = 1, size = (100_000, 3))

prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions_s1s2 = trainer_s1s2.infer(network_s1s2, obs, prior_samples)

# +
# Let's plot the results
swyft.corner(predictions_s1s2, ('pars_norm[0]', 'pars_norm[1]', 'pars_norm[2]'), bins = 200, smooth = 3)

if flag == 'exc':
    plt.savefig('../graph/O4_cornerplot_s1s2_exc.pdf')
else:
    plt.savefig('../graph/O4_cornerplot_s1s2.pdf')

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

if low_1sigma > cross_section_th: print('Distinguish at 1 $\sigma$')
if low_2sigma > cross_section_th: print('Distinguish at 2 $\sigma$')
if low_3sigma > cross_section_th: print('Distinguish at 3 $\sigma$')

# +
plt.plot(x, h1, c = 'black')
plt.axvline(x = pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1], c = 'orange')

if low_1sigma > cross_section_th: plt.axvline(low_1sigma, c = 'black')
if up_1sigma > cross_section_th: plt.axvline(up_1sigma, c = 'black')

if low_2sigma > cross_section_th: plt.axvline(low_2sigma, c = 'black', linestyle = '--')
if up_2sigma > cross_section_th: plt.axvline(up_2sigma, c = 'black', linestyle = '--')

if low_3sigma > cross_section_th: plt.axvline(low_3sigma, c = 'black', linestyle = ':')
if up_3sigma > cross_section_th: plt.axvline(up_3sigma, c = 'black', linestyle = ':')
#plt.ylim(0,4.5)
#plt.xscale('log')
plt.xlabel('$log(\sigma)$')
plt.ylabel('$P(\sigma|x)$')
plt.text(-43,2, '$m = {:.2e}$'.format(10**(pars_true[0])))
plt.text(-43,1.8, '$\sigma = {:.2e}$'.format(10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1])))
plt.text(-43,1.5, '$\\theta = {:.2f}$'.format(pars_true[0]))
if flag == 'exc':
    plt.savefig('../graph/O4_1Dposterior_s1s2_exc_' + str(i) + '.pdf')
else:
    plt.savefig('../graph/O4_1Dposterior_s1s2_disc_' + str(i) + '.pdf')
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
    plt.savefig('../graph/O4_loglikratio_s1s2_exc.pdf')
else:
    plt.savefig('../graph/O4_loglikratio_s1s2.pdf')
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

val, xaux, yaux,_ = stats.binned_statistic_2d(m_results, sigma_results, results_s1s2[:,0], 'max', bins = [np.logspace(0.81, 3, 15), np.logspace(-43.2, -35, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im20 = ax[0].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im20, ax = ax[0])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,0]), 10**(pars[:,1]), np.log10(rate + 7), 'min', bins = [np.logspace(0.81, 3, 10), np.logspace(-43.2, -35, 10)])
    
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

val, xaux, yaux,_ = stats.binned_statistic_2d(sigma_results, theta_results, results_s1s2[:,2], 'max', bins = [np.logspace(-43.2, -35, 15), np.linspace(-1.6, 1.6, 15)])
    
xbin = xaux[1] - xaux[0]
x_centers = xaux[:-1] + xbin

ybin = yaux[1] - yaux[0]
y_centers = yaux[:-1] + ybin

im22 = ax[2].contourf(x_centers, y_centers, val.T, alpha = 0.6, levels = [-100, -10, -5, -2, 0, 2, 5, 10, 100], colors = pallete)
clb = plt.colorbar(im22, ax = ax[2])
clb.ax.set_title('$\lambda$')

val, xaux, yaux,_ = stats.binned_statistic_2d(10**(pars[:,1]), pars[:,2], np.log10(rate + 7), 'min', bins = [np.logspace(-43.2, -35, 10), np.linspace(-1.6, 1.6, 10)])
    
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
    plt.savefig('../graph/O4_pars_s1s2_exc.pdf')
else:
    plt.savefig('../graph/O4_pars_s1s2.pdf')
# -
# ### Let's make the contour plot ($\sigma$)

# !ls ../data/andresData/O11-full/O11

pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice(['../data/andresData/O11-full/O11/theta-pluspidiv2/O11-slices01-pluspidiv2/'])

# +
#'../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0/'
# -

m_vals = np.logspace(np.min(pars_slices[:,0]), np.max(pars_slices[:,0]),30)
cross_vals = np.logspace(np.min(pars_slices[:,1]), np.max(pars_slices[:,1]),30)

# +
force = True
folders = [#'../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0/',
           '../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0-v2/',
           '../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0-v3/',
           '../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0-v4/',
           '../data/andresData/O11-full/O11/theta-0/O11-slices01-theta0-v5/'
         ]

cross_sec_sigmas_full       = []
cross_sec_int_prob_full     = []
cross_sec_int_prob_sup_full = []

masses_int_prob_sup_full = []

for folder in folders:
    pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice([folder])
 
    if (os.path.exists(folder + 'cross_sec_sigmas_s1s2.txt') & 
        os.path.exists(folder + 'cross_sec_int_prob_s1s2.txt') &
        os.path.exists(folder + 'cross_sec_int_prob_sup_s1s2.txt') &
        os.path.exists(folder + 'masses_int_prob_sup_s1s2.txt')
       ) == False or force == True:
        # Let's normalize testset between 0 and 1
        
        pars_norm = (pars_slices - pars_min) / (pars_max - pars_min)
        
        x_norm_s1s2 = x_s1s2 = s1s2_slices[:,:-1,:-1]
        
        res_1sigma = np.ones(len(pars_norm)) * -99
        res_2sigma = np.ones(len(pars_norm)) * -99
        res_3sigma = np.ones(len(pars_norm)) * -99
        
        cross_sec_sigmas = np.ones((len(pars_slices), 6))
    
        cross_sec_int_prob = np.ones(len(pars_norm)) * -99
        cross_sec_int_prob_sup = np.ones(len(pars_norm)) * -99
        masses_int_prob_sup = np.ones(len(pars_norm)) * -99
           
        for itest in tqdm(range(len(pars_norm))):
            x_obs = x_norm_s1s2[itest, :,:]
            
            # We have to put this "observation" into a swyft.Sample object
            obs = swyft.Sample(x = x_obs.reshape(1,96,96))
            
            # Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
            pars_prior    = np.random.uniform(low = 0, high = 1, size = (100_000, 3))
            prior_samples = swyft.Samples(z = pars_prior)
            
            # Finally we make the inference
            predictions_s1s2 = trainer_s1s2.infer(network_s1s2, obs, prior_samples)
        
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
        
            vals = sorted(swyft.plot.plot2.get_HDI_thresholds(h1, cred_level=[0.68268, 0.95450, 0.99730]))
            
            cross_sec_sigmas[itest,0] = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
            cross_sec_sigmas[itest,3] = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
            
            cross_sec_sigmas[itest,1] = np.min(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
            cross_sec_sigmas[itest,4] = np.max(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
            
            cross_sec_sigmas[itest,2] = np.min(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
            cross_sec_sigmas[itest,5] = np.max(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
            
            cr_th  = np.argmin(np.abs(x - (-45)))
            cross_sec_int_prob[itest]     = trapezoid(h1[:cr_th],x[:cr_th]) / trapezoid(h1,x)
            cross_sec_int_prob_sup[itest] = trapezoid(h1[cr_th:],x[cr_th:]) / trapezoid(h1,x)

            ratios_s1s2 = np.exp(np.asarray(predictions_s1s2[0].logratios[:,0]))
            masses_pred = np.asarray(predictions_s1s2[0].params[:,0,0]) * (pars_max[0] - pars_min[0]) + pars_min[0]           
            ind_sort    = np.argsort(masses_pred)
            ratios_s1s2 = ratios_s1s2[ind_sort]
            masses_pred = masses_pred[ind_sort]
            m_min = np.argmin(np.abs(masses_pred - 1))
            m_max = np.argmin(np.abs(masses_pred - 2.6))
            masses_int_prob_sup[itest] = trapezoid(ratios_s1s2[m_min:m_max], masses_pred[m_min:m_max]) / trapezoid(ratios_s1s2, masses_pred)

        cross_sec_sigmas_full.append(cross_sec_sigmas)
        cross_sec_int_prob_full.append(cross_sec_int_prob)
        cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
        masses_int_prob_sup_full.append(masses_int_prob_sup)
            
        np.savetxt(folder + 'cross_sec_sigmas_s1s2.txt', cross_sec_sigmas)
        np.savetxt(folder + 'cross_sec_int_prob_s1s2.txt', cross_sec_int_prob)
        np.savetxt(folder + 'cross_sec_int_prob_sup_s1s2.txt', cross_sec_int_prob_sup)
        np.savetxt(folder + 'masses_int_prob_sup_s1s2.txt', masses_int_prob_sup)
    else:
        print('pre-computed')
        cross_sec_sigmas = np.loadtxt(folder + 'cross_sec_sigmas_s1s2.txt')
        cross_sec_int_prob = np.loadtxt(folder + 'cross_sec_int_prob_s1s2.txt')
        cross_sec_int_prob_sup = np.loadtxt(folder + 'cross_sec_int_prob_sup_s1s2.txt')
        masses_int_prob_sup = np.loadtxt(folder + 'masses_int_prob_sup_s1s2.txt')

        cross_sec_sigmas_full.append(cross_sec_sigmas)
        cross_sec_int_prob_full.append(cross_sec_int_prob)
        cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
        masses_int_prob_sup_full.append(masses_int_prob_sup)


# +
cross_section_th = -45

if len(cross_sec_int_prob_full) > 1:
    cross_sec_int_prob_s1s2_0        = np.mean(np.asarray(cross_sec_int_prob_full), axis = 0)
    cross_sec_int_prob_sup_s1s2_0    = np.mean(np.asarray(cross_sec_int_prob_sup_full), axis = 0)
    cross_sec_int_prob_sup_s1s2_0_sd = np.std(np.asarray(cross_sec_int_prob_sup_full), axis = 0)
    masses_int_prob_sup_s1s2_0       = np.mean(np.asarray(masses_int_prob_sup_full), axis = 0)
    masses_int_prob_sup_s1s2_0_sd    = np.std(np.asarray(masses_int_prob_sup_full), axis = 0)
    cross_sec_sigmas                     = np.mean(np.asarray(cross_sec_sigmas_full), axis = 0)
else:
    cross_sec_int_prob_s1s2_0     = cross_sec_int_prob
    cross_sec_int_prob_sup_s1s2_0 = cross_sec_int_prob_sup
    masses_int_prob_sup_s1s2_0    = masses_int_prob_sup

s1s2_1sigma_0 = np.ones(900) * -99
s1s2_2sigma_0 = np.ones(900) * -99
s1s2_3sigma_0 = np.ones(900) * -99

s1s2_1sigma_0[np.where(cross_sec_sigmas[:,0] > cross_section_th)[0]] = 1
s1s2_2sigma_0[np.where(cross_sec_sigmas[:,1] > cross_section_th)[0]] = 1
s1s2_3sigma_0[np.where(cross_sec_sigmas[:,2] > cross_section_th)[0]] = 1
# -

email()

# +
fig, ax = plt.subplots(1,2)

sbn.kdeplot(cross_sec_int_prob_sup_s1s2_0, label = '$\\theta = 0$', ax = ax[0])
sbn.kdeplot(cross_sec_int_prob_sup_s1s2_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[0])
sbn.kdeplot(cross_sec_int_prob_sup_s1s2_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[0])
sbn.kdeplot(cross_sec_int_prob_sup_s1s2_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[0])
sbn.kdeplot(cross_sec_int_prob_sup_s1s2_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[0])
ax[0].legend()
ax[0].set_xlabel('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')
ax[0].set_title('S1-S2')

sbn.kdeplot(masses_int_prob_sup_s1s2_0, label = '$\\theta = 0$', ax = ax[1])
sbn.kdeplot(masses_int_prob_sup_s1s2_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[1])
sbn.kdeplot(masses_int_prob_sup_s1s2_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[1])
sbn.kdeplot(masses_int_prob_sup_s1s2_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[1])
sbn.kdeplot(masses_int_prob_sup_s1s2_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[1])
ax[1].legend()
ax[1].set_xlabel('$\int_{m_{min}}^{m_{max}} P(m_{DM}|x)$')
ax[1].set_title('S1-S2')

plt.savefig('../graph/O11_int_prob_distribution_s1s2.pdf')

# +
sigma = 0.2 # this depends on how noisy your data is, play with it!

s1s2_1sigma_0_g = gaussian_filter(s1s2_1sigma_0, sigma)
s1s2_1sigma_pi_2_g = gaussian_filter(s1s2_1sigma_pi_2, sigma)
s1s2_1sigma_pi_4_g = gaussian_filter(s1s2_1sigma_pi_4, sigma)
s1s2_1sigma_mpi_2_g = gaussian_filter(s1s2_1sigma_mpi_2, sigma)
s1s2_1sigma_mpi_4_g = gaussian_filter(s1s2_1sigma_mpi_4, sigma)

s1s2_2sigma_0_g = gaussian_filter(s1s2_2sigma_0, sigma)
s1s2_2sigma_pi_2_g = gaussian_filter(s1s2_2sigma_pi_2, sigma)
s1s2_2sigma_pi_4_g = gaussian_filter(s1s2_2sigma_pi_4, sigma)
s1s2_2sigma_mpi_2_g = gaussian_filter(s1s2_2sigma_mpi_2, sigma)
s1s2_2sigma_mpi_4_g = gaussian_filter(s1s2_2sigma_mpi_4, sigma)

s1s2_3sigma_0_g = gaussian_filter(s1s2_3sigma_0, sigma)
s1s2_3sigma_pi_2_g = gaussian_filter(s1s2_3sigma_pi_2, sigma)
s1s2_3sigma_pi_4_g = gaussian_filter(s1s2_3sigma_pi_4, sigma)
s1s2_3sigma_mpi_2_g = gaussian_filter(s1s2_3sigma_mpi_2, sigma)
s1s2_3sigma_mpi_4_g = gaussian_filter(s1s2_3sigma_mpi_4, sigma)

fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))
fig.subplots_adjust(hspace = 0, wspace = 0)

ax[0,0].contour(m_vals, cross_vals, s1s2_1sigma_pi_2_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[0,0].contour(m_vals, cross_vals, s1s2_2sigma_pi_2_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[0,0].contourf(m_vals, cross_vals, s1s2_3sigma_pi_2_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, s1s2_3sigma_pi_2_g.reshape(30,30).T, levels=[0])

ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-41, '$\\theta = \pi/2$')
#ax[0,0].legend(loc = 'lower right')

ax[0,1].contour(m_vals, cross_vals, s1s2_1sigma_pi_4_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[0,1].contour(m_vals, cross_vals, s1s2_2sigma_pi_4_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[0,1].contourf(m_vals, cross_vals, s1s2_3sigma_pi_4_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, s1s2_3sigma_pi_4_g.reshape(30,30).T, levels=[0])

ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-41, '$\\theta = \pi/4$')
ax[0,1].legend(loc = 'lower right')

#ax[1,0].contour(m_vals, cross_vals, int_prob_0.reshape(30,30).T, levels=10, linewidths = 2, zorder = 4, linestyles = '--')
ax[1,0].contour(m_vals, cross_vals, s1s2_1sigma_mpi_2_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[1,0].contour(m_vals, cross_vals, s1s2_2sigma_mpi_2_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[1,0].contourf(m_vals, cross_vals, s1s2_3sigma_mpi_2_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, s1s2_3sigma_mpi_2_g.reshape(30,30).T, levels=[0])

ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-41, '$\\theta = -\pi/2$')

ax[1,1].contour(m_vals, cross_vals, s1s2_1sigma_0_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[1,1].contour(m_vals, cross_vals, s1s2_2sigma_0_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[1,1].contourf(m_vals, cross_vals, s1s2_3sigma_0_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, s1s2_3sigma_0_g.reshape(30,30).T, levels=[0])

ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-41, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(1e-46, 4e-41)

plt.savefig('../graph/O11_contours_s1s2.pdf')


# +
levels = 5#[0,0.1,0.16,0.24,0.32,0.7]

sigma = 1.41 # this depends on how noisy your data is, play with it!

int_prob_0_g     = gaussian_filter(cross_sec_int_prob_s1s2_0, sigma)
int_prob_pi_2_g  = gaussian_filter(cross_sec_int_prob_s1s2_pi_2, sigma)
int_prob_pi_4_g  = gaussian_filter(cross_sec_int_prob_s1s2_pi_4, sigma)
int_prob_mpi_2_g = gaussian_filter(cross_sec_int_prob_s1s2_mpi_2, sigma)
int_prob_mpi_4_g = gaussian_filter(cross_sec_int_prob_s1s2_mpi_4, sigma)

fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))
fig.subplots_adjust(hspace = 0, wspace = 0)

fig00 = ax[0,0].contourf(m_vals, cross_vals, int_prob_pi_2_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, int_prob_pi_2_g.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)

ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-41, '$\\theta = \pi/2$')
ax[0,0].plot(masses, s1s2_90_CL_pi2[2,:], color = 'black', linestyle = '-.', label = 'Bin. Lik. [90%]')
ax[0,0].legend(loc = 'lower left')

ax[0,1].contourf(m_vals, cross_vals, int_prob_pi_4_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, int_prob_pi_4_g.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)

ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-41, '$\\theta = \pi/4$')

ax[1,0].contourf(m_vals, cross_vals, int_prob_mpi_2_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, int_prob_mpi_2_g.reshape(30,30).T, levels=levels)

ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-41, '$\\theta = -\pi/2$')

ax[1,1].contourf(m_vals, cross_vals, int_prob_0_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, int_prob_0_g.reshape(30,30).T, levels=levels)

ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-41, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(1e-46, 4e-41)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(fig00, cax=cbar_ax)
cbar.ax.set_title('$\int_{-\inf}^{\sigma_{th}} P(\sigma|x)$')

ax[0,1].plot(masses, s1s2_90_CL_pi4[2,:], color = 'black', linestyle = '-.')
ax[1,0].plot(masses, s1s2_90_CL_mpi2[2,:], color = 'black', linestyle = '-.')
ax[1,1].plot(masses, s1s2_90_CL_0[2,:], color = 'black', linestyle = '-.')

plt.savefig('../graph/O11_contours_s1s1_int_prob.pdf')

# +
sigma = 2.1 # this depends on how noisy your data is, play with it!

CR_int_prob_sup_0_s1s2          = gaussian_filter(cross_sec_int_prob_sup_s1s2_0, sigma)
CR_int_prob_sup_0_s1s2_max      = gaussian_filter(cross_sec_int_prob_sup_s1s2_0 + cross_sec_int_prob_sup_s1s2_0_sd, sigma)
CR_int_prob_sup_0_s1s2_min      = gaussian_filter(cross_sec_int_prob_sup_s1s2_0 - cross_sec_int_prob_sup_s1s2_0_sd, sigma)
CR_int_prob_sup_pi_2_s1s2       = gaussian_filter(cross_sec_int_prob_sup_s1s2_pi_2, sigma)
CR_int_prob_sup_pi_2_s1s2_max   = gaussian_filter(cross_sec_int_prob_sup_s1s2_pi_2 + cross_sec_int_prob_sup_s1s2_pi_2_sd, sigma)
CR_int_prob_sup_pi_2_s1s2_min   = gaussian_filter(cross_sec_int_prob_sup_s1s2_pi_2 - cross_sec_int_prob_sup_s1s2_pi_2_sd, sigma)
CR_int_prob_sup_pi_4_s1s2       = gaussian_filter(cross_sec_int_prob_sup_s1s2_pi_4, sigma)
CR_int_prob_sup_pi_4_s1s2_max   = gaussian_filter(cross_sec_int_prob_sup_s1s2_pi_4 + cross_sec_int_prob_sup_s1s2_pi_4_sd, sigma)
CR_int_prob_sup_pi_4_s1s2_min   = gaussian_filter(cross_sec_int_prob_sup_s1s2_pi_4 - cross_sec_int_prob_sup_s1s2_pi_4_sd, sigma)
CR_int_prob_sup_mpi_2_s1s2      = gaussian_filter(cross_sec_int_prob_sup_s1s2_mpi_2, sigma)
CR_int_prob_sup_mpi_2_s1s2_max  = gaussian_filter(cross_sec_int_prob_sup_s1s2_mpi_2 + cross_sec_int_prob_sup_s1s2_mpi_2_sd, sigma)
CR_int_prob_sup_mpi_2_s1s2_min  = gaussian_filter(cross_sec_int_prob_sup_s1s2_mpi_2 - cross_sec_int_prob_sup_s1s2_mpi_2_sd, sigma)
CR_int_prob_sup_mpi_4_s1s2      = gaussian_filter(cross_sec_int_prob_sup_s1s2_mpi_4, sigma)
CR_int_prob_sup_mpi_4_s1s2_max  = gaussian_filter(cross_sec_int_prob_sup_s1s2_mpi_4 + cross_sec_int_prob_sup_s1s2_mpi_4_sd, sigma)
CR_int_prob_sup_mpi_4_s1s2_min  = gaussian_filter(cross_sec_int_prob_sup_s1s2_mpi_4 - cross_sec_int_prob_sup_s1s2_mpi_4_sd, sigma)

M_int_prob_sup_0_s1s2         = gaussian_filter(masses_int_prob_sup_s1s2_0, sigma)
M_int_prob_sup_0_s1s2_max     = gaussian_filter(masses_int_prob_sup_s1s2_0 + masses_int_prob_sup_s1s2_0_sd, sigma)
M_int_prob_sup_0_s1s2_min     = gaussian_filter(masses_int_prob_sup_s1s2_0 - masses_int_prob_sup_s1s2_0_sd, sigma)
M_int_prob_sup_pi_2_s1s2      = gaussian_filter(masses_int_prob_sup_s1s2_pi_2, sigma)
M_int_prob_sup_pi_2_s1s2_max  = gaussian_filter(masses_int_prob_sup_s1s2_pi_2 + masses_int_prob_sup_s1s2_pi_2_sd, sigma)
M_int_prob_sup_pi_2_s1s2_min  = gaussian_filter(masses_int_prob_sup_s1s2_pi_2 - masses_int_prob_sup_s1s2_pi_2_sd, sigma)
M_int_prob_sup_pi_4_s1s2      = gaussian_filter(masses_int_prob_sup_s1s2_pi_4, sigma)
M_int_prob_sup_pi_4_s1s2_max  = gaussian_filter(masses_int_prob_sup_s1s2_pi_4 + masses_int_prob_sup_s1s2_pi_4_sd, sigma)
M_int_prob_sup_pi_4_s1s2_min  = gaussian_filter(masses_int_prob_sup_s1s2_pi_4 - masses_int_prob_sup_s1s2_pi_4_sd, sigma)
M_int_prob_sup_mpi_2_s1s2     = gaussian_filter(masses_int_prob_sup_s1s2_mpi_2, sigma)
M_int_prob_sup_mpi_2_s1s2_max = gaussian_filter(masses_int_prob_sup_s1s2_mpi_2 + masses_int_prob_sup_s1s2_mpi_2_sd, sigma)
M_int_prob_sup_mpi_2_s1s2_min = gaussian_filter(masses_int_prob_sup_s1s2_mpi_2 - masses_int_prob_sup_s1s2_mpi_2_sd, sigma)
M_int_prob_sup_mpi_4_s1s2     = gaussian_filter(masses_int_prob_sup_s1s2_mpi_4, sigma)
M_int_prob_sup_mpi_4_s1s2_max = gaussian_filter(masses_int_prob_sup_s1s2_mpi_4 + masses_int_prob_sup_s1s2_mpi_4_sd, sigma)
M_int_prob_sup_mpi_4_s1s2_min = gaussian_filter(masses_int_prob_sup_s1s2_mpi_4 - masses_int_prob_sup_s1s2_mpi_4_sd, sigma)

# +
levels = [0.55, 0.66, 0.84, 0.9, 1]

fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))
fig.subplots_adjust(hspace = 0, wspace = 0)

fig00 = ax[0,0].contourf(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['magenta'])
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['purple'])

ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-41, '$\\theta = \pi/2$')
ax[0,0].plot(masses, s1s2_90_CL_pi2[2,:], color = 'black', linestyle = '-.', label = 'Bin. Lik. [90%]')
ax[0,0].legend(loc = 'lower left')

ax[0,1].contourf(m_vals, cross_vals, CR_int_prob_sup_pi_4_s1s2.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_s1s2.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)
ax[0,1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['magenta'])
ax[0,1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['purple'])

ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-41, '$\\theta = \pi/4$')
#ax[0,1].legend()

ax[1,0].contourf(m_vals, cross_vals, CR_int_prob_sup_mpi_2_s1s2.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, CR_int_prob_sup_mpi_2_s1s2.reshape(30,30).T, levels=levels)
ax[1,0].contour(m_vals, cross_vals, CR_int_prob_sup_mpi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['magenta'])
ax[1,0].contour(m_vals, cross_vals, CR_int_prob_sup_mpi_2_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['purple'])

ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-41, '$\\theta = -\pi/2$')

ax[1,1].contourf(m_vals, cross_vals, CR_int_prob_sup_0_s1s2.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_s1s2.reshape(30,30).T, levels=levels)
ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['magenta'])
ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['purple'])

ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-41, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(1e-46, 2e-41)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(fig00, cax=cbar_ax)
cbar.ax.set_title('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')

ax[0,1].plot(masses, s1s2_90_CL_pi4[2,:], color = 'black', linestyle = '-.')
ax[1,0].plot(masses, s1s2_90_CL_mpi2[2,:], color = 'black', linestyle = '-.')
ax[1,1].plot(masses, s1s2_90_CL_0[2,:], color = 'black', linestyle = '-.')

#plt.savefig('../graph/O11_contours_s1s2_int_prob_sup.pdf')
# +
fig, ax = plt.subplots(1,3, sharex = True, sharey = True, figsize = (12,5))
fig.subplots_adjust(hspace = 0, wspace = 0)

ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_s1s2)
ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_s1s2)
ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_s1s2)
ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_rate)
# # #%ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_rate)
# # #%ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_rate)
ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_drate)
# # #%ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_drate)
# # #%ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_drate)
ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_s1s2)
ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_s1s2_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_s1s2_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_rate)
# # #%ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_rate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
# # #%ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_rate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_drate)
# # #%ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_drate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
# # #%ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_drate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)

ax[0].plot(masses, s1s2_90_CL_pi2[2,:], color = 'black', linestyle = ':', label = 'Bin. Lik. [90%]')
ax[0].fill_between(masses, s1s2_current_pi2[2,:], 5e-36, color = 'black', alpha = 0.2, label = 'Excluded')

ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].grid(which='both')
ax[0].text(3e2, 1e-41, '$\\theta = \pi/2$')
ax[0].legend(loc = 'lower left')

ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_s1s2)
ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_s1s2_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_s1s2)
ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_s1s2_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_s1s2)
ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_rate)
# # #%ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_rate)
# # #%ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_rate)
ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_drate)
# # #%ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_drate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_drate)
# # #%ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_drate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_drate)
ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_s1s2)
ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_s1s2_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_s1s2_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_rate)
# # #%ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_rate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
# # #%ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_rate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_drate)
# # #%ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_drate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
# # #%ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_drate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)

ax[1].plot(masses, s1s2_90_CL_pi4[2,:], color = 'black', linestyle = ':')
ax[1].fill_between(masses, s1s2_current_pi4[2,:], 5e-36, color = 'black', alpha = 0.2)

ax[1].grid(which='both')
ax[1].text(3e2, 1e-41, '$\\theta = \pi/4$')

ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_s1s2)
ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_s1s2_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_s1s2)
ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_s1s2_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_s1s2)
ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_rate)
# # #%ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_rate)
# # #%ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_rate)
ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_drate)
# # #%ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_drate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_drate)
# # #%ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_drate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_drate)
ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_s1s2)
ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_s1s2_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_s1s2_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_rate)
# # #%ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_rate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
# # #%ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_rate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_drate)
# # #%ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_drate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
# # #%ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_drate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)

ax[2].plot(masses, s1s2_90_CL_0[2,:], color = 'black', linestyle = ':')
ax[2].fill_between(masses, s1s2_current_0[2,:], 5e-36, color = 'black', alpha = 0.2, label = 'Excluded')
ax[2].legend(loc = 'lower right')

ax[2].grid(which='both')
ax[2].text(3e2, 1e-41, '$\\theta = 0$')

ax[0].set_ylabel('$\sigma \ [cm^{2}]$')
ax[0].set_xlabel('m [GeV]')
ax[1].set_xlabel('m [GeV]')
ax[2].set_xlabel('m [GeV]')

ax[0].set_ylim(1e-46, 2e-41)
ax[0].set_xlim(6, 9.8e2)

fig.subplots_adjust(right=0.8)

custom_lines = []
labels = ['Total Rate', 'Dif. Rate', 'S1-S2']
markers = ['solid','solid', 'solid']
colors = [color_rate, color_drate, color_s1s2]
for i in range(3):
    custom_lines.append( Line2D([0],[0], linestyle = markers[i], color = colors[i], 
            label = labels[i]) )
    
ax[1].legend(handles = custom_lines, loc = 'lower left')

custom_lines = []
labels = ['$\\sigma$', '$M_{DM}$']
markers = ['solid','--']
for i in range(2):
    custom_lines.append( Line2D([0],[0], linestyle = markers[i], color = 'black', 
            label = labels[i]) )
    
ax[2].legend(handles = custom_lines, loc = 'lower left')

plt.savefig('../graph/O11_contours_all_int_prob_sup.pdf')
# -

m_vals = np.logspace(np.min(pars_slices[:,0]), np.max(pars_slices[:,0]),30)
cross_vals = np.logspace(np.min(pars_slices[:,1]), np.max(pars_slices[:,1]),30)

pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice(['../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2/'])

# +
folders = ['../data/andresData/O4-fulldata/O4/theta-minuspidiv2/O4-slices01-minuspidiv2/',
           '../data/andresData/O4-fulldata/O4/theta-minuspidiv2/O4-slices01-minuspidiv2-v2/',
           '../data/andresData/O4-fulldata/O4/theta-minuspidiv2/O4-slices01-minuspidiv2-v3/',
           '../data/andresData/O4-fulldata/O4/theta-minuspidiv2/O4-slices01-minuspidiv2-v4/',
           '../data/andresData/O4-fulldata/O4/theta-minuspidiv2/O4-slices01-minuspidiv2-v5/'
         ]

sigmas_full       = []
int_prob_full     = []
int_prob_sup_full = []

#for folder in folders:
#    pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice([folder])
 
    # Let's normalize testset between 0 and 1
    
pars_norm = (pars_slices - pars_min) / (pars_max - pars_min)
    
x_norm_s1s2 = x_s1s2 = s1s2_slices[:,:-1,:-1]
    
bps_ind = np.where(np.round(pars_slices[:,0], 4) == np.round(np.log10(m_vals[15]), 4))[0]
c = 0
for itest in bps_ind:
    c = c + 1
    print(c)
    x_obs = x_norm_s1s2[itest, :,:]
    # We have to put this "observation" into a swyft.Sample object
    obs = swyft.Sample(x = x_obs.reshape(1,96,96))
    
    # Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
    pars_true = pars_slices[itest,:]
    pars_prior    = np.random.uniform(low = 0, high = 1, size = (10_000, 3))
    prior_samples = swyft.Samples(z = pars_prior)
    
    # Finally we make the inference   
    predictions_s1s2 = trainer_s1s2.infer(network_s1s2, obs, prior_samples)    
    ratios_s1s2 = np.exp(np.asarray(predictions_s1s2[0].logratios[:,0]))
    masses_pred = np.asarray(predictions_s1s2[0].params[:,0,0]) * (pars_max[0] - pars_min[0]) + pars_min[0]           
    ind_sort    = np.argsort(masses_pred)
    ratios_s1s2 = ratios_s1s2[ind_sort]
    masses_pred = masses_pred[ind_sort]
    m_min = np.argmin(np.abs(masses_pred - 1))
    m_max = np.argmin(np.abs(masses_pred - 2.6))
    masses_int_prob_sup = trapezoid(ratios_s1s2[m_min:m_max], masses_pred[m_min:m_max]) / trapezoid(ratios_s1s2, masses_pred)

    x_obs = x_norm_rate[itest, :]
    obs = swyft.Sample(x = x_obs)
    predictions_rate = trainer_rate.infer(network_rate, obs, prior_samples)    
    ratios_rate = np.exp(np.asarray(predictions_rate[0].logratios[:,0]))
    masses_pred = np.asarray(predictions_rate[0].params[:,0,0]) * (pars_max[0] - pars_min[0]) + pars_min[0]           
    ind_sort    = np.argsort(masses_pred)
    ratios_rate = ratios_rate[ind_sort]
    masses_pred = masses_pred[ind_sort]
    m_min = np.argmin(np.abs(masses_pred - 1))
    m_max = np.argmin(np.abs(masses_pred - 2.6))
    masses_int_prob_sup = trapezoid(ratios_rate[m_min:m_max], masses_pred[m_min:m_max]) / trapezoid(ratios_rate, masses_pred)
    
    
    x_obs = x_norm_drate[itest, :]
    obs = swyft.Sample(x = x_obs)
    predictions_drate = trainer_drate.infer(network_drate, obs, prior_samples)    
    ratios_drate = np.exp(np.asarray(predictions_drate[0].logratios[:,0]))
    masses_pred = np.asarray(predictions_drate[0].params[:,0,0]) * (pars_max[0] - pars_min[0]) + pars_min[0]           
    ind_sort    = np.argsort(masses_pred)
    ratios_drate = ratios_drate[ind_sort]
    masses_pred = masses_pred[ind_sort]
    m_min = np.argmin(np.abs(masses_pred - 1))
    m_max = np.argmin(np.abs(masses_pred - 2.6))
    masses_int_prob_sup = trapezoid(ratios_drate[m_min:m_max], masses_pred[m_min:m_max]) / trapezoid(ratios_drate, masses_pred)
 
    fig, ax = plt.subplots(1,2, figsize = (12,5))

    if masses_int_prob_sup > 0.9:
        im = plot_1dpost(masses_pred, ratios_s1s2, ax[0], color = color_s1s2)
        im = plot_1dpost(masses_pred, ratios_rate, ax[0], color = color_rate)
        im = plot_1dpost(masses_pred, ratios_drate, ax[0], color = color_drate)
    else:
        im = plot_1dpost(masses_pred, ratios_s1s2, ax[0], color = color_s1s2)
        im = plot_1dpost(masses_pred, ratios_rate, ax[0], color = color_rate)
        im = plot_1dpost(masses_pred, ratios_drate, ax[0], color = color_drate)

        
    ax[0].text(0.55, 0.9, 'm = {:.2e} [GeV]'.format(10**pars_true[0]), transform = ax[0].transAxes)
    ax[0].text(0.55, 0.8, '$\sigma$ = {:.2e} [$cm^2$]'.format(10**pars_true[1]), transform = ax[0].transAxes)
    ax[0].text(0.55, 0.7, 'Int. Prob = {:.2f}'.format(masses_int_prob_sup), transform = ax[0].transAxes)    
    
    fig00 = ax[1].contourf(m_vals, cross_vals, M_int_prob_sup_pi_2_s1s2.reshape(30,30).T, levels=5, alpha = 0.6, zorder = 1, cmap = 'inferno')
    ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2.reshape(30,30).T, levels=5, linewidths = 2, zorder = 4)

    #ax[1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
    ax[1].scatter(10**(pars_true[0]), 10**(pars_true[1]), c = 'red', marker = '*')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].grid(which='both')
    ax[1].text(3e2, 1e-36, '$\\theta = \pi/2$')
    ax[1].plot(masses, s1s2_90_CL_pi2[2,:], color = 'black', linestyle = '-.', label = 'Bin. Lik. [90%]')
    ax[1].legend(loc = 'lower left')
    
    ax[1].set_ylabel('$\sigma [cm^{2}]$')
    ax[0].set_ylabel('$P(m|x)$')
    ax[0].set_xlabel('$log(m)$')

    ax[0].set_xlim(0, 3)
    ax[1].set_ylim(1e-43, 3e-36)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(fig00, cax=cbar_ax)
    cbar.ax.set_title('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')
    
    plt.savefig('../graph/O4_Mass_gif_plot_m_84/' + str(c) + '_s1s2.jpg') 
# -

1

# ## Figure 1

linestyle = ['solid','--',':','-.']

# +
folders = ['../data/andresData/O4-fulldata/O4/theta-0/O4-slices01-theta0/',
           '../data/andresData/O4-fulldata/O4/theta-0/O4-slices01-theta0-v2/',
           '../data/andresData/O4-fulldata/O4/theta-0/O4-slices01-theta0-v3/',
           '../data/andresData/O4-fulldata/O4/theta-0/O4-slices01-theta0-v4/',
           '../data/andresData/O4-fulldata/O4/theta-0/O4-slices01-theta0-v5/'
         ]

sigmas_full       = []
int_prob_full     = []
int_prob_sup_full = []

bps = [16*30 + 10, 16*30 + 14, 16*30 + 18, 16*30 + 22]

pars_prior    = np.random.uniform(low = 0, high = 1, size = (100_000, 3))
prior_samples = swyft.Samples(z = pars_prior)

ratios_s1s2     = np.zeros((4, 5, 100_000))
low_1sigma_s1s2 = np.zeros((4, 5, 1))
up_1sigma_s1s2  = np.zeros((4, 5, 1))

ratios_drate     = np.zeros((4, 5, 100_000))
low_1sigma_drate = np.zeros((4, 5, 1))
up_1sigma_drate  = np.zeros((4, 5, 1))

ratios_rate     = np.zeros((4, 5, 100_000))
low_1sigma_rate = np.zeros((4, 5, 1))
up_1sigma_rate  = np.zeros((4, 5, 1)) 

m_true = []
sigma_true = []

for ifold, folder in enumerate(folders):
    pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice([folder])
    pars_norm = (pars_slices - pars_min) / (pars_max - pars_min)
    for i, itest in enumerate(bps):
        ratios_s1s2_aux     = []
        low_1sigma_s1s2_aux = []
        up_1sigma_s1s2_aux  = []
    #  ------------------------------  S1S2 -----------------------------------------------
        # Let's normalize testset between 0 and 1  
            
        x_norm_s1s2 = x_s1s2 = s1s2_slices[:,:-1,:-1]   
        
        x_obs = x_norm_s1s2[itest, :,:]
        
        # We have to put this "observation" into a swyft.Sample object
        obs = swyft.Sample(x = x_obs.reshape(1,96,96))
        
        # Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
        pars_true = pars_slices[itest,:]

        if ifold == 0:
            m_true.append(pars_true[0])
            sigma_true.append(pars_true[1])
        
        # Finally we make the inference
        predictions = trainer_s1s2.infer(network_s1s2, obs, prior_samples)
        
        bins = 50
        logratios = predictions[0].logratios[:,1]
        v         = predictions[0].params[:,1,0]
    
        low, upp = v.min(), v.max()
        weights  = torch.exp(logratios) / torch.exp(logratios).mean(axis = 0)
        h1       = torchist.histogramdd(predictions[0].params[:,1,:], bins, weights = weights, low=low, upp=upp)
        h1      /= len(predictions[0].params[:,1,:]) * (upp - low) / bins
        h1       = np.array(h1)
        vals = sorted(swyft.plot.plot2.get_HDI_thresholds(h1, cred_level=[0.68268, 0.95450, 0.99730]))
        
        edges = torch.linspace(v.min(), v.max(), bins + 1)
        x     = np.array((edges[1:] + edges[:-1]) / 2) * (pars_max[1] - pars_min[1]) + pars_min[1]
    
        low_1sigma_s1s2[i, ifold, :] = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
        up_1sigma_s1s2[i, ifold, :] = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
        
        
        cross_sec = np.asarray(v) * (pars_max[1] - pars_min[1]) + pars_min[1]
        
        ind_sort = np.argsort(cross_sec)
        
        logratios = logratios[ind_sort]
        ratios_s1s2[i, ifold, :] = np.exp(np.asarray(logratios))
        cross_sec = cross_sec[ind_sort]

        #  ------------------------------  drate -----------------------------------------------
        # Let's normalize testset between 0 and 1  
        
        x_drate = diff_rate_slices
        x_norm_drate = (x_drate - x_min_drate) / (x_max_drate - x_min_drate) 
        
        x_obs = x_norm_drate[itest, :]
        
        # We have to put this "observation" into a swyft.Sample object
        obs = swyft.Sample(x = x_obs)
        
        # Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
        pars_true = pars_slices[itest,:]
        
        # Finally we make the inference
        predictions = trainer_drate.infer(network_drate, obs, prior_samples)
        
        bins = 50
        logratios = predictions[0].logratios[:,1]
        v         = predictions[0].params[:,1,0]
    
        low, upp = v.min(), v.max()
        weights  = torch.exp(logratios) / torch.exp(logratios).mean(axis = 0)
        h1       = torchist.histogramdd(predictions[0].params[:,1,:], bins, weights = weights, low=low, upp=upp)
        h1      /= len(predictions[0].params[:,1,:]) * (upp - low) / bins
        h1       = np.array(h1)
        vals = sorted(swyft.plot.plot2.get_HDI_thresholds(h1, cred_level=[0.68268, 0.95450, 0.99730]))
        
        edges = torch.linspace(v.min(), v.max(), bins + 1)
        x     = np.array((edges[1:] + edges[:-1]) / 2) * (pars_max[1] - pars_min[1]) + pars_min[1]
    
        low_1sigma_drate[i, ifold, :] = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
        up_1sigma_drate[i, ifold, :] = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
        
        
        cross_sec = np.asarray(v) * (pars_max[1] - pars_min[1]) + pars_min[1]
        
        ind_sort = np.argsort(cross_sec)
        
        logratios = logratios[ind_sort]
        ratios_drate[i, ifold, :] = np.exp(np.asarray(logratios))
        cross_sec = cross_sec[ind_sort]

        #  ------------------------------  rate -----------------------------------------------
        # Let's normalize testset between 0 and 1  
        
        x_rate = np.log10(rate_slices)
        x_norm_rate = (x_rate - x_min_rate) / (x_max_rate - x_min_rate)
        x_norm_rate = x_norm_rate.reshape(len(x_norm_rate), 1)
        
        x_obs = x_norm_rate[itest, :]
        
        # We have to put this "observation" into a swyft.Sample object
        obs = swyft.Sample(x = x_obs)
        
        # Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
        pars_true = pars_slices[itest,:]
        
        # Finally we make the inference
        predictions = trainer_rate.infer(network_rate, obs, prior_samples)
        
        bins = 50
        logratios = predictions[0].logratios[:,1]
        v         = predictions[0].params[:,1,0]
    
        low, upp = v.min(), v.max()
        weights  = torch.exp(logratios) / torch.exp(logratios).mean(axis = 0)
        h1       = torchist.histogramdd(predictions[0].params[:,1,:], bins, weights = weights, low=low, upp=upp)
        h1      /= len(predictions[0].params[:,1,:]) * (upp - low) / bins
        h1       = np.array(h1)
        vals = sorted(swyft.plot.plot2.get_HDI_thresholds(h1, cred_level=[0.68268, 0.95450, 0.99730]))
        
        edges = torch.linspace(v.min(), v.max(), bins + 1)
        x     = np.array((edges[1:] + edges[:-1]) / 2) * (pars_max[1] - pars_min[1]) + pars_min[1]
    
        low_1sigma_rate[i, ifold, :] = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
        up_1sigma_rate[i, ifold, :] = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
        
        
        cross_sec = np.asarray(v) * (pars_max[1] - pars_min[1]) + pars_min[1]
        
        ind_sort = np.argsort(cross_sec)
        
        logratios = logratios[ind_sort]
        ratios_rate[i, ifold, :] = np.exp(np.asarray(logratios))
        cross_sec = cross_sec[ind_sort]


ratios_s1s2_0     = np.mean(ratios_s1s2, axis = 1)    
low_1sigma_s1s2_0 = np.mean(low_1sigma_s1s2, axis = 1)    
up_1sigma_s1s2_0  = np.mean(up_1sigma_s1s2, axis = 1)    

ratios_drate_0     = np.mean(ratios_drate, axis = 1)    
low_1sigma_drate_0 = np.mean(low_1sigma_drate, axis = 1)    
up_1sigma_drate_0  = np.mean(up_1sigma_drate, axis = 1)  

ratios_rate_0     = np.mean(ratios_rate, axis = 1)    
low_1sigma_rate_0 = np.mean(low_1sigma_rate, axis = 1)    
up_1sigma_rate_0  = np.mean(up_1sigma_rate, axis = 1)  

# +
fig, ax = plt.subplots(2,2, sharex = True, sharey=True)
fig.subplots_adjust(hspace = 0, wspace = 0)

ax[0,0].plot(cross_sec, ratios_s1s2_mpi2[0,:], linestyle = linestyle[0], color = color_s1s2, label = 'S1-S2')
ax[0,1].plot(cross_sec, ratios_s1s2_mpi2[1,:], linestyle = linestyle[0], color = color_s1s2)
ax[1,0].plot(cross_sec, ratios_s1s2_mpi2[2,:], linestyle = linestyle[0], color = color_s1s2)
ax[1,1].plot(cross_sec, ratios_s1s2_mpi2[3,:], linestyle = linestyle[0], color = color_s1s2)

ax[0,0].plot(cross_sec, ratios_drate_mpi2[0,:], linestyle = linestyle[1], color = color_drate, label = 'Dif. Rate')
ax[0,1].plot(cross_sec, ratios_drate_mpi2[1,:], linestyle = linestyle[1], color = color_drate)
ax[1,0].plot(cross_sec, ratios_drate_mpi2[2,:], linestyle = linestyle[1], color = color_drate)
ax[1,1].plot(cross_sec, ratios_drate_mpi2[3,:], linestyle = linestyle[1], color = color_drate)

ax[0,0].plot(cross_sec, ratios_rate_mpi2[0,:], linestyle = linestyle[2], color = color_rate, label = 'Rate')
ax[0,1].plot(cross_sec, ratios_rate_mpi2[1,:], linestyle = linestyle[2], color = color_rate)
ax[1,0].plot(cross_sec, ratios_rate_mpi2[2,:], linestyle = linestyle[2], color = color_rate)
ax[1,1].plot(cross_sec, ratios_rate_mpi2[3,:], linestyle = linestyle[2], color = color_rate)

ax[0,0].text(-43,7, '$\sigma = $' + '{:.2e}'.format(10**sigma_true[0]))
ax[0,1].text(-43,7, '$\sigma = $' + '{:.2e}'.format(10**sigma_true[1]))
ax[1,0].text(-43,7, '$\sigma = $' + '{:.2e}'.format(10**sigma_true[2]))
ax[1,1].text(-43,7, '$\sigma = $' + '{:.2e}'.format(10**sigma_true[3]))

ax[0,0].legend(loc = 'upper right')


ax[0,0].axvline(x = sigma_true[0], color = 'black')
ax[0,1].axvline(x = sigma_true[1], color = 'black')
ax[1,0].axvline(x = sigma_true[2], color = 'black')
ax[1,1].axvline(x = sigma_true[3], color = 'black')

ax[0,0].set_ylabel('$P(\sigma|x)$')
ax[1,0].set_ylabel('$P(\sigma|x)$')
ax[1,0].set_xlabel('$\log_{10}(\sigma)$')
ax[1,1].set_xlabel('$\log_{10}(\sigma)$')

plt.savefig('../graph/O4_PosteriorsExamples_fixSigma.pdf')

# +
plt.plot(cross_sec, ratios_s1s2_mpi2[1,:], linestyle = linestyle[0], color = color_s1s2, label = 'S1-S2')
plt.plot(cross_sec, ratios_s1s2_mpi4[1,:], linestyle = linestyle[1], color = color_s1s2)
plt.plot(cross_sec, ratios_s1s2_0[1,:], linestyle = linestyle[2], color = color_s1s2)

#plt.plot(cross_sec, ratios_drate_mpi2[1,:], linestyle = linestyle[0], color = color_drate, label = 'Dif. Rate')
#plt.plot(cross_sec, ratios_drate_mpi4[1,:], linestyle = linestyle[1], color = color_drate)
#plt.plot(cross_sec, ratios_drate_0[1,:], linestyle = linestyle[2], color = color_drate)

#plt.plot(cross_sec, ratios_rate_mpi2[1,:], linestyle = linestyle[0], color = color_rate, label = 'Rate')
#plt.plot(cross_sec, ratios_rate_mpi4[1,:], linestyle = linestyle[1], color = color_rate)
#plt.plot(cross_sec, ratios_rate_0[1,:], linestyle = linestyle[2], color = color_rate)

plt.text(-42.5,4.3, '$\sigma = $' + '{:.2e}'.format(10**sigma_true[1]))

legend0 = plt.legend(loc = 'upper left')

custom_lines = []
labels = ['$\\theta = -\pi/2$', '$\\theta = -\pi/4$', '$\\theta = 0$']
markers = ['solid','--', ':']
for i in range(3):
    custom_lines.append( Line2D([0],[0], linestyle = markers[i], color = 'black', 
            label = labels[i]) )
    
#legend1 = plt.legend(plot_lines[0], ["algo1", "algo2", "algo3"], loc=1)
#pyplot.legend([l[0] for l in plot_lines], parameters, loc=4)

legend1 = plt.legend(handles = custom_lines, loc = 'upper right')
plt.gca().add_artist(legend1)
#plt.gca().add_artist(legend0)

plt.axvline(x = sigma_true[1], color = 'black')

plt.ylabel('$P(\sigma|x)$')
plt.xlabel('$\log_{10}(\sigma)$')

plt.savefig('../graph/O4_PosteriorsExamples_varSigma.pdf')
# -


