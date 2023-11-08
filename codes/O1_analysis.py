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

# +
#from playsound import playsound
#playsound('/home/martinrios/Downloads/mario.mp3')
# -

from torchsummary import summary

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

# Let's transform the diff_rate to counts per energy bin

#diff_rate = np.round(diff_rate * 362440)
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
# -

diff_rate_WIMP[test_ind[i],:]

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
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O1_rate_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
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
    checkpoint_callback.to_yaml("./logs/O1_rate.yaml") 
    ckpt_path = swyft.best_from_yaml("./logs/O1_rate.yaml")
    email('Termino de entrenar rate O1')
    
else:
    ckpt_path = swyft.best_from_yaml("./logs/O1_rate.yaml")

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
    plt.savefig('../graph/O1_loss_rate.pdf')

# ### Let's make some inference

# +
# Let's normalize testset between 0 and 1

pars_norm = (pars_testset - pars_min) / (pars_max - pars_min)

x_rate = np.log10(rate_testset)
x_norm_rate = (x_rate - x_min_rate) / (x_max_rate - x_min_rate)
x_norm_rate = x_norm_rate.reshape(len(x_norm_rate), 1)

# +
# First let's create some observation from some "true" theta parameters
i = np.random.randint(ntest) # 239 (disc) 455 (exc) 203 (middle)
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
    plt.savefig('../graph/O1_cornerplot_rate_exc.pdf')
else:
    plt.savefig('../graph/O1_cornerplot_rate.pdf')

# +
cross_sec = np.asarray(predictions_rate[0].params[:,1,0]) * B + A
ratios = np.exp(np.asarray(predictions_rate[0].logratios[:,1]))

ind_sort = np.argsort(cross_sec)
ratios = ratios[ind_sort]
cross_sec = cross_sec[ind_sort]

plt.plot(cross_sec, ratios, c = 'blue')
plt.xlabel('$\log_{10}(\sigma)$')
plt.ylabel('$P(\sigma|x)\ /\ P(\sigma)$')
# -

trapezoid(ratios, cross_sec)

# +
cr_th = np.argmin(np.abs(cross_sec + 49))

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
cross_section_th = -49
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
plt.text(-50,3, '$m = {:.2e}$'.format(10**(pars_true[0])))
plt.text(-50,2.8, '$\sigma = {:.2e}$'.format(10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1])))
plt.text(-50,2.5, '$\\theta = {:.2f}$'.format(pars_true[0]))
plt.plot(cross_sec, ratios, c = 'blue')

if flag == 'exc':
    plt.savefig('../graph/O1_1Dposterior_rate_exc_' + str(i) + '.pdf')
else:
    plt.savefig('../graph/O1_1Dposterior_rate_' + str(i) + '.pdf')
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
    plt.savefig('../graph/O1_loglikratio_rate_exc.pdf')
else:
    plt.savefig('../graph/O1_loglikratio_rate.pdf')

# +
masses_pred = np.asarray(predictions_rate[0].params[:,0,0]) * (pars_max[0] - pars_min[0]) + pars_min[0]
ratios = np.exp(np.asarray(predictions_rate[0].logratios[:,0]))

ind_sort = np.argsort(masses_pred)
ratios = ratios[ind_sort]
masses_pred = masses_pred[ind_sort]

plt.plot(masses_pred, ratios, c = 'blue')
plt.xlabel('$\log_{10}(m_{DM})$')
plt.ylabel('$P(m_{DM}|x)\ /\ P(m_{DM})$')
# -

trapezoid(ratios, masses_pred)

# +
m_min = np.argmin(np.abs(masses_pred - 1))
m_max = np.argmin(np.abs(masses_pred - 2.6))

trapezoid(ratios[m_min:m_max], masses_pred[m_min:m_max]) / trapezoid(ratios, masses_pred)
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
# -

# ### Let's make the contour plot ($\sigma$)

# !ls ../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2

pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice(['../data/andresData/O1-slices-5vecescadatheta/theta-0/SI-slices01-theta0/'])

m_vals = np.logspace(np.min(pars_slices[:,0]), np.max(pars_slices[:,0]),30)
cross_vals = np.logspace(np.min(pars_slices[:,1]), np.max(pars_slices[:,1]),30)

# +
force = False
folders = ['../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v3/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v4/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v5/'
         ]


cross_sec_sigmas_full       = []
cross_sec_int_prob_full     = []
cross_sec_int_prob_sup_full = []

masses_int_prob_sup_full = []
masses_prob_sup_full     = []
masses_prob_inf_full     = []

for folder in folders:
    pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice([folder])
    
    if (os.path.exists(folder + 'cross_sec_sigmas_rate.txt') & 
        os.path.exists(folder + 'cross_sec_int_prob_rate.txt') &
        os.path.exists(folder + 'cross_sec_int_prob_sup_rate.txt') &
        os.path.exists(folder + 'masses_int_prob_sup_rate.txt') &
        os.path.exists(folder + 'masses_prob_sup_rate.txt') &
        os.path.exists(folder + 'masses_prob_inf_rate.txt') 
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
        masses_prob_sup        = np.ones(len(pars_norm)) * -99
        masses_prob_inf        = np.ones(len(pars_norm)) * -99
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
            
            cross_sec_sigmas[itest,0] = np.min(x[np.where(np.array(h1) >= np.array(vals[2]))[0]])
            cross_sec_sigmas[itest,3] = np.max(x[np.where(np.array(h1) >= np.array(vals[2]))[0]])
            
            cross_sec_sigmas[itest,1] = np.min(x[np.where(np.array(h1) >= np.array(vals[1]))[0]])
            cross_sec_sigmas[itest,4] = np.max(x[np.where(np.array(h1) >= np.array(vals[1]))[0]])
            
            cross_sec_sigmas[itest,2] = np.min(x[np.where(np.array(h1) >= np.array(vals[0]))[0]])
            cross_sec_sigmas[itest,5] = np.max(x[np.where(np.array(h1) >= np.array(vals[0]))[0]])
            
            cr_th = np.argmin(np.abs(x - (-49)))
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
            masses_prob_sup[itest] = trapezoid(ratios_rate[m_min:], masses_pred[m_min:]) / trapezoid(ratios_rate, masses_pred)
            masses_prob_inf[itest] = trapezoid(ratios_rate[:m_max], masses_pred[:m_max]) / trapezoid(ratios_rate, masses_pred)

        cross_sec_sigmas_full.append(cross_sec_sigmas)
        cross_sec_int_prob_full.append(cross_sec_int_prob)
        cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
        masses_int_prob_sup_full.append(masses_int_prob_sup)
        masses_prob_sup_full.append(masses_prob_sup)
        masses_prob_inf_full.append(masses_prob_inf)
            
        np.savetxt(folder + 'cross_sec_sigmas_rate.txt', cross_sec_sigmas)
        np.savetxt(folder + 'cross_sec_int_prob_rate.txt', cross_sec_int_prob)
        np.savetxt(folder + 'cross_sec_int_prob_sup_rate.txt', cross_sec_int_prob_sup)
        np.savetxt(folder + 'masses_int_prob_sup_rate.txt', masses_int_prob_sup)
        np.savetxt(folder + 'masses_prob_sup_rate.txt', masses_prob_sup)
        np.savetxt(folder + 'masses_prob_inf_rate.txt', masses_prob_inf)
    else:
        print('pre-computed')
                
        cross_sec_sigmas = np.loadtxt(folder + 'cross_sec_sigmas_rate.txt')
        cross_sec_int_prob = np.loadtxt(folder + 'cross_sec_int_prob_rate.txt')
        cross_sec_int_prob_sup = np.loadtxt(folder + 'cross_sec_int_prob_sup_rate.txt')
        masses_int_prob_sup = np.loadtxt(folder + 'masses_int_prob_sup_rate.txt')
        masses_prob_sup = np.loadtxt(folder + 'masses_prob_sup_rate.txt')
        masses_prob_inf = np.loadtxt(folder + 'masses_prob_inf_rate.txt')
        
        cross_sec_sigmas_full.append(cross_sec_sigmas)
        cross_sec_int_prob_full.append(cross_sec_int_prob)
        cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
        masses_int_prob_sup_full.append(masses_int_prob_sup)
        masses_prob_sup_full.append(masses_prob_sup)
        masses_prob_inf_full.append(masses_prob_inf)

email('Termino de analizar las slices')

# +
cross_section_th = -49

if len(cross_sec_int_prob_full) > 1:
    cross_sec_int_prob_mpi_2        = np.mean(np.asarray(cross_sec_int_prob_full), axis = 0)
    cross_sec_int_prob_sup_mpi_2    = np.mean(np.asarray(cross_sec_int_prob_sup_full), axis = 0)
    cross_sec_int_prob_sup_mpi_2_sd = np.std(np.asarray(cross_sec_int_prob_sup_full), axis = 0)
    cross_sec_sigmas_mpi_2          = np.mean(np.asarray(cross_sec_sigmas_full), axis = 0)
    masses_int_prob_sup_mpi_2       = np.mean(np.asarray(masses_int_prob_sup_full), axis = 0)
    masses_int_prob_sup_mpi_2_sd    = np.std(np.asarray(masses_int_prob_sup_full), axis = 0)
    masses_prob_sup_mpi_2           = np.mean(np.asarray(masses_prob_sup_full), axis = 0)
    masses_prob_sup_mpi_2_sd        = np.std(np.asarray(masses_prob_sup_full), axis = 0)
    masses_prob_inf_mpi_2           = np.mean(np.asarray(masses_prob_inf_full), axis = 0)
    masses_prob_inf_mpi_2_sd        = np.std(np.asarray(masses_prob_inf_full), axis = 0)
else:
    cross_sec_int_prob_mpi_2     = cross_sec_int_prob
    cross_sec_int_prob_sup_mpi_2 = cross_sec_int_prob_sup
    masses_int_prob_sup_mpi_2    = masses_int_prob_sup
    masses_prob_sup_mpi_2        = masses_prob_sup
    masses_prob_inf_mpi_2        = masses_prob_inf

rate_1sigma_mpi_2 = np.ones(900) * -99
rate_2sigma_mpi_2 = np.ones(900) * -99
rate_3sigma_mpi_2 = np.ones(900) * -99

rate_1sigma_mpi_2[np.where(cross_sec_sigmas_mpi_2[:,0] > cross_section_th)[0]] = 1
rate_2sigma_mpi_2[np.where(cross_sec_sigmas_mpi_2[:,1] > cross_section_th)[0]] = 1
rate_3sigma_mpi_2[np.where(cross_sec_sigmas_mpi_2[:,2] > cross_section_th)[0]] = 1

# +
fig, ax = plt.subplots(2,2)

sbn.kdeplot(cross_sec_int_prob_sup_0, label = '$\\theta = 0$', ax = ax[0,0])
sbn.kdeplot(cross_sec_int_prob_sup_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[0,0])
sbn.kdeplot(cross_sec_int_prob_sup_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[0,0])
sbn.kdeplot(cross_sec_int_prob_sup_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[0,0])
sbn.kdeplot(cross_sec_int_prob_sup_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[0,0])
ax[0,0].legend()
ax[0,0].set_xlabel('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')
ax[0,0].set_title('Total Rate')

sbn.kdeplot(masses_int_prob_sup_0, label = '$\\theta = 0$', ax = ax[0,1])
sbn.kdeplot(masses_int_prob_sup_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[0,1])
sbn.kdeplot(masses_int_prob_sup_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[0,1])
sbn.kdeplot(masses_int_prob_sup_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[0,1])
sbn.kdeplot(masses_int_prob_sup_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[0,1])
ax[0,1].legend()
ax[0,1].set_xlabel('$\int_{m_{min}}^{m_{max}} P(m_{DM}|x)$')
ax[0,1].set_title('Total Rate')

sbn.kdeplot(masses_prob_sup_0, label = '$\\theta = 0$', ax = ax[1,0])
sbn.kdeplot(masses_prob_sup_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[1,0])
sbn.kdeplot(masses_prob_sup_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[1,0])
sbn.kdeplot(masses_prob_sup_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[1,0])
sbn.kdeplot(masses_prob_sup_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[1,0])
ax[1,0].legend()
ax[1,0].set_xlabel('$\int_{m_{min}}^{\inf} P(m_{DM}|x)$')

sbn.kdeplot(masses_prob_inf_0, label = '$\\theta = 0$', ax = ax[1,1])
sbn.kdeplot(masses_prob_inf_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[1,1])
sbn.kdeplot(masses_prob_inf_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[1,1])
sbn.kdeplot(masses_prob_inf_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[1,1])
sbn.kdeplot(masses_prob_inf_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[1,1])
ax[1,1].legend()
ax[1,1].set_xlabel('$\int_{0}^{m_{max}} P(m_{DM}|x)$')

#plt.savefig('../graph/O1_int_prob_distribution_rate.pdf')

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

ax[0,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
ax[0,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
ax[0,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[0,0].fill_between(neutrino_fog[:,0], neutrino_fog[:,1], 1e-50, color = 'lightblue', alpha = 0.6)
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-44, '$\\theta = \pi/2$')
#ax[0,0].legend(loc = 'lower right')

ax[0,1].contour(m_vals, cross_vals, rate_1sigma_pi_4_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[0,1].contour(m_vals, cross_vals, rate_2sigma_pi_4_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[0,1].contourf(m_vals, cross_vals, rate_3sigma_pi_4_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, rate_3sigma_pi_4_g.reshape(30,30).T, levels=[0])

ax[0,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--', label = 'XENON nT [$3\sigma$]')
ax[0,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':', label = 'XENON nT [$5\sigma$]')
ax[0,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
ax[0,1].fill_between(neutrino_fog[:,0], neutrino_fog[:,1], 1e-50, color = 'lightblue', alpha = 0.6)
ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-44, '$\\theta = \pi/4$')
ax[0,1].legend(loc = 'lower right')

#ax[1,0].contour(m_vals, cross_vals, int_prob_0.reshape(30,30).T, levels=10, linewidths = 2, zorder = 4, linestyles = '--')
ax[1,0].contour(m_vals, cross_vals, rate_1sigma_mpi_2_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[1,0].contour(m_vals, cross_vals, rate_2sigma_mpi_2_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[1,0].contourf(m_vals, cross_vals, rate_3sigma_mpi_2_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, rate_3sigma_mpi_2_g.reshape(30,30).T, levels=[0])

ax[1,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
ax[1,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
ax[1,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,0].fill_between(neutrino_fog[:,0], neutrino_fog[:,1], 1e-50, color = 'lightblue', alpha = 0.6)
ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-44, '$\\theta = -\pi/2$')

ax[1,1].contour(m_vals, cross_vals, rate_1sigma_0_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[1,1].contour(m_vals, cross_vals, rate_2sigma_0_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[1,1].contourf(m_vals, cross_vals, rate_3sigma_0_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, rate_3sigma_0_g.reshape(30,30).T, levels=[0])

ax[1,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
ax[1,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
ax[1,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,1].fill_between(neutrino_fog[:,0], neutrino_fog[:,1], 1e-50, color = 'lightblue', alpha = 0.6)
ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-44, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma$ []')
ax[1,0].set_ylabel('$\sigma$ []')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(1e-49, 1e-43)
ax[0,0].set_xlim(5, 1e3)

plt.savefig('../graph/O1_contours_rate.pdf')

# +
levels = [0, 0.1, 0.16, 0.24, 0.32]

sigma = 1.81 # this depends on how noisy your data is, play with it!

CR_int_prob_0_g     = gaussian_filter(cross_sec_int_prob_0, sigma)
CR_int_prob_pi_2_g  = gaussian_filter(cross_sec_int_prob_pi_2, sigma)
CR_int_prob_pi_4_g  = gaussian_filter(cross_sec_int_prob_pi_4, sigma)
CR_int_prob_mpi_2_g = gaussian_filter(cross_sec_int_prob_mpi_2, sigma)
CR_int_prob_mpi_4_g = gaussian_filter(cross_sec_int_prob_mpi_4, sigma)

fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))

fig00 = ax[0,0].contourf(m_vals, cross_vals, CR_int_prob_pi_2_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_pi_2_g.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)


#ax[0,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
#ax[0,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
ax[0,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
ax[0,0].fill_between(neutrino_fog[:,0], neutrino_fog[:,1], 1e-50, color = 'lightblue', alpha = 0.6)
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-44, '$\\theta = \pi/2$')
ax[0,0].plot(masses, rate_90_CL_pi2[2,:], color = 'black', linestyle = '-.', label = 'Bin. Lik. [90%]')
ax[0,0].legend(loc = 'lower left')

ax[0,1].contourf(m_vals, cross_vals, CR_int_prob_pi_4_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, CR_int_prob_pi_4_g.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)

#ax[0,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--', label = 'XENON nT [$3\sigma$]')
#ax[0,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':', label = 'XENON nT [$5\sigma$]')
#ax[0,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-44, '$\\theta = \pi/4$')

ax[1,0].contourf(m_vals, cross_vals, CR_int_prob_mpi_2_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, CR_int_prob_mpi_2_g.reshape(30,30).T, levels=levels)

#ax[1,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
#ax[1,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
#ax[1,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-44, '$\\theta = -\pi/2$')

ax[1,1].contourf(m_vals, cross_vals, CR_int_prob_0_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, CR_int_prob_0_g.reshape(30,30).T, levels=levels)

#ax[1,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
#ax[1,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
#ax[1,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-44, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(1e-49, 1e-43)
ax[0,0].set_xlim(5, 1e3)
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar    = fig.colorbar(fig00, cax=cbar_ax)
cbar.ax.set_title('$\int_{-\inf}^{\sigma_{th}} P(\sigma|x)$')

ax[0,1].plot(masses, rate_90_CL_pi4[2,:], color = 'black', linestyle = '-.')
ax[1,0].plot(masses, rate_90_CL_mpi2[2,:], color = 'black', linestyle = '-.')
ax[1,1].plot(masses, rate_90_CL_0[2,:], color = 'black', linestyle = '-.')

plt.savefig('../graph/O1_contours_rate_int_prob.pdf')

# +
sigma = 1.21 # this depends on how noisy your data is, play with it!

CR_int_prob_sup_0_rate     = gaussian_filter(cross_sec_int_prob_sup_0, sigma)
CR_int_prob_sup_0_rate_min = gaussian_filter(cross_sec_int_prob_sup_0 - cross_sec_int_prob_sup_0_sd, sigma)
CR_int_prob_sup_0_rate_max = gaussian_filter(cross_sec_int_prob_sup_0 + cross_sec_int_prob_sup_0_sd, sigma)

M_int_prob_sup_0_rate     = gaussian_filter(masses_int_prob_sup_0, sigma)
M_int_prob_sup_0_rate_min = gaussian_filter(masses_int_prob_sup_0 - masses_int_prob_sup_0_sd, sigma)
M_int_prob_sup_0_rate_max = gaussian_filter(masses_int_prob_sup_0 + masses_int_prob_sup_0_sd, sigma)
M_prob_sup_0_rate         = gaussian_filter(masses_prob_sup_0, sigma)
M_prob_inf_0_rate         = gaussian_filter(masses_prob_inf_0, sigma)

CR_int_prob_sup_pi_2_rate  = gaussian_filter(cross_sec_int_prob_sup_pi_2, sigma)
CR_int_prob_sup_pi_2_rate_min = gaussian_filter(cross_sec_int_prob_sup_pi_2 - cross_sec_int_prob_sup_pi_2_sd, sigma)
CR_int_prob_sup_pi_2_rate_max = gaussian_filter(cross_sec_int_prob_sup_pi_2 + cross_sec_int_prob_sup_pi_2_sd, sigma)

M_int_prob_sup_pi_2_rate     = gaussian_filter(masses_int_prob_sup_pi_2, sigma)
M_int_prob_sup_pi_2_rate_min = gaussian_filter(masses_int_prob_sup_pi_2 - masses_int_prob_sup_pi_2_sd, sigma)
M_int_prob_sup_pi_2_rate_max = gaussian_filter(masses_int_prob_sup_pi_2 + masses_int_prob_sup_pi_2_sd, sigma)
M_prob_sup_pi_2_rate         = gaussian_filter(masses_prob_sup_pi_2, sigma)
M_prob_inf_pi_2_rate         = gaussian_filter(masses_prob_inf_pi_2, sigma)

CR_int_prob_sup_pi_4_rate  = gaussian_filter(cross_sec_int_prob_sup_pi_4, sigma)
CR_int_prob_sup_pi_4_rate_min = gaussian_filter(cross_sec_int_prob_sup_pi_4 - cross_sec_int_prob_sup_pi_4_sd, sigma)
CR_int_prob_sup_pi_4_rate_max = gaussian_filter(cross_sec_int_prob_sup_pi_4 + cross_sec_int_prob_sup_pi_4_sd, sigma)

M_int_prob_sup_pi_4_rate   = gaussian_filter(masses_int_prob_sup_pi_4, sigma)
M_int_prob_sup_pi_4_rate_min = gaussian_filter(masses_int_prob_sup_pi_4 - masses_int_prob_sup_pi_4_sd, sigma)
M_int_prob_sup_pi_4_rate_max = gaussian_filter(masses_int_prob_sup_pi_4 + masses_int_prob_sup_pi_4_sd, sigma)
M_prob_sup_pi_4_rate         = gaussian_filter(masses_prob_sup_pi_4, sigma)
M_prob_inf_pi_4_rate         = gaussian_filter(masses_prob_inf_pi_4, sigma)

CR_int_prob_sup_mpi_2_rate = gaussian_filter(cross_sec_int_prob_sup_mpi_2, sigma)
CR_int_prob_sup_mpi_2_rate_min = gaussian_filter(cross_sec_int_prob_sup_mpi_2 - cross_sec_int_prob_sup_mpi_2_sd, sigma)
CR_int_prob_sup_mpi_2_rate_max = gaussian_filter(cross_sec_int_prob_sup_mpi_2 + cross_sec_int_prob_sup_mpi_2_sd, sigma)

M_int_prob_sup_mpi_2_rate  = gaussian_filter(masses_int_prob_sup_mpi_2, sigma)
M_int_prob_sup_mpi_2_rate_min = gaussian_filter(masses_int_prob_sup_mpi_2 - masses_int_prob_sup_mpi_2_sd, sigma)
M_int_prob_sup_mpi_2_rate_max = gaussian_filter(masses_int_prob_sup_mpi_2 + masses_int_prob_sup_mpi_2_sd, sigma)
M_prob_sup_mpi_2_rate         = gaussian_filter(masses_prob_sup_mpi_2, sigma)
M_prob_inf_mpi_2_rate         = gaussian_filter(masses_prob_inf_mpi_2, sigma)

CR_int_prob_sup_mpi_4_rate = gaussian_filter(cross_sec_int_prob_sup_mpi_4, sigma)
CR_int_prob_sup_mpi_4_rate_min = gaussian_filter(cross_sec_int_prob_sup_mpi_4 - cross_sec_int_prob_sup_mpi_4_sd, sigma)
CR_int_prob_sup_mpi_4_rate_max = gaussian_filter(cross_sec_int_prob_sup_mpi_4 + cross_sec_int_prob_sup_mpi_4_sd, sigma)

M_int_prob_sup_mpi_4_rate  = gaussian_filter(masses_int_prob_sup_mpi_4, sigma)
M_int_prob_sup_mpi_4_rate_min = gaussian_filter(masses_int_prob_sup_mpi_4 - masses_int_prob_sup_mpi_4_sd, sigma)
M_int_prob_sup_mpi_4_rate_max = gaussian_filter(masses_int_prob_sup_mpi_4 + masses_int_prob_sup_mpi_4_sd, sigma)
M_prob_sup_mpi_4_rate         = gaussian_filter(masses_prob_sup_mpi_4, sigma)
M_prob_inf_mpi_4_rate         = gaussian_filter(masses_prob_inf_mpi_4, sigma)

# +
cmap = mpl.colormaps.get_cmap('viridis')
norm = mpl.colors.Normalize(vmin=np.min(CR_int_prob_sup_pi_2_rate), vmax=np.max(CR_int_prob_sup_pi_2_rate))

plt.scatter(pars_slices[:,1],np.abs(cross_sec_sigmas_pi_2[:,0] - pars_slices[:,1]), color = cmap(norm(CR_int_prob_sup_pi_2_rate)))
cbar = plt.colorbar()
#cbar.ax.set_yticklabels([])

# +
levels = [0.67, 0.76, 0.84, 0.9, 1] 


fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))

fig00 = ax[0,0].contourf(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
#fig00 = ax[0,0].contourf(m_vals, cross_vals, np.abs(cross_sec_sigmas_pi_2[:,0] - pars_slices[:,1]).reshape(30,30).T, levels=7, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)
ax[0,0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_rate.reshape(30,30).T, levels=levels, linewidths = 2, cmap = 'inferno', linestyles = [':'])
ax[0,0].contour(m_vals, cross_vals, M_prob_sup_pi_2_rate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = ['--'])
ax[0,0].contour(m_vals, cross_vals, M_prob_inf_pi_2_rate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = [(0, (3,5,1,5,1,5))])


#ax[0,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
#ax[0,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
ax[0,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-44, '$\\theta = \pi/2$')
ax[0,0].plot(masses, rate_90_CL_pi2[2,:], color = 'black', linestyle = '-.', label = 'Bin. Lik. [90%]')
ax[0,0].legend(loc = 'lower left')

ax[0,1].contourf(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate.reshape(30,30).T, levels=levels, alpha = 1, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)
fig01 = ax[0,1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_rate.reshape(30,30).T, levels=levels, linestyles = [':'], cmap = 'inferno')
ax[0,1].contour(m_vals, cross_vals, M_prob_sup_pi_4_rate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = ['--'])
ax[0,1].contour(m_vals, cross_vals, M_prob_inf_pi_4_rate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = [(0, (3,5,1,5,1,5))])

#fig01 = ax[0,1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_rate.reshape(30,30).T, levels=5, zorder = 1, cmap = 'inferno')

#ax[0,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--', label = 'XENON nT [$3\sigma$]')
#ax[0,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':', label = 'XENON nT [$5\sigma$]')
#ax[0,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-44, '$\\theta = \pi/4$')

ax[1,0].contourf(m_vals, cross_vals, CR_int_prob_sup_mpi_2_rate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, CR_int_prob_sup_mpi_2_rate.reshape(30,30).T, levels=levels)
ax[1,0].contour(m_vals, cross_vals, M_int_prob_sup_mpi_2_rate.reshape(30,30).T, levels=levels, linestyles = [':'], cmap = 'inferno')
ax[1,0].contour(m_vals, cross_vals, M_prob_sup_mpi_2_rate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = ['--'])
ax[1,0].contour(m_vals, cross_vals, M_prob_inf_mpi_2_rate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = [(0, (3,5,1,5,1,5))])

#ax[1,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
#ax[1,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
#ax[1,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-44, '$\\theta = -\pi/2$')

ax[1,1].contourf(m_vals, cross_vals, CR_int_prob_sup_0_rate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate.reshape(30,30).T, levels=levels)
# #%ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate_min.reshape(30,30).T, levels=[0.9], colors='green', linestyles = '-.')
# #%ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate_max.reshape(30,30).T, levels=[0.9], colors='green', linestyles = '-.')
ax[1,1].contour(m_vals, cross_vals, M_int_prob_sup_0_rate.reshape(30,30).T, levels=levels, cmap = 'inferno', linestyles = ':')
ax[1,1].contour(m_vals, cross_vals, M_prob_sup_0_rate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = ['--'])
ax[1,1].contour(m_vals, cross_vals, M_prob_inf_0_rate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = [(0, (3,5,1,5,1,5))])

#ax[1,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
#ax[1,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
#ax[1,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-44, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(1e-49, 1e-43)

fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar    = fig.colorbar(fig00, cax = cbar_ax)
cbar.ax.set_title('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')

fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
cbar    = fig.colorbar(fig01, cax = cbar_ax)
cbar.ax.set_title('$\int_{m_{min}}^{m_{max}} P(m|x)$')

ax[0,1].plot(masses, rate_90_CL_pi4[2,:], color = 'black', linestyle = '-.')
ax[1,0].plot(masses, rate_90_CL_mpi2[2,:], color = 'black', linestyle = '-.')
ax[1,1].plot(masses, rate_90_CL_0[2,:], color = 'black', linestyle = '-.')

#plt.savefig('../graph/O1_contours_rate_int_prob_sup.pdf')
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
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O1_drate_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
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
    checkpoint_callback.to_yaml("./logs/O1_drate.yaml") 
    ckpt_path = swyft.best_from_yaml("./logs/O1_drate.yaml")
    email('Termino el entramiento del drate para O1')
else:
    ckpt_path = swyft.best_from_yaml("./logs/O1_drate.yaml")

# ---------------------------------------------- 
# It converges to val_loss = -1.8 @ epoch 20
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
    plt.savefig('../graph/O1_loss_drate.pdf')

# ### Let's make some inference

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
plt.text(-50,3, '$m = {:.2e}$'.format(10**(pars_true[0])))
plt.text(-50,2.8, '$\sigma = {:.2e}$'.format(10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1])))
plt.text(-50,2.5, '$\\theta = {:.2f}$'.format(pars_true[0]))
if flag == 'exc':
    plt.savefig('../graph/1Dposterior_drate_exc_' + str(i) + '.pdf')
else:
    plt.savefig('../graph/1Dposterior_drate_' + str(i) + '.pdf')
# -

ratios_drate = np.exp(np.asarray(predictions_drate[0].logratios[:,0]))
masses_pred  = np.asarray(predictions_drate[0].params[:,0,0]) * (pars_max[0] - pars_min[0]) + pars_min[0]           
ind_sort     = np.argsort(masses_pred)
ratios_drate = ratios_drate[ind_sort]
masses_pred  = masses_pred[ind_sort]

plt.plot(masses_pred, ratios_drate)

m_min = np.argmin(np.abs(masses_pred - 1))
m_max = np.argmin(np.abs(masses_pred - 2.6))
print(trapezoid(ratios_drate[m_min:m_max], masses_pred[m_min:m_max]) / trapezoid(ratios_drate, masses_pred))
print(trapezoid(ratios_drate[m_min:], masses_pred[m_min:]) / trapezoid(ratios_drate, masses_pred))
print(trapezoid(ratios_drate[:m_max], masses_pred[:m_max]) / trapezoid(ratios_drate, masses_pred))

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

# ### Let's make the contour plot

# !ls ../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv4

m_vals = np.logspace(np.min(pars_slices[:,0]), np.max(pars_slices[:,0]),30)
cross_vals = np.logspace(np.min(pars_slices[:,1]), np.max(pars_slices[:,1]),30)

# +
force = False 
folders = ['../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v3/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v4/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v5/'
         ]

cross_sec_sigmas_full       = []
cross_sec_int_prob_full     = []
cross_sec_int_prob_sup_full = []

masses_int_prob_sup_full = []
masses_prob_sup_full     = []
masses_prob_inf_full     = []

for folder in folders:
    pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice([folder])

    if (os.path.exists(folder + 'cross_sec_sigmas_drate.txt') & 
        os.path.exists(folder + 'cross_sec_int_prob_drate.txt') &
        os.path.exists(folder + 'cross_sec_int_prob_sup_drate.txt') &
        os.path.exists(folder + 'masses_int_prob_sup_drate.txt') &
        os.path.exists(folder + 'masses_prob_sup_drate.txt') &
        os.path.exists(folder + 'masses_prob_inf_drate.txt') 
       ) == False or force == True:
        # Let's normalize testset between 0 and 1
        
        pars_norm = (pars_slices - pars_min) / (pars_max - pars_min)
        x_drate = diff_rate_slices
        x_norm_drate = (x_drate - x_min_drate) / (x_max_drate - x_min_drate)
        
        cross_sec_sigmas = np.ones((len(pars_slices), 6))
    
        cross_sec_int_prob     = np.ones(len(pars_norm)) * -99
        cross_sec_int_prob_sup = np.ones(len(pars_norm)) * -99
        masses_int_prob_sup    = np.ones(len(pars_norm)) * -99
        masses_prob_sup        = np.ones(len(pars_norm)) * -99
        masses_prob_inf        = np.ones(len(pars_norm)) * -99
           
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
            
            cross_sec_sigmas[itest,0] = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
            cross_sec_sigmas[itest,3] = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
            
            cross_sec_sigmas[itest,1] = np.min(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
            cross_sec_sigmas[itest,4] = np.max(x[np.where(np.array(h1) > np.array(vals[1]))[0]])
            
            cross_sec_sigmas[itest,2] = np.min(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
            cross_sec_sigmas[itest,5] = np.max(x[np.where(np.array(h1) > np.array(vals[0]))[0]])
            
            cr_th = np.argmin(np.abs(x - (-49)))
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
            masses_prob_sup[itest] = trapezoid(ratios_drate[m_min:], masses_pred[m_min:]) / trapezoid(ratios_drate, masses_pred)
            masses_prob_inf[itest] = trapezoid(ratios_drate[:m_max], masses_pred[:m_max]) / trapezoid(ratios_drate, masses_pred)

        cross_sec_sigmas_full.append(cross_sec_sigmas)
        cross_sec_int_prob_full.append(cross_sec_int_prob)
        cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
        masses_int_prob_sup_full.append(masses_int_prob_sup)
        masses_prob_sup_full.append(masses_prob_sup)
        masses_prob_inf_full.append(masses_prob_inf)
            
        np.savetxt(folder + 'cross_sec_sigmas_drate.txt', cross_sec_sigmas)
        np.savetxt(folder + 'cross_sec_int_prob_drate.txt', cross_sec_int_prob)
        np.savetxt(folder + 'cross_sec_int_prob_sup_drate.txt', cross_sec_int_prob_sup)
        np.savetxt(folder + 'masses_int_prob_sup_drate.txt', masses_int_prob_sup)
        np.savetxt(folder + 'masses_prob_sup_drate.txt', masses_prob_sup)
        np.savetxt(folder + 'masses_prob_inf_drate.txt', masses_prob_inf)
    else:
        print('pre-computed')
                
        cross_sec_sigmas = np.loadtxt(folder + 'cross_sec_sigmas_drate.txt')
        cross_sec_int_prob = np.loadtxt(folder + 'cross_sec_int_prob_drate.txt')
        cross_sec_int_prob_sup = np.loadtxt(folder + 'cross_sec_int_prob_sup_drate.txt')
        masses_int_prob_sup = np.loadtxt(folder + 'masses_int_prob_sup_drate.txt')
        masses_prob_sup = np.loadtxt(folder + 'masses_prob_sup_drate.txt')
        masses_prob_inf = np.loadtxt(folder + 'masses_prob_inf_drate.txt')
        
        cross_sec_sigmas_full.append(cross_sec_sigmas)
        cross_sec_int_prob_full.append(cross_sec_int_prob)
        cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
        masses_int_prob_sup_full.append(masses_int_prob_sup)
        masses_prob_sup_full.append(masses_prob_sup)
        masses_prob_inf_full.append(masses_prob_inf)
email('Termino analisis slice')

# +
cross_section_th = -49

if len(cross_sec_int_prob_full) > 1:
    cross_sec_int_prob_drate_mpi_2        = np.mean(np.asarray(cross_sec_int_prob_full), axis = 0)
    cross_sec_int_prob_drate_sup_mpi_2    = np.mean(np.asarray(cross_sec_int_prob_sup_full), axis = 0)
    cross_sec_int_prob_drate_sup_mpi_2_sd = np.std(np.asarray(cross_sec_int_prob_sup_full), axis = 0)
    masses_int_prob_drate_sup_mpi_2       = np.mean(np.asarray(masses_int_prob_sup_full), axis = 0)
    masses_int_prob_drate_sup_mpi_2_sd    = np.std(np.asarray(masses_int_prob_sup_full), axis = 0)
    masses_prob_drate_sup_mpi_2           = np.mean(np.asarray(masses_prob_sup_full), axis = 0)
    masses_prob_drate_sup_mpi_2_sd        = np.std(np.asarray(masses_prob_sup_full), axis = 0)
    masses_prob_drate_inf_mpi_2           = np.mean(np.asarray(masses_prob_inf_full), axis = 0)
    masses_prob_drate_inf_mpi_2_sd        = np.std(np.asarray(masses_prob_inf_full), axis = 0)
    cross_sec_sigmas_mpi_2                = np.mean(np.asarray(cross_sec_sigmas_full), axis = 0)
else:
    cross_sec_int_prob_drate_mpi_2     = cross_sec_int_prob
    cross_sec_int_prob_drate_sup_mpi_2 = cross_sec_int_prob_sup
    masses_int_prob_drate_sup_mpi_2    = masses_int_prob_sup
    masses_prob_drate_sup_mpi_2        = masses_prob_sup
    masses_prob_drate_inf_mpi_2        = masses_prob_inf

rate_1sigma_mpi_2 = np.ones(900) * -99
rate_2sigma_mpi_2 = np.ones(900) * -99
rate_3sigma_mpi_2 = np.ones(900) * -99

rate_1sigma_mpi_2[np.where(cross_sec_sigmas_mpi_2[:,0] > cross_section_th)[0]] = 1
rate_2sigma_mpi_2[np.where(cross_sec_sigmas_mpi_2[:,1] > cross_section_th)[0]] = 1
rate_3sigma_mpi_2[np.where(cross_sec_sigmas_mpi_2[:,2] > cross_section_th)[0]] = 1

# +
fig, ax = plt.subplots(2,2)

sbn.kdeplot(cross_sec_int_prob_drate_sup_0, label = '$\\theta = 0$', ax = ax[0,0])
sbn.kdeplot(cross_sec_int_prob_drate_sup_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[0,0])
sbn.kdeplot(cross_sec_int_prob_drate_sup_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[0,0])
sbn.kdeplot(cross_sec_int_prob_drate_sup_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[0,0])
sbn.kdeplot(cross_sec_int_prob_drate_sup_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[0,0])
ax[0,0].legend()
ax[0,0].set_xlabel('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')
ax[0,0].set_title('Diff. Rate')

sbn.kdeplot(masses_int_prob_drate_sup_0, label = '$\\theta = 0$', ax = ax[0,1])
sbn.kdeplot(masses_int_prob_drate_sup_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[0,1])
sbn.kdeplot(masses_int_prob_drate_sup_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[0,1])
sbn.kdeplot(masses_int_prob_drate_sup_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[0,1])
sbn.kdeplot(masses_int_prob_drate_sup_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[0,1])
ax[0,1].legend()
ax[0,1].set_xlabel('$\int_{m_{min}}^{m_{max}} P(m_{DM}|x)$')
ax[0,1].set_title('Diff. Rate')

sbn.kdeplot(masses_prob_drate_sup_0, label = '$\\theta = 0$', ax = ax[1,0])
sbn.kdeplot(masses_prob_drate_sup_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[1,0])
sbn.kdeplot(masses_prob_drate_sup_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[1,0])
sbn.kdeplot(masses_prob_drate_sup_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[1,0])
sbn.kdeplot(masses_prob_drate_sup_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[1,0])
ax[1,0].legend()
ax[1,0].set_xlabel('$\int_{m_{min}}^{\inf} P(m_{DM}|x)$')

sbn.kdeplot(masses_prob_drate_inf_0, label = '$\\theta = 0$', ax = ax[1,1])
sbn.kdeplot(masses_prob_drate_inf_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[1,1])
sbn.kdeplot(masses_prob_drate_inf_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[1,1])
sbn.kdeplot(masses_prob_drate_inf_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[1,1])
sbn.kdeplot(masses_prob_drate_inf_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[1,1])
ax[1,1].legend()
ax[1,1].set_xlabel('$\int_{0}^{m_{max}} P(m_{DM}|x)$')

plt.savefig('../graph/O1_int_prob_distribution_drate.pdf')

# +
sigma = 1.21 # this depends on how noisy your data is, play with it!

CR_int_prob_sup_0_drate         = gaussian_filter(cross_sec_int_prob_drate_sup_0, sigma)
CR_int_prob_sup_0_drate_min     = gaussian_filter(cross_sec_int_prob_drate_sup_0 - cross_sec_int_prob_drate_sup_0_sd, sigma)
CR_int_prob_sup_0_drate_max     = gaussian_filter(cross_sec_int_prob_drate_sup_0 + cross_sec_int_prob_drate_sup_0_sd, sigma)
CR_int_prob_sup_pi_2_drate      = gaussian_filter(cross_sec_int_prob_drate_sup_pi_2, sigma)
CR_int_prob_sup_pi_2_drate_min  = gaussian_filter(cross_sec_int_prob_drate_sup_pi_2 - cross_sec_int_prob_drate_sup_pi_2_sd, sigma)
CR_int_prob_sup_pi_2_drate_max  = gaussian_filter(cross_sec_int_prob_drate_sup_pi_2 + cross_sec_int_prob_drate_sup_pi_2_sd, sigma)
CR_int_prob_sup_pi_4_drate      = gaussian_filter(cross_sec_int_prob_drate_sup_pi_4, sigma)
CR_int_prob_sup_pi_4_drate_min  = gaussian_filter(cross_sec_int_prob_drate_sup_pi_4 - cross_sec_int_prob_drate_sup_pi_4_sd, sigma)
CR_int_prob_sup_pi_4_drate_max  = gaussian_filter(cross_sec_int_prob_drate_sup_pi_4 + cross_sec_int_prob_drate_sup_pi_4_sd, sigma)
CR_int_prob_sup_mpi_2_drate     = gaussian_filter(cross_sec_int_prob_drate_sup_mpi_2, sigma)
CR_int_prob_sup_mpi_2_drate_min = gaussian_filter(cross_sec_int_prob_drate_sup_mpi_2 - cross_sec_int_prob_drate_sup_mpi_2_sd, sigma)
CR_int_prob_sup_mpi_2_drate_max = gaussian_filter(cross_sec_int_prob_drate_sup_mpi_2 + cross_sec_int_prob_drate_sup_mpi_2_sd, sigma)
CR_int_prob_sup_mpi_4_drate     = gaussian_filter(cross_sec_int_prob_drate_sup_mpi_4, sigma)
CR_int_prob_sup_mpi_4_drate_min = gaussian_filter(cross_sec_int_prob_drate_sup_mpi_4 - cross_sec_int_prob_drate_sup_mpi_4_sd, sigma)
CR_int_prob_sup_mpi_4_drate_max = gaussian_filter(cross_sec_int_prob_drate_sup_mpi_4 + cross_sec_int_prob_drate_sup_mpi_4_sd, sigma)

M_int_prob_sup_0_drate         = gaussian_filter(masses_int_prob_drate_sup_0, sigma)
M_int_prob_sup_0_drate_min     = gaussian_filter(masses_int_prob_drate_sup_0 - masses_int_prob_drate_sup_0_sd, sigma)
M_int_prob_sup_0_drate_max     = gaussian_filter(masses_int_prob_drate_sup_0 + masses_int_prob_drate_sup_0_sd, sigma)
M_prob_sup_0_drate             = gaussian_filter(masses_prob_drate_sup_0, sigma)
M_prob_inf_0_drate             = gaussian_filter(masses_prob_drate_inf_0, sigma)

M_int_prob_sup_pi_2_drate      = gaussian_filter(masses_int_prob_drate_sup_pi_2, sigma)
M_int_prob_sup_pi_2_drate_min  = gaussian_filter(masses_int_prob_drate_sup_pi_2 - masses_int_prob_drate_sup_pi_2_sd, sigma)
M_int_prob_sup_pi_2_drate_max  = gaussian_filter(masses_int_prob_drate_sup_pi_2 + masses_int_prob_drate_sup_pi_2_sd, sigma)
M_prob_sup_pi_2_drate          = gaussian_filter(masses_prob_drate_sup_pi_2, sigma)
M_prob_inf_pi_2_drate          = gaussian_filter(masses_prob_drate_inf_pi_2, sigma)

M_int_prob_sup_pi_4_drate      = gaussian_filter(masses_int_prob_drate_sup_pi_4, sigma)
M_int_prob_sup_pi_4_drate_min  = gaussian_filter(masses_int_prob_drate_sup_pi_4 - masses_int_prob_drate_sup_pi_4_sd, sigma)
M_int_prob_sup_pi_4_drate_max  = gaussian_filter(masses_int_prob_drate_sup_pi_4 + masses_int_prob_drate_sup_pi_4_sd, sigma)
M_prob_sup_pi_4_drate          = gaussian_filter(masses_prob_drate_sup_pi_4, sigma)
M_prob_inf_pi_4_drate          = gaussian_filter(masses_prob_drate_inf_pi_4, sigma)

M_int_prob_sup_mpi_2_drate     = gaussian_filter(masses_int_prob_drate_sup_mpi_2, sigma)
M_int_prob_sup_mpi_2_drate_min = gaussian_filter(masses_int_prob_drate_sup_mpi_2 - masses_int_prob_drate_sup_mpi_2_sd, sigma)
M_int_prob_sup_mpi_2_drate_max = gaussian_filter(masses_int_prob_drate_sup_mpi_2 + masses_int_prob_drate_sup_mpi_2_sd, sigma)
M_prob_sup_mpi_2_drate         = gaussian_filter(masses_prob_drate_sup_mpi_2, sigma)
M_prob_inf_mpi_2_drate         = gaussian_filter(masses_prob_drate_inf_mpi_2, sigma)

M_int_prob_sup_mpi_4_drate     = gaussian_filter(masses_int_prob_drate_sup_mpi_4, sigma)
M_int_prob_sup_mpi_4_drate_min = gaussian_filter(masses_int_prob_drate_sup_mpi_4 - masses_int_prob_drate_sup_mpi_4_sd, sigma)
M_int_prob_sup_mpi_4_drate_max = gaussian_filter(masses_int_prob_drate_sup_mpi_4 + masses_int_prob_drate_sup_mpi_4_sd, sigma)
M_prob_sup_mpi_4_drate         = gaussian_filter(masses_prob_drate_sup_mpi_4, sigma)
M_prob_inf_mpi_4_drate         = gaussian_filter(masses_prob_drate_inf_mpi_4, sigma)


# +
levels = [0.67, 0.76, 0.84, 0.9, 1] 

fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))

fig00 = ax[0,0].contourf(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['magenta'])
ax[0,0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_rate.reshape(30,30).T, levels = levels, linestyles = ':', cmap = 'inferno')
ax[0,0].contour(m_vals, cross_vals, M_prob_sup_pi_2_drate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = ['--'])
ax[0,0].contour(m_vals, cross_vals, M_prob_inf_pi_2_drate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = [(0, (3,5,1,5,1,5))])

ax[0,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-44, '$\\theta = \pi/2$')
ax[0,0].legend(loc = 'lower left')

ax[0,1].contourf(m_vals, cross_vals, CR_int_prob_sup_pi_4_drate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_drate.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)
ax[0,1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['magenta'])
ax[0,1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_rate.reshape(30,30).T, levels = 5, linestyles = '-.', cmap = 'inferno')
ax[0,1].contour(m_vals, cross_vals, M_prob_sup_pi_4_drate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = ['--'])
ax[0,1].contour(m_vals, cross_vals, M_prob_inf_pi_4_drate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = [(0, (3,5,1,5,1,5))])

ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-44, '$\\theta = \pi/4$')

ax[1,0].contourf(m_vals, cross_vals, CR_int_prob_sup_mpi_2_drate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, CR_int_prob_sup_mpi_2_drate.reshape(30,30).T, levels=levels)
ax[1,0].contour(m_vals, cross_vals, CR_int_prob_sup_mpi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['magenta'])
ax[1,0].contour(m_vals, cross_vals, M_int_prob_sup_mpi_2_rate.reshape(30,30).T, levels = 5, linestyles = '-.', cmap = 'inferno')
ax[1,0].contour(m_vals, cross_vals, M_prob_sup_mpi_2_drate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = ['--'])
ax[1,0].contour(m_vals, cross_vals, M_prob_inf_mpi_2_drate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = [(0, (3,5,1,5,1,5))])

ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-44, '$\\theta = -\pi/2$')

ax[1,1].contourf(m_vals, cross_vals, CR_int_prob_sup_0_drate.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_drate.reshape(30,30).T, levels=levels)
ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['magenta'])
ax[1,1].contour(m_vals, cross_vals, M_int_prob_sup_0_rate.reshape(30,30).T, levels = 5, linestyles = '-.', cmap = 'inferno')
ax[1,1].contour(m_vals, cross_vals, M_prob_sup_0_drate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = ['--'])
ax[1,1].contour(m_vals, cross_vals, M_prob_inf_0_drate.reshape(30,30).T, levels=levels, linewidths = 1, cmap = 'inferno', linestyles = [(0, (3,5,1,5,1,5))])

ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-44, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')


fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar    = fig.colorbar(fig00, cax = cbar_ax)
cbar.ax.set_title('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')

ax[0,0].set_ylim(1e-49, 1e-43)
ax[0,0].set_xlim(5, 1e3)

plt.savefig('../graph/O1_contours_drate_int_prob_sup.pdf')
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


prueba = torch.nn.Sequential(
          torch.nn.Conv2d(1, 10, kernel_size=5),
          torch.nn.MaxPool2d(2),
          torch.nn.ReLU(),
          torch.nn.Dropout(0.5),
          torch.nn.Conv2d(10, 20, kernel_size=5, padding=2),
          torch.nn.MaxPool2d(2),
          torch.nn.ReLU(),
          torch.nn.Dropout(0.5),
          torch.nn.Flatten()
        )


# +
# Now let's define a network that estimates all the 1D and 2D marginal posteriors
# #%class Network(swyft.SwyftModule):
#%    def __init__(self, lr = 1e-3, gamma = 1.):
#%        super().__init__()
#%        self.optimizer_init = swyft.OptimizerInit(torch.optim.Adam, dict(lr = lr, weight_decay=1e-5),
#%              torch.optim.lr_scheduler.ExponentialLR, dict(gamma = gamma))
#%        self.net = torch.nn.Sequential(
#%          torch.nn.Conv2d(1, 10, kernel_size=5),
#%          torch.nn.MaxPool2d(2),
#%          torch.nn.ReLU(),
#%          torch.nn.Dropout(0.5),
#%          torch.nn.Conv2d(10, 20, kernel_size=5, padding=2),
#%          torch.nn.MaxPool2d(2),
#%          torch.nn.ReLU(),
#%          torch.nn.Dropout(0.5),
#%          torch.nn.Flatten(),
#%          torch.nn.Linear(10580, 50),
#%          torch.nn.ReLU(),
#%          torch.nn.Dropout(0.5),
#%          torch.nn.Linear(50, 10),
#%        )
#%        marginals = ((0, 1), (0, 2), (1, 2))
#%        self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = 10, num_params = 3, varnames = 'pars_norm')
#%        self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features = 10, marginals = marginals, varnames = 'pars_norm')
#%
#%    def forward(self, A, B):
#%        img = torch.tensor(A['x'])
#%        #z   = torch.tensor(B['z'])
#%        f   = self.net(img)
#%        logratios1 = self.logratios1(f, B['z'])
#%        logratios2 = self.logratios2(f, B['z'])
#%        return logratios1, logratios2

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
checkpoint_callback     = ModelCheckpoint(monitor='val_loss', dirpath='./logs/', filename='O1_s1s2_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
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
fit = False
if fit:
    trainer_s1s2.fit(network_s1s2, dm_s1s2)
    checkpoint_callback.to_yaml("./logs/O1_s1s2.yaml") 
    ckpt_path = swyft.best_from_yaml("./logs/O1_s1s2.yaml")
else:
    ckpt_path = swyft.best_from_yaml("./logs/O1_s1s2.yaml")

# ---------------------------------------
# Min val loss value at 48 epochs. -3.31
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
    plt.savefig('../graph/O1_loss_s1s2.pdf')

# ### Let's make some inference

# +
# Let's normalize testset between 0 and 1

pars_norm = (pars_testset - pars_min) / (pars_max - pars_min)

x_norm_s1s2 = x_s1s2 = s1s2_testset[:,:-1,:-1]
# -

flag = 'disc'

# +
# First let's create some observation from some "true" theta parameters
i = 189 #np.random.randint(ntest) # 189 (disc) 455 (exc) 203 (middle)
print(i)

pars_true = pars_norm[i,:]
x_obs     = x_norm_s1s2[i,:].reshape(1,96,96)

plt.imshow(x_obs[0].T, origin = 'lower')

# +
# We have to put this "observation" into a swyft.Sample object
obs = swyft.Sample(x = x_obs)

# Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
pars_prior = np.random.uniform(low = 0, high = 1, size = (10_000, 3))
pars_prior[:,2] = np.random.normal(pars_true[2], 0.001, (len(pars_prior)))
prior_samples = swyft.Samples(z = pars_prior)

# Finally we make the inference
predictions_s1s2 = trainer_s1s2.infer(network_s1s2, obs, prior_samples)

# +
# Let's plot the results
swyft.corner(predictions_s1s2, ('pars_norm[0]', 'pars_norm[1]', 'pars_norm[2]'), bins = 200, smooth = 3)

if flag == 'exc':
    plt.savefig('../graph/cornerplot_s1s2_exc_thetaFix.pdf')
else:
    plt.savefig('../graph/cornerplot_s1s2_thetaFix.pdf')

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
cross_section_th = -49
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
plt.text(-50,2, '$m = {:.2e}$'.format(10**(pars_true[0])))
plt.text(-50,1.8, '$\sigma = {:.2e}$'.format(10**(pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1])))
plt.text(-50,1.5, '$\\theta = {:.2f}$'.format(pars_true[0]))
if flag == 'exc':
    plt.savefig('../graph/1Dposterior_s1s2_exc_' + str(i) + '_thetaFix.pdf')
else:
    plt.savefig('../graph/1Dposterior_s1s2_disc_' + str(i) + '_thetaFix.pdf')
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
# ### Let's make the contour plot

pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice(['../data/andresData/O1-slices-5vecescadatheta/theta-0/SI-slices01-theta0/'])

m_vals = np.logspace(np.min(pars_slices[:,0]), np.max(pars_slices[:,0]),30)
cross_vals = np.logspace(np.min(pars_slices[:,1]), np.max(pars_slices[:,1]),30)

# +
force = False
folders = ['../data/andresData/O1-slices-5vecescadatheta/theta-pluspidiv2/SI-slices01-pluspidiv2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-pluspidiv2/SI-slices01-pluspidiv2-v2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-pluspidiv2/SI-slices01-pluspidiv2-v3/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-pluspidiv2/SI-slices01-pluspidiv2-v4/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-pluspidiv2/SI-slices01-pluspidiv2-v5/'
         ]

cross_sec_sigmas_full       = []
cross_sec_int_prob_full     = []
cross_sec_int_prob_sup_full = []

masses_int_prob_sup_full = []
masses_prob_sup_full     = []
masses_prob_inf_full     = []

for folder in folders:
    pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice([folder])
    
    if (os.path.exists(folder + 'cross_sec_sigmas_s1s2.txt') & 
        os.path.exists(folder + 'cross_sec_int_prob_s1s2.txt') &
        os.path.exists(folder + 'cross_sec_int_prob_sup_s1s2.txt') &
        os.path.exists(folder + 'masses_int_prob_sup_s1s2.txt') &
        os.path.exists(folder + 'masses_prob_sup_s1s2.txt') &
        os.path.exists(folder + 'masses_prob_inf_s1s2.txt') 
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
        masses_prob_sup     = np.ones(len(pars_norm)) * -99
        masses_prob_inf     = np.ones(len(pars_norm)) * -99
           
        for itest in tqdm(range(len(pars_norm))):
            x_obs = x_norm_s1s2[itest, :,:]
            
            # We have to put this "observation" into a swyft.Sample object
            obs = swyft.Sample(x = x_obs.reshape(1,96,96))
            
            # Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
            pars_prior    = np.random.uniform(low = 0, high = 1, size = (10_000, 3))
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
            
            cr_th = np.argmin(np.abs(x - (-49)))
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
            masses_prob_sup[itest] = trapezoid(ratios_s1s2[m_min:], masses_pred[m_min:]) / trapezoid(ratios_s1s2, masses_pred)
            masses_prob_inf[itest] = trapezoid(ratios_s1s2[:m_max], masses_pred[:m_max]) / trapezoid(ratios_s1s2, masses_pred)

        cross_sec_sigmas_full.append(cross_sec_sigmas)
        cross_sec_int_prob_full.append(cross_sec_int_prob)
        cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
        masses_int_prob_sup_full.append(masses_int_prob_sup)
        masses_prob_sup_full.append(masses_prob_sup)
        masses_prob_inf_full.append(masses_prob_inf)
            
        np.savetxt(folder + 'cross_sec_sigmas_s1s2.txt', cross_sec_sigmas)
        np.savetxt(folder + 'cross_sec_int_prob_s1s2.txt', cross_sec_int_prob)
        np.savetxt(folder + 'cross_sec_int_prob_sup_s1s2.txt', cross_sec_int_prob_sup)
        np.savetxt(folder + 'masses_int_prob_sup_s1s2.txt', masses_int_prob_sup)
        np.savetxt(folder + 'masses_prob_sup_s1s2.txt', masses_prob_sup)
        np.savetxt(folder + 'masses_prob_inf_s1s2.txt', masses_prob_inf)
    else:
        print('pre-computed')
        cross_sec_sigmas       = np.loadtxt(folder + 'cross_sec_sigmas_s1s2.txt')
        cross_sec_int_prob     = np.loadtxt(folder + 'cross_sec_int_prob_s1s2.txt')
        cross_sec_int_prob_sup = np.loadtxt(folder + 'cross_sec_int_prob_sup_s1s2.txt')
        masses_int_prob_sup    = np.loadtxt(folder + 'masses_int_prob_sup_s1s2.txt')
        masses_prob_sup        = np.loadtxt(folder + 'masses_prob_sup_s1s2.txt')
        masses_prob_inf        = np.loadtxt(folder + 'masses_prob_inf_s1s2.txt')

        cross_sec_sigmas_full.append(cross_sec_sigmas)
        cross_sec_int_prob_full.append(cross_sec_int_prob)
        cross_sec_int_prob_sup_full.append(cross_sec_int_prob_sup)
        masses_int_prob_sup_full.append(masses_int_prob_sup)
        masses_prob_sup_full.append(masses_prob_sup)
        masses_prob_inf_full.append(masses_prob_inf)


# +
cross_section_th = -49

if len(cross_sec_int_prob_full) > 1:
    cross_sec_int_prob_s1s2_pi_2        = np.mean(np.asarray(cross_sec_int_prob_full), axis = 0)
    cross_sec_int_prob_sup_s1s2_pi_2    = np.mean(np.asarray(cross_sec_int_prob_sup_full), axis = 0)
    cross_sec_int_prob_sup_s1s2_pi_2_sd = np.std(np.asarray(cross_sec_int_prob_sup_full), axis = 0)
    masses_int_prob_sup_s1s2_pi_2       = np.mean(np.asarray(masses_int_prob_sup_full), axis = 0)
    masses_int_prob_sup_s1s2_pi_2_sd    = np.std(np.asarray(masses_int_prob_sup_full), axis = 0)
    masses_prob_sup_s1s2_pi_2           = np.mean(np.asarray(masses_prob_sup_full), axis = 0)
    masses_prob_sup_s1s2_pi_2_sd        = np.std(np.asarray(masses_prob_sup_full), axis = 0)
    masses_prob_inf_s1s2_pi_2           = np.mean(np.asarray(masses_prob_inf_full), axis = 0)
    masses_prob_inf_s1s2_pi_2_sd        = np.std(np.asarray(masses_prob_inf_full), axis = 0)
    cross_sec_sigmas_pi_2               = np.mean(np.asarray(cross_sec_sigmas_full), axis = 0)
else:
    cross_sec_int_prob_s1s2_pi_2     = cross_sec_int_prob
    cross_sec_int_prob_sup_s1s2_pi_2 = cross_sec_int_prob_sup
    masses_int_prob_sup_s1s2_pi_2    = masses_int_prob_sup
    masses_prob_sup_s1s2_pi_2        = masses_prob_sup
    masses_prob_inf_s1s2_pi_2        = masses_prob_inf

s1s2_1sigma_pi_2 = np.ones(900) * -99
s1s2_2sigma_pi_2 = np.ones(900) * -99
s1s2_3sigma_pi_2 = np.ones(900) * -99

s1s2_1sigma_pi_2[np.where(cross_sec_sigmas_pi_2[:,0] > cross_section_th)[0]] = 1
s1s2_2sigma_pi_2[np.where(cross_sec_sigmas_pi_2[:,1] > cross_section_th)[0]] = 1
s1s2_3sigma_pi_2[np.where(cross_sec_sigmas_pi_2[:,2] > cross_section_th)[0]] = 1

# +
fig, ax = plt.subplots(2,2)

sbn.kdeplot(cross_sec_int_prob_sup_s1s2_0, label = '$\\theta = 0$', ax = ax[0,0])
sbn.kdeplot(cross_sec_int_prob_sup_s1s2_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[0,0])
sbn.kdeplot(cross_sec_int_prob_sup_s1s2_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[0,0])
sbn.kdeplot(cross_sec_int_prob_sup_s1s2_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[0,0])
sbn.kdeplot(cross_sec_int_prob_sup_s1s2_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[0,0])
ax[0,0].legend()
ax[0,0].set_xlabel('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')
ax[0,0].set_title('S1-S2')

sbn.kdeplot(masses_int_prob_sup_s1s2_0, label = '$\\theta = 0$', ax = ax[0,1])
sbn.kdeplot(masses_int_prob_sup_s1s2_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[0,1])
sbn.kdeplot(masses_int_prob_sup_s1s2_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[0,1])
sbn.kdeplot(masses_int_prob_sup_s1s2_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[0,1])
sbn.kdeplot(masses_int_prob_sup_s1s2_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[0,1])
ax[0,1].legend()
ax[0,1].set_xlabel('$\int_{m_{min}}^{m_{max}} P(m_{DM}|x)$')
ax[0,1].set_title('S1-S2')

sbn.kdeplot(masses_prob_sup_s1s2_0, label = '$\\theta = 0$', ax = ax[1,0])
sbn.kdeplot(masses_prob_sup_s1s2_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[1,0])
sbn.kdeplot(masses_prob_sup_s1s2_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[1,0])
sbn.kdeplot(masses_prob_sup_s1s2_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[1,0])
sbn.kdeplot(masses_prob_sup_s1s2_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[1,0])
ax[1,0].legend()
ax[1,0].set_xlabel('$\int_{m_{min}}^{\inf} P(m_{DM}|x)$')

sbn.kdeplot(masses_prob_inf_s1s2_0, label = '$\\theta = 0$', ax = ax[1,1])
sbn.kdeplot(masses_prob_inf_s1s2_pi_2, label = '$\\theta = \\frac{\pi}{2}$', ax = ax[1,1])
sbn.kdeplot(masses_prob_inf_s1s2_pi_4, label = '$\\theta = \\frac{\pi}{4}$', ax = ax[1,1])
sbn.kdeplot(masses_prob_inf_s1s2_mpi_2, label = '$\\theta = - \\frac{\pi}{2}$', ax = ax[1,1])
sbn.kdeplot(masses_prob_inf_s1s2_mpi_4, label = '$\\theta = - \\frac{\pi}{4}$', ax = ax[1,1])
ax[1,1].legend()
ax[1,1].set_xlabel('$\int_{0}^{m_{max}} P(m_{DM}|x)$')

#plt.savefig('../graph/O1_int_prob_distribution_s1s2.pdf')

# +
sigma = 0.2 # this depends on how noisy your data is, play with it!

s1s2_1sigma_0_g     = gaussian_filter(s1s2_1sigma_0, sigma)
s1s2_1sigma_pi_2_g  = gaussian_filter(s1s2_1sigma_pi_2, sigma)
s1s2_1sigma_pi_4_g  = gaussian_filter(s1s2_1sigma_pi_4, sigma)
s1s2_1sigma_mpi_2_g = gaussian_filter(s1s2_1sigma_mpi_2, sigma)
s1s2_1sigma_mpi_4_g = gaussian_filter(s1s2_1sigma_mpi_4, sigma)

s1s2_2sigma_0_g     = gaussian_filter(s1s2_2sigma_0, sigma)
s1s2_2sigma_pi_2_g  = gaussian_filter(s1s2_2sigma_pi_2, sigma)
s1s2_2sigma_pi_4_g  = gaussian_filter(s1s2_2sigma_pi_4, sigma)
s1s2_2sigma_mpi_2_g = gaussian_filter(s1s2_2sigma_mpi_2, sigma)
s1s2_2sigma_mpi_4_g = gaussian_filter(s1s2_2sigma_mpi_4, sigma)

s1s2_3sigma_0_g     = gaussian_filter(s1s2_3sigma_0, sigma)
s1s2_3sigma_pi_2_g  = gaussian_filter(s1s2_3sigma_pi_2, sigma)
s1s2_3sigma_pi_4_g  = gaussian_filter(s1s2_3sigma_pi_4, sigma)
s1s2_3sigma_mpi_2_g = gaussian_filter(s1s2_3sigma_mpi_2, sigma)
s1s2_3sigma_mpi_4_g = gaussian_filter(s1s2_3sigma_mpi_4, sigma)

fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))
fig.subplots_adjust(hspace = 0, wspace = 0)

ax[0,0].contour(m_vals, cross_vals, s1s2_1sigma_pi_2_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[0,0].contour(m_vals, cross_vals, s1s2_2sigma_pi_2_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[0,0].contourf(m_vals, cross_vals, s1s2_3sigma_pi_2_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, s1s2_3sigma_pi_2_g.reshape(30,30).T, levels=[0])

ax[0,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
ax[0,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
ax[0,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-44, '$\\theta = \pi/2$')
#ax[0,0].legend(loc = 'lower right')

ax[0,1].contour(m_vals, cross_vals, s1s2_1sigma_pi_4_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[0,1].contour(m_vals, cross_vals, s1s2_2sigma_pi_4_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[0,1].contourf(m_vals, cross_vals, s1s2_3sigma_pi_4_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, s1s2_3sigma_pi_4_g.reshape(30,30).T, levels=[0])

ax[0,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--', label = 'XENON nT [$3\sigma$]')
ax[0,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':', label = 'XENON nT [$5\sigma$]')
ax[0,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-44, '$\\theta = \pi/4$')
ax[0,1].legend(loc = 'lower right')

#ax[1,0].contour(m_vals, cross_vals, int_prob_0.reshape(30,30).T, levels=10, linewidths = 2, zorder = 4, linestyles = '--')
ax[1,0].contour(m_vals, cross_vals, s1s2_1sigma_mpi_2_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[1,0].contour(m_vals, cross_vals, s1s2_2sigma_mpi_2_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[1,0].contourf(m_vals, cross_vals, s1s2_3sigma_mpi_2_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, s1s2_3sigma_mpi_2_g.reshape(30,30).T, levels=[0])

ax[1,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
ax[1,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
ax[1,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-44, '$\\theta = -\pi/2$')

ax[1,1].contour(m_vals, cross_vals, s1s2_1sigma_0_g.reshape(30,30).T, levels=[0], linewidths = 2, zorder = 4, linestyles = '--')
ax[1,1].contour(m_vals, cross_vals, s1s2_2sigma_0_g.reshape(30,30).T, levels=[0], linestyles = ':')
ax[1,1].contourf(m_vals, cross_vals, s1s2_3sigma_0_g.reshape(30,30).T, levels=[-1, 0, 1], alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, s1s2_3sigma_0_g.reshape(30,30).T, levels=[0])

ax[1,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
ax[1,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
ax[1,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-44, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(1e-49, 1e-43)

#plt.savefig('../graph/O1_contours_s1s2.pdf')


# +
levels = [0,0.1,0.16,0.24,0.32]

sigma = 1.41 # this depends on how noisy your data is, play with it!

CR_int_prob_0_g     = gaussian_filter(cross_sec_int_prob_0, sigma)
CR_int_prob_pi_2_g  = gaussian_filter(cross_sec_int_prob_pi_2, sigma)
CR_int_prob_pi_4_g  = gaussian_filter(cross_sec_int_prob_pi_4, sigma)
CR_int_prob_mpi_2_g = gaussian_filter(cross_sec_int_prob_mpi_2, sigma)
CR_int_prob_mpi_4_g = gaussian_filter(cross_sec_int_prob_mpi_4, sigma)

fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))
fig.subplots_adjust(hspace = 0, wspace = 0)

fig00 = ax[0,0].contourf(m_vals, cross_vals, CR_int_prob_pi_2_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_pi_2_g.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)

#ax[0,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
#ax[0,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
ax[0,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-44, '$\\theta = \pi/2$')
ax[0,0].plot(masses, s1s2_90_CL_pi2[2,:], color = 'black', linestyle = '-.', label = 'Bin. Lik. [90%]')
ax[0,0].legend(loc = 'lower left')

ax[0,1].contourf(m_vals, cross_vals, CR_int_prob_pi_4_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, CR_int_prob_pi_4_g.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)

#ax[0,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--', label = 'XENON nT [$3\sigma$]')
#ax[0,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':', label = 'XENON nT [$5\sigma$]')
#ax[0,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-44, '$\\theta = \pi/4$')

ax[1,0].contourf(m_vals, cross_vals, CR_int_prob_mpi_2_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, CR_int_prob_mpi_2_g.reshape(30,30).T, levels=levels)

#ax[1,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
#ax[1,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
#ax[1,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-44, '$\\theta = -\pi/2$')

ax[1,1].contourf(m_vals, cross_vals, CR_int_prob_0_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, CR_int_prob_0_g.reshape(30,30).T, levels=levels)

#ax[1,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
#ax[1,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
#ax[1,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-44, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(1e-49, 1e-43)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(fig00, cax=cbar_ax)
cbar.ax.set_title('$\int_{-\inf}^{\sigma_{th}} P(\sigma|x)$')

ax[0,1].plot(masses, s1s2_90_CL_pi4[2,:], color = 'black', linestyle = '-.')
ax[1,0].plot(masses, s1s2_90_CL_mpi2[2,:], color = 'black', linestyle = '-.')
ax[1,1].plot(masses, s1s2_90_CL_0[2,:], color = 'black', linestyle = '-.')

plt.savefig('../graph/O1_contours_s1s2_int_prob.pdf')

# +
sigma = 0.8 # this depends on how noisy your data is, play with it!

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
M_prob_sup_0_s1s2             = gaussian_filter(masses_prob_sup_s1s2_0, sigma)
M_prob_inf_0_s1s2             = gaussian_filter(masses_prob_inf_s1s2_0, sigma)

M_int_prob_sup_pi_2_s1s2      = gaussian_filter(masses_int_prob_sup_s1s2_pi_2, sigma)
M_int_prob_sup_pi_2_s1s2_max  = gaussian_filter(masses_int_prob_sup_s1s2_pi_2 + masses_int_prob_sup_s1s2_pi_2_sd, sigma)
M_int_prob_sup_pi_2_s1s2_min  = gaussian_filter(masses_int_prob_sup_s1s2_pi_2 - masses_int_prob_sup_s1s2_pi_2_sd, sigma)
M_prob_sup_pi_2_s1s2          = gaussian_filter(masses_prob_sup_s1s2_pi_2, sigma)
M_prob_inf_pi_2_s1s2          = gaussian_filter(masses_prob_inf_s1s2_pi_2, sigma)

M_int_prob_sup_pi_4_s1s2      = gaussian_filter(masses_int_prob_sup_s1s2_pi_4, sigma)
M_int_prob_sup_pi_4_s1s2_max  = gaussian_filter(masses_int_prob_sup_s1s2_pi_4 + masses_int_prob_sup_s1s2_pi_4_sd, sigma)
M_int_prob_sup_pi_4_s1s2_min  = gaussian_filter(masses_int_prob_sup_s1s2_pi_4 - masses_int_prob_sup_s1s2_pi_4_sd, sigma)
M_prob_sup_pi_4_s1s2          = gaussian_filter(masses_prob_sup_s1s2_pi_4, sigma)
M_prob_inf_pi_4_s1s2          = gaussian_filter(masses_prob_inf_s1s2_pi_4, sigma)

M_int_prob_sup_mpi_2_s1s2     = gaussian_filter(masses_int_prob_sup_s1s2_mpi_2, sigma)
M_int_prob_sup_mpi_2_s1s2_max = gaussian_filter(masses_int_prob_sup_s1s2_mpi_2 + masses_int_prob_sup_s1s2_mpi_2_sd, sigma)
M_int_prob_sup_mpi_2_s1s2_min = gaussian_filter(masses_int_prob_sup_s1s2_mpi_2 - masses_int_prob_sup_s1s2_mpi_2_sd, sigma)
M_prob_sup_mpi_2_s1s2         = gaussian_filter(masses_prob_sup_s1s2_mpi_2, sigma)
M_prob_inf_mpi_2_s1s2         = gaussian_filter(masses_prob_inf_s1s2_mpi_2, sigma)

M_int_prob_sup_mpi_4_s1s2     = gaussian_filter(masses_int_prob_sup_s1s2_mpi_4, sigma)
M_int_prob_sup_mpi_4_s1s2_max = gaussian_filter(masses_int_prob_sup_s1s2_mpi_4 + masses_int_prob_sup_s1s2_mpi_4_sd, sigma)
M_int_prob_sup_mpi_4_s1s2_min = gaussian_filter(masses_int_prob_sup_s1s2_mpi_4 - masses_int_prob_sup_s1s2_mpi_4_sd, sigma)
M_prob_sup_mpi_4_s1s2         = gaussian_filter(masses_prob_sup_s1s2_mpi_4, sigma)
M_prob_inf_mpi_4_s1s2         = gaussian_filter(masses_prob_inf_s1s2_mpi_4, sigma)
# -


sigmas_pi_2 = np.abs( pars_slices[:,1] - cross_sec_sigmas_pi_2[:,0] )
sigmas_pi_2[np.where(CR_int_prob_sup_pi_2_s1s2 < 0.9)[0]] = 10 

# +
levels = [0.67, 0.76, 0.84, 0.9, 1]

fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (10,10))
fig.subplots_adjust(hspace = 0, wspace = 0)

fig00 = ax[0,0].contourf(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2_max.reshape(30,30).T, levels=[0.9], linewidths = 1, zorder = 4)
ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2_min.reshape(30,30).T, levels=[0.9], linewidths = 1, zorder = 4)
# #%ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['magenta'])
# #%ax[0,0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['purple'])
#ax[0,0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_s1s2.reshape(30,30).T, levels = levels, linestyles = '-.', cmap = 'inferno')

#ax[0,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--', label = 'XENON nT [$3\sigma$]')
#ax[0,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':', label = 'XENON nT [$5\sigma$]')
ax[0,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].grid(which='both')
ax[0,0].text(3e2, 1e-44, '$\\theta = \pi/2$')
ax[0,0].plot(masses, s1s2_90_CL_pi2[2,:], color = 'black', linestyle = '-.', label = 'Bin. Lik. [90%]')
ax[0,0].legend(loc = 'lower left')

ax[0,1].contourf(m_vals, cross_vals, CR_int_prob_sup_pi_4_s1s2.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[0,1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_s1s2.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)
# #%ax[0,1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['magenta'])
# #%ax[0,1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['purple'])

#ax[0,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--', label = 'XENON nT [$3\sigma$]')
#ax[0,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':', label = 'XENON nT [$5\sigma$]')
#ax[0,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
ax[0,1].grid(which='both')
ax[0,1].text(3e2, 1e-44, '$\\theta = \pi/4$')
#ax[0,1].legend()

ax[1,0].contourf(m_vals, cross_vals, CR_int_prob_sup_mpi_2_s1s2.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,0].contour(m_vals, cross_vals, CR_int_prob_sup_mpi_2_s1s2.reshape(30,30).T, levels=levels)
# #%ax[1,0].contour(m_vals, cross_vals, CR_int_prob_sup_mpi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['magenta'])
# #%ax[1,0].contour(m_vals, cross_vals, CR_int_prob_sup_mpi_2_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['purple'])

#ax[1,0].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
#ax[1,0].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
#ax[1,0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,0].grid(which='both')
ax[1,0].text(3e2, 1e-44, '$\\theta = -\pi/2$')

ax[1,1].contourf(m_vals, cross_vals, CR_int_prob_sup_0_s1s2.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_s1s2.reshape(30,30).T, levels=levels)
# #%ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['magenta'])
# #%ax[1,1].contour(m_vals, cross_vals, CR_int_prob_sup_0_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, zorder = 4, linestyles = ':', colors = ['purple'])

#ax[1,1].plot(xenon_nt_3s[:,0], xenon_nt_3s[:,1], color = 'blue', linestyle = '--')
#ax[1,1].plot(xenon_nt_5s[:,0], xenon_nt_5s[:,1], color = 'blue', linestyle = ':')
#ax[1,1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue')
ax[1,1].grid(which='both')
ax[1,1].text(3e2, 1e-44, '$\\theta = 0$')

ax[0,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_ylabel('$\sigma [cm^{2}]$')
ax[1,0].set_xlabel('m [GeV]')
ax[1,1].set_xlabel('m [GeV]')

ax[0,0].set_ylim(1e-49, 1e-43)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(fig00, cax=cbar_ax)
cbar.ax.set_title('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')

ax[0,1].plot(masses, s1s2_90_CL_pi4[2,:], color = 'black', linestyle = '-.')
ax[1,0].plot(masses, s1s2_90_CL_mpi2[2,:], color = 'black', linestyle = '-.')
ax[1,1].plot(masses, s1s2_90_CL_0[2,:], color = 'black', linestyle = '-.')

#plt.savefig('../graph/O1_contours_s1s2_int_prob_sup.pdf')
# +
levels = [0.67, 0.76, 0.84, 0.9, 1]

color_rate  = "#d55e00"
color_drate = "#0072b2"
color_s1s2  = "#009e73"

fig, ax = plt.subplots(1,3, sharex = True, sharey = True, figsize = (12,5))
fig.subplots_adjust(hspace = 0, wspace = 0)

fig00 = ax[0].contourf(m_vals, cross_vals, sigmas_pi_2.reshape(30,30).T, levels = [0,1,5,11], zorder = 1, colors = ['orange', 'red','white'])
fig.colorbar(fig00)
ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_s1s2)
# #%ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_s1s2)
# #%ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_s1s2_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_s1s2)
ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_rate)
# #%ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_rate)
# #%ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_rate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_rate)
ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_drate)
# #%ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_drate)
# #%ax[0].contour(m_vals, cross_vals, CR_int_prob_sup_pi_2_drate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_drate)
ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_s1s2)
ax[0].contour(m_vals, cross_vals, M_prob_sup_pi_2_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
#ax[0].contour(m_vals, cross_vals, M_prob_inf_pi_2_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
# #%ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_s1s2_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
# #%ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_s1s2_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_rate)
ax[0].contour(m_vals, cross_vals, M_prob_sup_pi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
#ax[0].contour(m_vals, cross_vals, M_prob_inf_pi_2_rate.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
# #%ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_rate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
# #%ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_rate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_drate)
ax[0].contour(m_vals, cross_vals, M_prob_sup_pi_2_drate.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
#ax[0].contour(m_vals, cross_vals, M_prob_inf_pi_2_drate.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
# #%ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_drate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
# #%ax[0].contour(m_vals, cross_vals, M_int_prob_sup_pi_2_drate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)

ax[0].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]', linestyle = ':')
#ax[0].fill_between(neutrino_fog[:,0], neutrino_fog[:,1], -50, color = 'lightblue', label = '$\\nu$ fog')
ax[0].plot(masses, s1s2_90_CL_pi2[2,:], color = 'black', linestyle = ':', label = 'Bin. Lik. [90%]')
ax[0].fill_between(masses, s1s2_current_pi2[2,:], 1e-43, color = 'gray', alpha = 1, label = 'Excluded', zorder = 1)

ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].grid(which='both')
ax[0].text(3e2, 1e-44, '$\\theta = \pi/2$')
ax[0].legend(loc = 'lower left')

ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_s1s2)
# #%ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_s1s2_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_s1s2)
# #%ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_s1s2_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_s1s2)
ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_rate)
# #%ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_rate)
# #%ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_rate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_rate)
ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_drate)
# #%ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_drate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_drate)
# #%ax[1].contour(m_vals, cross_vals, CR_int_prob_sup_pi_4_drate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_drate)
ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_s1s2)
ax[1].contour(m_vals, cross_vals, M_prob_sup_pi_4_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
#ax[1].contour(m_vals, cross_vals, M_prob_inf_pi_4_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
# #%ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_s1s2_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
# #%ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_s1s2_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_rate)
ax[1].contour(m_vals, cross_vals, M_prob_sup_pi_4_rate.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
#ax[1].contour(m_vals, cross_vals, M_prob_inf_pi_4_rate.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
# #%ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_rate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
# #%ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_rate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_drate)
ax[1].contour(m_vals, cross_vals, M_prob_sup_pi_4_drate.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
#ax[1].contour(m_vals, cross_vals, M_prob_inf_pi_4_drate.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
# #%ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_drate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
# #%ax[1].contour(m_vals, cross_vals, M_int_prob_sup_pi_4_drate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)

ax[1].plot(masses, s1s2_90_CL_pi4[2,:], color = 'black', linestyle = ':')
ax[1].fill_between(masses, s1s2_current_pi4[2,:], 1e-43, color = 'black', alpha = 0.2)

ax[1].grid(which='both')
ax[1].text(3e2, 1e-44, '$\\theta = \pi/4$')

ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_s1s2)
# #%ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_s1s2_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_s1s2)
# #%ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_s1s2_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_s1s2)
ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_rate)
# #%ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_rate)
# #%ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_rate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_rate)
ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, colors = color_drate)
# #%ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_drate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_drate)
# #%ax[2].contour(m_vals, cross_vals, CR_int_prob_sup_0_drate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, colors = color_drate)
ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_s1s2)
ax[2].contour(m_vals, cross_vals, M_prob_sup_0_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
#ax[2].contour(m_vals, cross_vals, M_prob_inf_0_s1s2.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
# #%ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_s1s2_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
# #%ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_s1s2_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_s1s2)
ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_rate.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_rate)
ax[2].contour(m_vals, cross_vals, M_prob_sup_0_rate.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
#ax[2].contour(m_vals, cross_vals, M_prob_inf_0_rate.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
# #%ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_rate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
# #%ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_rate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_rate)
ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_drate.reshape(30,30).T, levels = [0.9], linewidths = 2, linestyles = '--', colors = color_drate)
ax[2].contour(m_vals, cross_vals, M_prob_sup_0_drate.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
#ax[2].contour(m_vals, cross_vals, M_prob_inf_0_drate.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
# #%ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_drate_min.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)
# #%ax[2].contour(m_vals, cross_vals, M_int_prob_sup_0_drate_max.reshape(30,30).T, levels = [0.9], linewidths = 1, linestyles = '--', colors = color_drate)

ax[2].plot(masses, s1s2_90_CL_0[2,:], color = 'black', linestyle = ':')
ax[2].fill_between(masses, s1s2_current_0[2,:], 1e-43, color = 'black', alpha = 0.2, label = 'Excluded')
ax[2].legend(loc = 'lower right')

ax[2].grid(which='both')
ax[2].text(3e2, 1e-44, '$\\theta = 0$')

ax[0].set_ylabel('$\sigma \ [cm^{2}]$')
ax[0].set_xlabel('m [GeV]')
ax[1].set_xlabel('m [GeV]')
ax[2].set_xlabel('m [GeV]')

ax[0].set_ylim(1e-49, 5e-44)
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

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
#cbar = fig.colorbar(fig00, ax = cbar_ax)

#plt.savefig('../graph/O1_contours_all_int_prob_sup.pdf')

# +
rec_indices = np.where(CR_int_prob_sup_pi_2_s1s2 > 0.9)[0]
plt.scatter(pars_slices[rec_indices,1], np.abs( pars_slices[rec_indices,1] - cross_sec_sigmas_pi_2[rec_indices,2] ), 
            c = CR_int_prob_sup_pi_2_s1s2[rec_indices],vmin = p_min, vmax = p_max)
cbar = plt.colorbar()

norec_indices = np.where(CR_int_prob_sup_pi_2_s1s2 < 0.9)[0]
plt.scatter(pars_slices[norec_indices,1], np.abs( pars_slices[norec_indices,1] - cross_sec_sigmas_pi_2[norec_indices,2] ), 
            c = CR_int_prob_sup_pi_2_s1s2[norec_indices],vmin = p_min, vmax = p_max, cmap = 'inferno')
# -

# # Some other plots


m_vals = np.logspace(np.min(pars_slices[:,0]), np.max(pars_slices[:,0]),30)
cross_vals = np.logspace(np.min(pars_slices[:,1]), np.max(pars_slices[:,1]),30)

pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice(['../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2/'])

m_vals[15]

np.where(np.round(pars_slices[:,0], 4) == np.round(np.log10(m_vals[15]), 4))[0]

# +
folders = ['../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v3/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v4/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v5/'
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
    vals = sorted(swyft.plot.plot2.get_HDI_thresholds(h1, cred_level=[0.68268, 0.95450, 0.99730]))
    
    edges = torch.linspace(v.min(), v.max(), bins + 1)
    x     = np.array((edges[1:] + edges[:-1]) / 2) * (pars_max[1] - pars_min[1]) + pars_min[1]

    low_1sigma = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
    up_1sigma = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
    
    cr_th    = np.argmin(np.abs(x - (-49)))
    int_prob = trapezoid(h1[cr_th:],x[cr_th:]) / trapezoid(h1,x)
    
    levels = [0.67, 0.76, 0.84, 0.9, 1]
    
    sigma = 1.41 # this depends on how noisy your data is, play with it!
    
    int_prob_sup_pi_2_g = gaussian_filter(int_prob_sup_pi_2, sigma)
    
    fig, ax = plt.subplots(1,2, figsize = (12,5))

    if int_prob > 0.9:
        im = plot_1dpost(x,h1, ax[0], low_1sigma, up_1sigma)
    else:
        im = plot_1dpost(x,h1, ax[0])
    ax[0].text(0.55, 0.9, 'm = {:.2e} [GeV]'.format(10**pars_true[0]), transform = ax[0].transAxes)
    ax[0].text(0.55, 0.8, '$\sigma$ = {:.2e} [$cm^2$]'.format(10**pars_true[1]), transform = ax[0].transAxes)
    ax[0].text(0.55, 0.7, 'Int. Prob = {:.2f}'.format(int_prob), transform = ax[0].transAxes)
    
    fig00 = ax[1].contourf(m_vals, cross_vals, int_prob_sup_pi_2_g.reshape(30,30).T, levels=levels, alpha = 0.6, zorder = 1)
    ax[1].contour(m_vals, cross_vals, int_prob_sup_pi_2_g.reshape(30,30).T, levels=levels, linewidths = 2, zorder = 4)
    
    
    ax[1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
    ax[1].scatter(10**(pars_true[0]), 10**(pars_true[1]), c = 'red', marker = '*')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].grid(which='both')
    ax[1].text(3e2, 1e-44, '$\\theta = \pi/2$')
    ax[1].plot(masses, s1s2_90_CL_pi2, color = 'black', linestyle = '-.', label = 'Bin. Lik. [90%]')
    ax[1].legend(loc = 'lower left')
    
    ax[1].set_ylabel('$\sigma [cm^{2}]$')
    ax[0].set_ylabel('$P(\sigma|x)$')
    
    ax[1].set_ylim(1e-49, 1e-43)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(fig00, cax=cbar_ax)
    cbar.ax.set_title('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')
    
    plt.savefig('../graph/gif_plot_m_84/' + str(c) + '.jpg')    
# +
folders = ['../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v3/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v4/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-minuspidiv2/SI-slices01-minuspidiv2-v5/'
         ]

pars_slices, rate_slices, diff_rate_slices, s1s2_slices = read_slice([folders[0]])
pars_norm = (pars_slices - pars_min) / (pars_max - pars_min)
    
x_norm_s1s2 = x_s1s2 = s1s2_slices[:,:-1,:-1]
    
bps_ind = np.where(np.round(pars_slices[:,0], 4) == np.round(np.log10(m_vals[15]), 4))[0]
c = 0

fig, ax = plt.subplots(1,1, figsize = (7,7))

for itest in bps_ind:
    c = c + 1
    print(c)
    x_obs = x_norm_s1s2[itest, :,:]
    
    # We have to put this "observation" into a swyft.Sample object
    obs = swyft.Sample(x = x_obs.reshape(1,96,96))
    
    # Then we generate a prior over the theta parameters that we want to infer and add them to a swyft.Sample object
    pars_true = pars_slices[itest,:]
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

    low_1sigma = np.min(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
    up_1sigma  = np.max(x[np.where(np.array(h1) > np.array(vals[2]))[0]])
    
    cr_th    = np.argmin(np.abs(x - (-49)))
    int_prob = trapezoid(h1[cr_th:] * (10**x[cr_th:]), x[cr_th:]) / trapezoid(h1 * (10**x), x)
    
    print(int_prob)
    if int_prob > 0.9:
        im = plot_1dpost(x, h1 * (10**x), ax, color = 'black', alpha = 0.3, real_val = False)
    else:
        im = plot_1dpost(x, h1 * (10**x), ax, color = 'red', alpha = 0.3, real_val = False)
        

ax.set_ylabel('$P(\sigma|x)$')
ax.set_xlabel('$Log_{10}(\sigma)$')
custom_lines = [Line2D([0],[0], color = 'black', label = 'Rec'), 
                Line2D([0],[0], color = 'red', label = 'Non-Rec')]
ax.legend(handles = custom_lines, loc = 'upper left')

#plt.savefig('../graph/plot_1Dposteriors_m_84.pdf') 
# -
x[np.where(np.array(h1) > np.array(vals[2]))[0]]

fig, ax = plt.subplots(1)
plot_1dpost(x, h1 * (10**x), ax, color = 'black', alpha = 0.3, real_val = False)

# +
folders = ['../data/andresData/O1-slices-5vecescadatheta/theta-pluspidiv2/SI-slices01-pluspidiv2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-pluspidiv2/SI-slices01-pluspidiv2-v2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-pluspidiv2/SI-slices01-pluspidiv2-v3/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-pluspidiv2/SI-slices01-pluspidiv2-v4/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-pluspidiv2/SI-slices01-pluspidiv2-v5/'
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
    # #%predictions_s1s2 = trainer_rate.infer(network_rate, obs, prior_samples)    
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

    #print(M_int_prob_sup_pi_2_rate[itest])
    #print(masses_int_prob_sup)
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

    ax[1].plot(xenon_nt_90cl[:,0], xenon_nt_90cl[:,1], color = 'blue', label = 'XENON nT [90%]')
    ax[1].scatter(10**(pars_true[0]), 10**(pars_true[1]), c = 'red', marker = '*')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].grid(which='both')
    ax[1].text(3e2, 1e-44, '$\\theta = \pi/2$')
    ax[1].plot(masses, s1s2_90_CL_pi2[2,:], color = 'black', linestyle = '-.', label = 'Bin. Lik. [90%]')
    ax[1].legend(loc = 'lower left')
    
    ax[1].set_ylabel('$\sigma [cm^{2}]$')
    ax[0].set_ylabel('$P(m|x)$')
    ax[0].set_xlabel('$log(m)$')

    ax[0].set_xlim(0, 3)
    ax[1].set_ylim(1e-43, 1e-3)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(fig00, cax=cbar_ax)
    cbar.ax.set_title('$\int_{\sigma_{th}}^{\inf} P(\sigma|x)$')
    
    plt.savefig('../graph/O1_Mass_gif_plot_m_84/' + str(c) + '_s1s2.jpg')  

os.system('cd ../graph/ ; convert -delay 60 -loop 0 {1..30}_s1s2.jpg O1_Mass_m_84_1Dposteriors_s1s2.gif')
# -

os.system('convert -delay 60 -loop 0 ../graph/O1_Mass_gif_plot_m_84/{0..30}_s1s2.jpg ../graph/O1_Mass_gif_plot_m_84/O1_Mass_m_84_1Dposteriors_all.gif')

# # Paper plots

# ## Figure 1

linestyle = ['solid','--',':','-.']

# +
folders = ['../data/andresData/O1-slices-5vecescadatheta/theta-0/SI-slices01-theta0/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-0/SI-slices01-theta0-v2/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-0/SI-slices01-theta0-v3/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-0/SI-slices01-theta0-v4/',
           '../data/andresData/O1-slices-5vecescadatheta/theta-0/SI-slices01-theta0-v5/'
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

#ax[0,0].text(-50,7, '$m_{DM} = $' + '{:.2e}'.format(10**m_true[0]))
ax[0,0].text(-50,7, '$\sigma = $' + '{:.2e}'.format(10**sigma_true[0]))
#ax[0,1].text(-50,7, '$m_{DM} = $' + '{:.2e}'.format(10**m_true[1]))
ax[0,1].text(-50,7, '$\sigma = $' + '{:.2e}'.format(10**sigma_true[1]))
#ax[1,0].text(-50,7, '$m_{DM} = $' + '{:.2e}'.format(10**m_true[2]))
ax[1,0].text(-50,7, '$\sigma = $' + '{:.2e}'.format(10**sigma_true[2]))
#ax[1,1].text(-50,7, '$m_{DM} = $' + '{:.2e}'.format(10**m_true[3]))
ax[1,1].text(-50,7, '$\sigma = $' + '{:.2e}'.format(10**sigma_true[3]))

ax[0,0].legend(loc = 'upper right')


ax[0,0].axvline(x = sigma_true[0], color = 'black')
ax[0,1].axvline(x = sigma_true[1], color = 'black')
ax[1,0].axvline(x = sigma_true[2], color = 'black')
ax[1,1].axvline(x = sigma_true[3], color = 'black')

ax[0,0].set_ylabel('$P(\sigma|x)$')
ax[1,0].set_ylabel('$P(\sigma|x)$')
ax[1,0].set_xlabel('$\log_{10}(\sigma)$')
ax[1,1].set_xlabel('$\log_{10}(\sigma)$')

plt.savefig('../graph/PosteriorsExamples_fixSigma.pdf')

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

plt.text(-44.5,4.3, '$\sigma = $' + '{:.2e}'.format(10**sigma_true[1]))

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

plt.savefig('../graph/PosteriorsExamples_varSigma.pdf')
# -




