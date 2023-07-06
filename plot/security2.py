# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import base

base.setup()

df = pd.read_csv('security.csv', comment='#')

n = 50000

legend = ['Attack with Prediction', 'Attack with Pred. & Sign']
legend = ['RA-P', 'RA-PS']
xlabel = 'Proportion of data used in attack (%)'

# %% draw accuracy

plt.figure()
plt.plot(df['nlimit']/n*100, df[['acc-y', 'acc-ys']].values)
plt.legend(legend)
plt.xlabel(xlabel)
plt.ylabel('Accuracy (%)')
plt.ylim([25,70])
plt.tight_layout()

# %% draw data volume

# unit data for prediction and sign
udp = 12328
uds = 1019904 # store original intermediate data
uds = 1019904 / 4 # using byte to store bool

x = df['nlimit'].values
vlm_pred = udp*x / 1e6
vlm_sign = uds*x / 1e6

plt.figure()
plt.plot(df['nlimit']/n*100, np.array([vlm_pred, vlm_pred+vlm_sign]).T)
plt.yscale('log')
plt.legend(legend)
plt.xlabel(xlabel)
plt.ylabel('Data volume needed (MB)')
plt.tight_layout()

# %% draw time

x = df['nlimit'].values

# rate limit (# per hour)
rl = 100 / 3600
collect_time = x/rl

# in second
plt.figure()
plt.plot(df['nlimit']/n*100, df[['train-time-y', 'train-time-ys']].values)
plt.legend(legend)
plt.xlabel(xlabel)
plt.ylabel('Training time (s)')
plt.tight_layout()

# in hour
plt.figure()
plt.plot(df['nlimit']/n*100, df[['train-time-y', 'train-time-ys']].values/3600)
plt.legend(legend)
plt.xlabel(xlabel)
plt.ylabel('Training time (hour)')
plt.tight_layout()

plt.figure()
plt.plot(df['nlimit']/n*100, np.array([collect_time,collect_time]).T + df[['train-time-y', 'train-time-ys']].values)
#plt.yscale('log')
plt.legend(legend)
plt.xlabel(xlabel)
plt.ylabel('Total attack time (s)')
plt.tight_layout()

# %% draw one case (bar)

acclevel=60
idy = np.argmax(df['acc-y']>acclevel)
idys = np.argmax(df['acc-ys']>acclevel)

itmy = df.iloc[idy]
itmys = df.iloc[idys]

datay = np.array([itmy['nlimit'], vlm_pred[idy], itmy['train-time-y']/3600, collect_time[idy]/3600])
datays = np.array([itmys['nlimit'], vlm_pred[idys]+vlm_sign[idys], itmys['train-time-ys']/3600, collect_time[idys]/3600])


w=0.4

plt.figure()
plt.bar(np.arange(4)-w/2, datay/datay, width=w, align='center')
plt.bar(np.arange(4)+w/2, datays/datay, width=w, align='center')
plt.xticks(np.arange(4), ['# data sample', 'train data volume (MB)', 'train time (h)', 'collect time (h)'])




