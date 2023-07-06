import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def setup(figsize=(4, 3), fontsize=12):
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['figure.figsize'] = figsize


def load_acc(filename):
    df = pd.read_csv(filename, comment='#')
    return df['name'].values, df['accuracy'].values

def load_time(filename):
    df = pd.read_csv(filename, comment='#')
    return df['name'].values, df[['time-offline', 'time-online']].values

def load_comm(filename):
    df = pd.read_csv(filename, comment='#')
    return df['name'].values, df[['comm-offline', 'comm-online']].values

def break_names(names):
    '''Break long names into multiple lines.'''
    return [name.replace(' ', '\n') for name in names]

def merge_groups(*data_list):
    '''Merge a group of values into one value.'''
    return np.array([*data_list])
