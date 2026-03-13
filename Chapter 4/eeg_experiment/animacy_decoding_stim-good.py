#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Nov 18 2024

@author: almudena
'''


#%%

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge,LinearRegression
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVR,LinearSVC

import mne
from mne.decoding import (
    SlidingEstimator,
    cross_val_multiscore,
)

datapath = os.path.expanduser('~\projects\phd\hybrids\hybrids_eeg\data')

#%%

code_list = {'stim000':'flowers-dogs', 'stim001':'armchairs-butterflies', 'stim002':'armchairs-chickens',
            'stim003':'fruits-beetles', 'stim004':'fruits-lizards', 'stim005':'fruits-butterflies',
            'stim006':'fruits-parrots', 'stim007': 'bottles-chickens', 'stim008':'pillows-beetles',
            'stim009': 'handbags-lizards', 'stim010': 'stim010-animate-parrots-fruits', 
            'stim011':'stim011-animate-parrots-fruits', 'stim012': 'chickens-pillows', 
            'stim013':'chickens-armchairs', 'stim014': 'birds-bottles', 
            'stim015': 'birds-fruits', 'stim016': 'fish-pillows', 
            'stim017': 'butterflies-handbags', 'stim018':'beetles-fruits', 'stim019': 'peacocks-flowers'} #only hybrids


for participant in range(8,32):
    subjectnr = '%02i' % participant
    print('Processing subject:',subjectnr)

    infn = f'{datapath}/derivatives/mne/sub-{subjectnr}_mne_epo.fif'
    behavfn = f'{datapath}/sub-{subjectnr}/eeg/sub-{subjectnr}_task-fix_events.tsv'
    fig1fn = f'{datapath}/derivatives/results/figures/sub-{subjectnr}.png'
    outfn = f'{datapath}/derivatives/results/sub-{subjectnr}_predict.csv'

    if os.path.exists(infn) and os.path.exists(behavfn):
        epochs = mne.read_epochs(infn)
        avg = epochs.average()
        p=avg.plot_joint(show=0)
        p.savefig(fig1fn)

        T = pd.read_csv(behavfn,delimiter='\t')
        X = epochs.get_data()
        Y = pd.DataFrame({'time':epochs.times})
        T['hybrid'] = T['stim'].str.contains('stim0').astype(int) # 0 is control, 1 is hybrid
        T['animate_hsf'] = T['stim'].str.contains('animate').astype(int) # 0 is inanimate, 1 is animate
        T['train'] = 0
        train_mask = [control == 0 for control in T['hybrid']] #or T['hybrid'] == 0 directly
        T.loc[train_mask, 'train'] = 1

        y = ['-animate-' in x for x in T['stim']]
        y = np.asarray(y)

        for i in np.unique(T['sequencenumber']):
            trainIndices = T[list(map(all,zip(T['sequencenumber']!=i, T['train'] == 1)))].index.values.astype(int)
            trainIndices = np.asarray(trainIndices, dtype=int)

            y_train = y[trainIndices] #used for the scores

            clf = make_pipeline(LinearDiscriminantAnalysis(priors=(1+0*np.unique(y))/len(np.unique(y)))) #TODO tried prior =None or prior = [.5,.5] (doesnt make any difference)
            time_decod = SlidingEstimator(clf, n_jobs=1, scoring='balanced_accuracy', verbose=0) #solution to pickling-> n_jobs =-1-> n_jobs=1; try (n_jobs=1, prefer='threads') next??
            time_decod.fit(X[trainIndices], y_train) #TODO trying [y[i] for i in trainIndices] instead of y[trainIndices]????


            for key, value in code_list.items():
                print('Keys:', key, 'Values:', value)
                T['test'] = T['stim'].str.contains(key).astype(int)

                testIndices = T[list(map(all,zip(T['sequencenumber']==i, T['test'] == 1)))].index.values.astype(int)
                print('DEBUG: X.shape', X.shape, 'len(y)', len(y))
                testIndices  = np.asarray(testIndices, dtype=int)

                if trainIndices.size == 0 or testIndices.size == 0:
                    print(f'Skipping sequence {i}: train {trainIndices.size}, test {testIndices.size}') #solution to 'ValueError: X and y must have the same length'
                    continue
                y_test  = y[testIndices] #used for the scores

                predict = time_decod.predict(X[testIndices])
                Y[f'decoding_{key}_{i}'] = predict.mean(axis=0) #getting mean over trials


            #print('DEBUG UNIQUE Y':', np.unique(y))


        for key, value in code_list.items():
            fig2fn = f'{datapath}/derivatives/results/figures/sub-{subjectnr}_{value}_predict.png'

            Y[f'mean_{key}'] = Y[[col for col in Y.columns if f'decoding_{key}_' in col]].mean(axis=1)

            fig, ax = plt.subplots()
            ax.plot(epochs.times, Y[f'mean_{key}'], label=f'animacy {value} decoding')
            ax.axhline(1/len(np.unique(y)), color='k', linestyle='--', label='chance')
            ax.set_xlabel('Time (s) relative to stimulus onset')
            ax.set_ylabel('LSF <- prediction-> HSF')
            ax.legend()
            ax.axvline(0.0, color='k', linestyle='-')
            ax.set_title('Hybrids images prediction')
            #ax.set_ylim(overall_y_min, overall_y_max)
            ax.set_xlim(-0.1, 0.8)
            fig.savefig(fig2fn)

        Y.to_csv(outfn)

    print(f'Control decoding participant {subjectnr} done')
