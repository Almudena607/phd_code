#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 2024

@author: almudena
"""


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
def run_control_decoding(subjectnr, decode, overwrite=0):
    """
    Run decoding analysis on a single subject
    This function loads the preprocessed data and the behavioural data and runs a decoding analysis using a linear discriminant analysis.
    
    Parameters
    ----------
    subjectnr : str
        Subject number
    decode : str
        Decoding type
    overwrite : bool
        Overwrite existing files. 0 does not overwrite, 1 overwrites

    Returns
    -------
    scores : array
        The decoding scores

    Notes
    -----
    y : array  
        The target variable. In this case it is determined by the decoding type (decode parameter)
    groups : array
        The groups variable. In this case it is the block sequence number.
    cv : cross-validation generator
    scoring : str
    n_jobs : int
    verbose : int
    """

    infn = f'{datapath}/derivatives/mne/sub-{subjectnr}_mne_epo.fif'
    behavfn = f'{datapath}/sub-{subjectnr}/eeg/sub-{subjectnr}_task-fix_events.tsv'
    outfn = f'{datapath}/derivatives/results/sub-{subjectnr}_control.csv'
    fig2fn = f'{datapath}/derivatives/results/figures/sub-{subjectnr}_control_plot.png'
    
    if os.path.exists(infn) and os.path.exists(behavfn):
        if os.path.exists(outfn) and not overwrite:
            print('file exists:',outfn)
            return
        else:
            epochs = mne.read_epochs(infn)

            avg = epochs.average()
            p=avg.plot_joint(show=0)

            T = pd.read_csv(behavfn,delimiter='\t')
            #selecting the non-target epochs (we don't need this now)
            #idx = ['_invalid_' in x for x in T['stimpath']]
            #idx = [idx for idx, t in enumerate(T_all)] #we don't need to separate targets
            #T = T_all[idx]
            X = epochs.get_data() #getting data from the epochs if avoiding the targets-> [idx,:,:]
            Y = pd.DataFrame({'time':epochs.times})
            T['hybrid'] = T['stim'].str.contains('stim0').astype(int) # 0 is control, 1 is hybrid
            T['animate_hsf'] = T['stim'].str.contains('animate').astype(int) # 0 is inanimate, 1 is animate
            
            if decode == 'animacy':
                y = ['-animate-' in x for x in T['stim']]
                decode_label = 'animate'
            else:
                print('select something valid to decode')
                

            CV = []
            for i in np.unique(T['sequencenumber']):
                trainIndices = T[list(map(all,zip(T['sequencenumber']!=i, T['hybrid'] == 0)))].index.values.astype(int)
                testIndices = T[list(map(all,zip(T['sequencenumber']==i, T['hybrid'] == 0)))].index.values.astype(int)
                CV.append((trainIndices, testIndices))

            clf = make_pipeline(LinearDiscriminantAnalysis(priors=(1+0*np.unique(y))/len(np.unique(y))))
            #clf = make_pipeline(StandardScaler(),LinearModel(LinearSVC(dual="auto")))
            time_decod = SlidingEstimator(clf, n_jobs=1, scoring="balanced_accuracy", verbose=0) #DEBUG pickling-> n_jobs =-1-> n_jobs=1; try (n_jobs=1, prefer="threads") next
    
            try:
                scores = cross_val_multiscore(time_decod, X, y, cv=CV, n_jobs=1) #DEBUG pickling-> n_jobs =-1-> n_jobs=1)
                Y[f'{decode}_decoding'] = np.mean(scores, axis=0)

            except ValueError as e:
                print("Error during cross-validation:", e)
            except TypeError as e:
                print("TypeError during cross-validation:", e)


            fig, ax = plt.subplots()
            ax.plot(epochs.times, Y[f'{decode}_decoding'], label=decode_label)
            ax.axhline(1/len(np.unique(y)), color="k", linestyle="--", label="chance")
            ax.set_xlabel("Time (s) relative to stimulus onset")
            ax.set_ylabel("LSF <- prediction-> HSF")
            ax.legend()
            ax.axvline(0.0, color="k", linestyle="-")
            ax.set_title(f'Hybrids images prediction')
            #ax.set_ylim(overall_y_min, overall_y_max)
            ax.set_xlim(-0.1, 0.8)
            fig.savefig(fig2fn)

            Y.to_csv(outfn)

    print(f'Control decoding participant {subjectnr} done')        
    
    
#%%
for participant in range(0,50):
    try:
        run_control_decoding("%02i" % participant, 'animacy', 0) #participant number, control condition to decode, 0 = no overwrite
    except Exception as e:
        print(e)