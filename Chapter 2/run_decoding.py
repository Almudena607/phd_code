#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:40:18 2023

@author: tijl
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
    GeneralizingEstimator,
    Scaler,
    cross_val_multiscore,
    LinearModel,
    get_coef, 
    Vectorizer,
    CSP,
)

#%%

category_codes = {'animacy': 'anim', 'category': 'cat', 
                   'object': 'obj', 'image': 'im'}

for s in range(0,50):
    subjectnr='%02d'%s
    datapath = os.path.expanduser('~\projects\phd\occlusion_speed')
    os.makedirs(f'{datapath}\\derivatives\\results', exist_ok=True)
    infn = f'{datapath}\\derivatives\\mne\\sub-{subjectnr}_mne_epo.fif'
    behavfn = f'{datapath}\\sub-{subjectnr}\\eeg\\sub-{subjectnr}_task-occlusion_events.tsv'
    outfn = f'{datapath}\\derivatives\\results\\sub-{subjectnr}_results.csv'
    fig1fn = f'{datapath}\\derivatives\\results\\figures\\sub-{subjectnr}_epochs.png'
    
    if os.path.exists(infn) and not os.path.exists(outfn):
        epochs = mne.read_epochs(infn)
        
        avg = epochs.average()
        p=avg.plot_joint(show=0)
        #p.savefig(fig1fn)

        Y = pd.DataFrame({'time':epochs.times})

        print("participant:", subjectnr)

        for hz,soa in ((5,0.2), (20,0.05)):
            print("condition:", hz)
            T = pd.read_csv(behavfn,delimiter='\t')
            T['occlusion'] = T['occlusionTL'] + T['occlusionTR'] + T['occlusionBL']+ T['occlusionBR']
            idx = list(map(all,zip([not x for x in T['istarget']],[x==soa for x in T['SOA']])))
            T = T[idx].reset_index()
            X = epochs.get_data()[idx,:,:]
            
            
            #%% decoding category
            #c,y = np.unique(T['stimpath'],return_inverse=True)
            for l in category_codes:
                print("category:", l)

                obj_y = np.ceil(T['stimnumber']/4)
                cat_y = np.ceil(T['stimnumber']/20)
                anim_y = np.ceil(T['stimnumber']/100)
                im_y = T['stimnumber']

                if l == list(category_codes)[0]:
                    y = anim_y
                elif l == list(category_codes)[1]:
                    y = cat_y
                elif l == list(category_codes)[2]:
                    y = obj_y
                elif l == list(category_codes)[3]:
                    y = im_y

                for o in np.unique(T['occlusion']):
                    print("occlusion:", o)
                    CV = []
                    print(f'participant: {s}, soa: {soa}, category level: {l}, occlusion level: {o}')

                    for i in np.unique(T['sequencenumber']):
                        trainIndices = T[ list(map(all,zip(T['sequencenumber']!=i, T['occlusion']==0))) ].index.values.astype(int)
                        testIndices =  T[ list(map(all,zip(T['sequencenumber']==i, T['occlusion']==o))) ].index.values.astype(int)
                        CV.append( (trainIndices, testIndices) )
                
                    clf = make_pipeline(LinearDiscriminantAnalysis(priors=(1+0*np.unique(y))/len(np.unique(y))))
                    #clf = make_pipeline(StandardScaler(),LinearModel(LinearSVC(dual="auto")))
                    time_decod = SlidingEstimator(clf, n_jobs=-1, scoring="balanced_accuracy", verbose=0)
                    scores= cross_val_multiscore(time_decod, X, y, cv=CV, n_jobs=-1)
                    
                    # Mean scores across cross-validation splits
                    Y[f'{l}_decoding_{hz}Hz_{o}occlusion'] = np.mean(scores, axis=0)
            
                    fig, ax = plt.subplots()
                    ax.plot(epochs.times, Y[f'{l}_decoding_{hz}Hz_{o}occlusion'], label="score")
                    ax.axhline(1/len(np.unique(y)), color="k", linestyle="--", label="chance")
                    ax.set_xlabel("Times")
                    ax.set_ylabel("Accuracy")
                    ax.legend()
                    ax.axvline(0.0, color="k", linestyle="-")
                    ax.set_title(f"{l} decoding")
                    fig.savefig(fig1fn.replace('_epochs',f'{l}_decoding_{hz}Hz_{o}occlusion'))
                    
        #%%
        Y.to_csv(outfn)