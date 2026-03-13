import os
import pandas as pd
import numpy as np
import mne
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

datapath = os.path.expanduser('~\projects\phd\hybrids\hybrids_eeg\data')

code_list = {'stim000':'flowers-dogs', 'stim001':'armchairs-butterflies', 'stim002':'armchairs-chickens',
            'stim003':'fruits-beetles', 'stim004':'fruits-lizards', 'stim005':'fruits-butterflies',
            'stim006':'fruits-parrots', 'stim007': 'bottles-chickens', 'stim008':'pillows-beetles',
            'stim009': 'handbags-lizards', 'stim010': 'stim010-animate-parrots-fruits', 
            'stim011':'stim011-animate-parrots-fruits', 'stim012': 'chickens-pillows', 
            'stim013':'chickens-armchairs', 'stim014': 'birds-bottles', 
            'stim015': 'birds-fruits', 'stim016': 'fish-pillows', 
            'stim017': 'butterflies-handbags', 'stim018':'beetles-fruits', 'stim019': 'peacocks-flowers'} #only hybrids





for participant in range(1,32):
    subjectnr = "%02i" % participant
    print('Processing subject:',subjectnr)
    eegfn = f'{datapath}/derivatives/results/sub-{subjectnr}_predict.csv'
    behavfn = f'{datapath}/derivatives/results/behavioural_categorisation_stim.csv'

    if os.path.exists(eegfn) and os.path.exists(behavfn):
        eeg = pd.read_csv(eegfn)
        behav = pd.read_csv(behavfn)
    
        animate_corr = {} 
        inanimate_corr = {} 

        timepoints_list = None

        eeg_anim_dict = {}
        eeg_inanim_dict = {}
        behav_anim_dict = {}
        behav_inanim_dict = {}

        for keys, val in code_list.items():
            corrfn = f'{datapath}/derivatives/results/correlations/sub-{subjectnr}_{val}_correlations_spearman.csv'

            if eeg.empty or behav.empty:
                print(f"Skipping subject {subjectnr} for keys {val} due to empty data.")
                continue
            if timepoints_list is None:
                timepoints_list = eeg['time'].values

            animacy = {
                description: 'inanimate' if key.startswith('stim00') else 'animate'
                for key, description in code_list.items()
                if key.startswith('stim00') or key.startswith('stim01')
            }
            behav['animacy'] = behav['stimulus'].map(animacy)

            behav_row = behav[behav['stimulus'] == val]
            print("DEBUG behav_row:", behav_row)
            if not behav_row.empty:
                animacy_value = behav_row['mean_clear'].values[0] if behav_row['animacy'].values[0] == 'animate' else np.nan
                inanimacy_value = behav_row['mean_clear'].values[0] if behav_row['animacy'].values[0] == 'inanimate' else np.nan

                animate_cols = [g for g in eeg.columns if f'mean_{keys}' in g and 'stim01' in keys]
                inanimate_cols = [g for g in eeg.columns if f'mean_{keys}' in g and 'stim00' in keys]
                if animate_cols:
                    eeg_anim_dict[val] = eeg[animate_cols].values[:, 0]
                elif inanimate_cols:
                    eeg_inanim_dict[val] = eeg[inanimate_cols].values[:, 0]

                eeg_anim_dict['time'] = timepoints_list
                eeg_inanim_dict['time'] = timepoints_list
                behav_anim_dict[val] = animacy_value
                behav_inanim_dict[val] = inanimacy_value


    #print("DEBUG", eeg_data_dict)
    common_stimuli_anim = set(eeg_anim_dict.keys()) & set(behav_anim_dict.keys())
    common_stimuli_inanim = set(eeg_inanim_dict.keys()) & set(behav_inanim_dict.keys())
    common_stimuli = common_stimuli_anim.union(common_stimuli_inanim)
    #common_stimuli = [s for s in common_stimuli if not (np.isnan(behav_anim_dict.get(s, np.nan)) or np.isnan(behav_inanim_dict.get(s, np.nan)))]
    print("DEBUG common_stimuli:", common_stimuli)
    if len(common_stimuli) < 2:
        #print(f"Not enough common stimuli with valid data for subject {subjectnr}. Skipping correlation calculation.")
        continue
    print(f'Calculating overall correlations for subject {subjectnr} across {len(common_stimuli)} stimuli.')

    for timepoint in timepoints_list:
        eeg_vector_anim = []
        eeg_vector_inanim = []
        behav_vector_anim = []
        behav_vector_inanim = []

        for stimulus in common_stimuli:
            time_idx = np.where(eeg_anim_dict['time'] == timepoint)[0]
            if len(time_idx) > 0:
                if stimulus in eeg_anim_dict:
                    print(f'got inside eeg_anim')
                    eeg_anim = eeg_anim_dict[stimulus][time_idx[0]]
                    eeg_vector_anim.append(eeg_anim)
                    behav_vector_anim.append(behav_anim_dict[stimulus])
                elif stimulus in eeg_inanim_dict:
                    print(f'got inside eeg_inanim')
                    eeg_inanim = eeg_inanim_dict[stimulus][time_idx[0]]
                    eeg_vector_inanim.append(eeg_inanim)
                    behav_vector_inanim.append(behav_inanim_dict[stimulus])

            print('behavior vector anim:', behav_vector_anim, 'behavior vector inanim:', behav_vector_inanim)
        
        if len(behav_vector_inanim) >= 2:
            anim_corr, anim_p = spearmanr(eeg_vector_anim, behav_vector_anim)
            inanim_corr, inanim_p = spearmanr(eeg_vector_inanim, behav_vector_inanim)
            
            animate_corr[timepoint] = (anim_corr, anim_p)
            inanimate_corr[timepoint] = (inanim_corr, inanim_p)

            animate_corr_df = pd.DataFrame([
                    {'timepoint': tp, 'correlation': corr[0], 'p_value': corr[1]} 
                    for tp, corr in animate_corr.items()])
            inanimate_corr_df = pd.DataFrame([{'timepoint': tp, 'correlation': corr[0], 'p_value': corr[1]} for tp, corr in inanimate_corr.items()])

    animate_corrfn = f'{datapath}/derivatives/results/correlations/sub-{subjectnr}_animate_correlations_spearman.csv'
    inanimate_corrfn = f'{datapath}/derivatives/results/correlations/sub-{subjectnr}_inanimate_correlations_spearman.csv'

    animate_corr_df.to_csv(animate_corrfn, index=False)
    inanimate_corr_df.to_csv(inanimate_corrfn, index=False)