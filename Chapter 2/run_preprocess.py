import os
import pandas as pd
import numpy as np
import difflib
import mne

def run_preprocess(subjectnr,overwrite=0):
    #%% subject to run
    print(f'preprocessing sub-{subjectnr}')
    
    # path
    datapath = os.path.expanduser('~\projects\phd\occlusion_speed')
    os.makedirs(f'{datapath}/derivatives/mne', exist_ok=True)
    
    outfn = f'{datapath}/derivatives/mne/sub-{subjectnr}_mne_epo.fif'
    behavfn = f'{datapath}/sub-{subjectnr}/eeg/sub-{subjectnr}_task-occlusion_events.tsv'
    source_behavfn = f'{datapath}/sourcedata/sub-{subjectnr}_task-occlusion_events.csv'
    source_filename = f'{datapath}/sourcedata/sub-{subjectnr}_task-occlusion_eeg.bdf'
    raw_filename = f'{datapath}/sub-{subjectnr}/eeg/sub-{subjectnr}_task-occlusion_eeg.bdf'
    
    if os.path.exists(outfn) and not overwrite:
        print('file exists:',outfn)
        return
    
    if not os.path.exists(source_behavfn):
        print('file does not exists:',source_behavfn)
        return
    
    if not (os.path.exists(source_filename) or os.path.exists(raw_filename)):
        print('file does not exists:',source_filename)
        return

    if not os.path.exists(raw_filename):
        os.makedirs(os.path.dirname(raw_filename), exist_ok=True)
        os.rename(source_filename, raw_filename)

    #%% Load EEG file
    raw = mne.io.read_raw_bdf(raw_filename, preload=True)
    sfreq = raw.info['sfreq']

    # Read events
    T = pd.read_csv(source_behavfn)

    # Find the STATUS channel and read the values from it
    stim_channel = raw.ch_names.index('Status')
    stim_data = raw.get_data(picks=stim_channel)
        
    triggertimes = 1 + np.where((np.diff(stim_data)[0]!=0))[0]
    triggervalues = stim_data[0][triggertimes]
        
    a,b = np.unique(triggervalues,return_counts=1)
    for x,y in zip(a,b):print('%d %d'%(x,y))
    
    stim_onset_sample = [x for x,y in zip(triggertimes,triggervalues) if y==40708]
    stim_offset_sample = [x for x,y in zip(triggertimes,triggervalues) if y==36612]
    [x-y for x,y in zip(stim_offset_sample,stim_onset_sample)]
        
    # fix missing triggers
    missingtriggers = len(T)-len(stim_onset_sample)
    if missingtriggers>0:
        print('reconstructing %i triggers'%(missingtriggers))
        print('triggers: %i'%len(stim_onset_sample),'expected: %i'%len(T))
        assert missingtriggers<100, 'too many missing triggers'
        a=[int(x)/sfreq for x in stim_onset_sample]
        b=[x for x in T['time_stimon']]
        for j in range(1,len(a)):
            da = a[j]-a[j-1]
            db = b[j]-b[j-1]
            if da-db > .11:
                print('inserting pos:%i +%.3fs'%(j,a[j-1]+db),'data:%.3fs_diff(%.3fs,%.3fs])'%(da,a[j-1],a[j]),
                      'expected:%.3fs_diff(%.3fs,%.3fs])'%(db,b[j-1],b[j]))
                a.insert(j,a[j-1]+db)
                stim_onset_sample.insert(j,int(round(sfreq*(a[j-1]+db))))
    if len(stim_onset_sample) > len(T):
        print('found not enough events!',len(stim_onset_sample), len(T))
        
    assert len(stim_onset_sample) <= len(T), 'found too many events!'
    assert len(stim_onset_sample) >= len(T), 'found not enough events!'
    stim_onset_sample = np.array(stim_onset_sample)
    events = np.transpose(np.vstack((stim_onset_sample,0*stim_onset_sample,range(0,len(stim_onset_sample)))));
    
    #%% Behaviour
    T2 = pd.DataFrame({
        'onset': [int(x)/sfreq for x in stim_onset_sample],
        'duration': 0.20,
        'onsetsample': [int(x) for x in stim_onset_sample],
        'eventnumber': range(0, len(stim_onset_sample)),
        'subjectnr': int(subjectnr)
        })
    T2 = pd.concat((T2,T),axis=1)
    
    assert all(np.diff(T2['onset']) - np.diff(T2['time_stimon']) < 0.11), 'event times do not seem to match'
    T2.to_csv(behavfn, sep='\t', index=False)
    T2.to_csv(behavfn.replace('.tsv','.csv'), sep=',', index=False)

    # Preprocessing
    # layout = mne.channels.read_layout('biosemi')
    # raw.set_montage()
    print(raw)
    raw.pick('eeg')
    montage = mne.channels.make_standard_montage("biosemi64")
    #raw.info.ch_names = montage.ch_names
    
    # rename A1 A2 etc to Fp1 AF7 etc
    mne.rename_channels(raw.info,dict(zip(raw.info.ch_names,montage.ch_names)))
    print(raw.info.ch_names)
    raw.set_montage(montage)
    raw.set_eeg_reference()
    raw.filter(l_freq=0.1, h_freq=100)

    # Epoch
    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, baseline=(None,0), detrend=0, proj=False, preload=True)
    
    # Resample
    epochs.resample(200)
    print(epochs)
    epochs.save(outfn,overwrite=1)

    print('Done')

#%%% Example usage to run all
for s in range(0,50):
    try:run_preprocess("%02i"%s)
    except Exception as e:print(e)