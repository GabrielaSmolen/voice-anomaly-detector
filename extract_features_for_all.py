from os import listdir
import feature_extraction as fe
import librosa
from os.path import isfile, join
import pandas as pd
import numpy as np
import tqdm


def feature_extraction(audio_path):
    x, sr = librosa.load(audio_path)
    signal_zero_crossings = fe.zero_crossing(x)
    spectral_centroid = fe.spectral_centroid(x, sr)
    spectral_rolloff = fe.spectral_rolloff(x, sr)
    centroid_auc = fe.get_auc(spectral_centroid)
    rolloff_auc = fe.get_auc(spectral_rolloff)
    centroid_mean = fe.mean(spectral_centroid)
    rolloff_mean = fe.mean(spectral_rolloff)
    centroid_std = fe.std(spectral_centroid)
    rolloff_std = fe.std(spectral_rolloff)
    centroid_ptp_value = fe.max_ptp_value(spectral_centroid)
    rolloff_ptp_value = fe.max_ptp_value(spectral_rolloff)
    signal_mfcc = fe.mfcc(x, sr)
    signal_mfcc_shape = signal_mfcc.shape
    signal_mfcc_mean = fe.mfcc_means(signal_mfcc)
    return '', signal_zero_crossings, centroid_auc, rolloff_auc, centroid_mean, rolloff_mean, centroid_std,\
           rolloff_std, centroid_ptp_value, rolloff_ptp_value, signal_mfcc_mean, signal_mfcc_shape


if __name__ == '__main__':
    root_healthy = 'data/healthy'
    root_unhealthy = 'data/unhealthy'
    files_healthy = [join(root_healthy, file) for file in listdir(root_healthy) if isfile(join(root_healthy, file))]
    columns = ['ID', 'Zero crossings', 'Centroid AUC', 'Rolloff AUC', 'Centroid mean', 'Rolloff mean', 'Centroid STD',
              'Rolloff STD', 'Centroid p-t-p value', 'Rolloff p-t-p value', 'MFCC means', "MFCC shape", 'Label']
    df = pd.DataFrame(columns=columns)
    for file in tqdm.tqdm(files_healthy):
        feats = feature_extraction(file)
        label = 'healthy'
        feats = list(feats)
        feats.append(label)
        df2 = pd.DataFrame([feats], columns=columns)
        df = df.append(df2)

    files_unhealthy = [join(root_unhealthy, file) for file in listdir(root_unhealthy) if isfile(join(root_unhealthy, file))]

    for file in tqdm.tqdm(files_unhealthy):
        feats = feature_extraction(file)
        label = 'unhealthy'
        feats = list(feats)
        feats.append(label)
        df2 = pd.DataFrame([feats], columns=columns)
        df = df.append(df2)

    df.to_csv(join('data', 'features.csv'), index=False)
