from os import listdir
import feature_extraction as fe
import librosa
from os.path import isfile, join
import pandas as pd
import numpy as np
import tqdm


def feature_extraction(audio_path):
    x, sr = librosa.load(audio_path)
    features = np.zeros((75,))
    features[0] = audio_path.split("\\")[1].split("-")[0]
    features[1] = fe.zero_crossing(x)
    spectral_centroid = fe.spectral_centroid(x, sr)
    spectral_rolloff = fe.spectral_rolloff(x, sr)
    features[2] = fe.get_auc(spectral_centroid)
    features[3] = fe.get_auc(spectral_rolloff)
    features[4] = fe.mean(spectral_centroid)
    features[5] = fe.mean(spectral_rolloff)
    features[6] = fe.std(spectral_centroid)
    features[7] = fe.std(spectral_rolloff)
    features[8] = fe.max_ptp_value(spectral_centroid)
    features[9] = fe.max_ptp_value(spectral_rolloff)
    signal_mfcc = fe.mfcc(x, sr)
    features[10:23] = np.mean(signal_mfcc, axis=1)
    features[23:36] = fe.mfcc_delta(signal_mfcc)
    features[36:49] = fe.mfcc_delta2(signal_mfcc)

    features[49:62] = fe.mfcc_max_min(signal_mfcc[:, 6:-6])
    features[62:75] = fe.mfcc_std(signal_mfcc[:, 6:-6])
    return features


if __name__ == '__main__':
    root_healthy = 'data/healthy'
    root_unhealthy = 'data/unhealthy'

    means = [f'MFCC mean {x}' for x in range(1, 14)]
    deltas = ['MFCC delta ' + str(x) for x in range(1, 14)]
    deltas2 = ['MFCC delta 2 ' + str(x) for x in range(1, 14)]
    max_min = ['MFCC max_min ' + str(x) for x in range(1, 14)]
    std = ['MFCC std ' + str(x) for x in range(1, 14)]

    files_healthy = [join(root_healthy, file) for file in listdir(root_healthy) if isfile(join(root_healthy, file))]
    columns = ['ID', 'Zero crossings', 'Centroid AUC', 'Rolloff AUC', 'Centroid mean', 'Rolloff mean', 'Centroid STD',
              'Rolloff STD', 'Centroid p-t-p value', 'Rolloff p-t-p value'] + means + deltas + deltas2 + max_min + std + ['Label']
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
