import librosa
import librosa.display
import numpy as np
from numpy import trapz


def zero_crossing(x):
    zero_crossings = librosa.zero_crossings(x, pad=False)
    return sum(zero_crossings)


def spectral_centroid(x, sr):
    return librosa.feature.spectral_centroid(x, sr=sr)[0]


def spectral_rolloff(x, sr):
    return librosa.feature.spectral_rolloff(x, sr=sr)[0]


def mfcc(x, sr):
    mfccs = librosa.feature.mfcc(x, sr=sr)
    print(mfccs.shape)
    return mfccs


# def mfccs(mfccs, sr):
#     return librosa.display.specshow(mfccs, sr=sr, x_axis='time')


def get_auc(array):
    auc = trapz(array, dx=5)
    return auc


def mean(array):
    return np.mean(array)


def std(array):
    return np.std(array)


def percentile(array, p):
    return np.percentile(array, p)


def max_ptp_value(array):
    max_value = np.max(array)
    min_value = np.min(array)
    value = abs(max_value-min_value)
    return value


if __name__ == '__main__':
    x, sr = librosa.load('data/wav/1-a_h.wav')

    # waveform = waveform(x, sr)
    # spectrogram = spectrogram(x, sr)
    zero_crossing = zero_crossing(x)
    spectral_centroids = spectral_centroid(x, sr)
    # sc_waveform = sc_waveform(spectral_centroids)
    spectral_rolloff = spectral_rolloff(x, sr)
    # sr_waveform = sr_waveform(spectral_rolloff, spectral_centroids)
    mfccs = mfcc(x, sr)
    # mfccs_daria = mfccs(mfccs, sr)
    auc_centroids = get_auc(spectral_centroids)
    print("AUC centroids: ", auc_centroids)
    auc_rolloff = get_auc(spectral_rolloff)
    print("AUC rolloff: ", auc_rolloff)
    mean_centroids = mean(spectral_centroids)
    print("Mean centroids: ", mean_centroids)
    mean_rolloff = mean(spectral_rolloff)
    print("Mean rolloff: ", mean_rolloff)
    std_centroids = std(spectral_centroids)
    print("STD centroids: ", std_centroids)
    std_rolloff = std(spectral_rolloff)
    print("STD rolloff: ", std_rolloff)
    percentile_25 = percentile(spectral_centroids, 25)
    print(percentile_25)
    print(np.max(spectral_centroids))
    print(np.min(spectral_centroids))
    print(max_ptp_value(spectral_centroids))
    # question
