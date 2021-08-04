from os import listdir
import feature_extraction as fe
import librosa
from os.path import isfile, join


def feature_extraction(audio_path):
    x, sr = librosa.load(audio_path)
    signal_zero_crossing = fe.zero_crossing(x)
    signal_auc = fe.get_auc(x)
    signal_mean = fe.mean(x)
    signal_std = fe.std(x)
    signal_ptp_value = fe.max_ptp_value(x)
    signal_mfcc = fe.mfcc(x, sr)
    signal_mfcc_shape = signal_mfcc.shape
    signal_mfcc_mean = fe.mfcc_mean(signal_mfcc)
    return signal_zero_crossing, signal_auc, signal_mean, signal_std,\
           signal_ptp_value, signal_mfcc_mean, signal_mfcc_shape


if __name__ == '__main__':
    root = 'data/healthy'
    files = [join(root, file) for file in listdir(root) if isfile(join(root, file))]
    for file in files:
        zero_crossing, auc, mean, std, ptp_value, mfcc_mean, mfcc_shape = feature_extraction(file)
