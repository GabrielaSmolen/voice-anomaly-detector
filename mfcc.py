import librosa
import matplotlib.pyplot as plt
import librosa.display
import sklearn


def audio(audio_path):
    x, sr = librosa.load(audio_path)
    return x, sr


def waveform(x, sr):
    plt.figure()
    librosa.display.waveplot(x, sr=sr)
    return plt.show()


def spectrogram(x, sr):
    x_fourier = librosa.stft(x)
    x_db = librosa.amplitude_to_db(abs(x_fourier))
    plt.figure()
    plt.subplot(1, 2, 1)
    librosa.display.specshow(x_db, sr=sr, x_axis='time', y_axis='hz')
    plt.subplot(1, 2, 2)
    librosa.display.specshow(x_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    return plt.show()


def zero_crossing(x):
    zero_crossings = librosa.zero_crossings(x, pad=False)
    return print(sum(zero_crossings))


def spectral_centroid(x, sr):
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    return spectral_centroids


def sc_waveform(spectral_centroids):
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)

    def normalize(x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis=axis)

    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_centroids), color='r')
    return plt.show()


def spectral_rolloff(x, sr):
    spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
    return spectral_rolloff


def sr_waveform(spectral_rolloff):
    frames = range(len(spectral_rolloff))
    t = librosa.frames_to_time(frames)

    def normalize(x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis=axis)

    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_rolloff), color='r')
    return plt.show()


def mfcc(x, sr):
    mfccs = librosa.feature.mfcc(x, sr=sr)
    print(mfccs.shape)
    return mfccs


if __name__ == '__main__':
    x, sr = audio('data/wav/1-a_h.wav')

    waveform = waveform(x, sr)
    spectrogram = spectrogram(x, sr)
    zero_crossing = zero_crossing(x)
    spectral_centroids = spectral_centroid(x, sr)
    sc_waveform = sc_waveform(spectral_centroids)
    spectral_rolloff = spectral_rolloff(x, sr)
    sr_waveform = sr_waveform(spectral_rolloff)
    mfccs = mfcc(x, sr)
