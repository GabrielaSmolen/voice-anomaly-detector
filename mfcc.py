import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display

audio_path = 'data/wav/1-a_l.wav'
x, sr = librosa.load(audio_path)
print(type(x), type(sr))

ipd.Audio(audio_path)


def waveplot(x, sr):
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
    return sum(zero_crossings)


waveplot = waveplot(x, sr)
spectrogram = spectrogram(x, sr)
zero_crossing = zero_crossing(x)

