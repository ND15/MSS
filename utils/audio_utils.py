import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import musdb
import soundfile as sf
import numpy as np
from hparams import *


def save_files(audio, samplerate, path='../'):
    sf.write(path, data=audio,
             samplerate=samplerate)


def resample_files(path, samplerate=22050):
    x, _ = librosa.load(path, sr=samplerate)
    sf.write(path, x, samplerate=samplerate)
    print("Saving Resampled ", path)


def save_musdb_files(root='D:/Downloads/root/MUSDB18/MUSDB18-7',
                     subsets='train'):
    mus = musdb.DB(root=root, subsets=subsets)
    for i, track in enumerate(mus):
        out_path = '../musdb_resampled/' + subsets + '/{}/'.format(i)

        os.makedirs(out_path, exist_ok=True)

        mixture = track.audio[:, 0]
        vocals = track.targets['vocals'].audio[:, 0]
        accompaniment = track.targets['accompaniment'].audio[:, 0]

        save_files(mixture, 44100, out_path + 'mixture.wav')
        save_files(vocals, 44100, out_path + 'vocals.wav')
        save_files(accompaniment, 44100, out_path + 'accompaniment.wav')


def magnitude_spectrogram(audio):
    y, _ = librosa.load(audio, sr=22050)
    spectrogram = librosa.stft(y, n_fft=1024, hop_length=768)
    mag_spectrogram, _ = librosa.magphase(spectrogram)
    return mag_spectrogram.astype('float')


def save_to_npz(dir_path, subset, folder):
    mix = magnitude_spectrogram(f'{dir_path}/{subset}/{folder}/mixture.wav')
    vocals = magnitude_spectrogram(f'{dir_path}/{subset}/{folder}/vocals.wav')
    accompaniment = magnitude_spectrogram(f'{dir_path}/{subset}/{folder}/accompaniment.wav')

    mix_max = mix.max()
    mix = mix / mix_max
    vocals = vocals / mix_max
    accompaniment = accompaniment / mix_max

    if not os.path.exists(f'../musdb_npz/{subset}'):
        os.makedirs(f'../musdb_npz/{subset}')

    np.savez_compressed(
        f'../musdb_npz/{subset}/{folder}.npz',
        mix=mix, vocals=vocals, accompaniment=accompaniment
    )
    print(f"Saved {folder} of {subset}")


def process_musdb(folder, subset='train'):
    if not os.path.exists('../musdb_npz'):
        os.makedirs('../musdb_npz')

    dirs = list(os.walk(f'{folder}/{subset}/'))[0][1]

    for i in range(len(dirs)):
        save_to_npz(dir_path=f'{folder}', subset=subset,
                    folder=dirs[i])


def preprocess_musdb(folder_path, subset='train'):
    dirs = list(os.walk(f'{folder_path}/{subset}/'))[0][1]

    # resample the files
    for i in range(len(dirs)):
        resample_files(f'{folder_path}/{subset}/{dirs[i]}/mixture.wav')
        resample_files(f'{folder_path}/{subset}/{dirs[i]}/vocals.wav')
        resample_files(f'{folder_path}/{subset}/{dirs[i]}/accompaniment.wav')

    # calculate the magnitude spectrograms and save to npz
    process_musdb(folder_path, subset)


if __name__ == "__main__":
    pass
    # save_musdb_files()
    # save_musdb_files(subsets='test')
    # preprocess_musdb('../musdb_resampled')
    # preprocess_musdb('../musdb_resampled', subset='test')
