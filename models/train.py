import numpy as np
from keras import Input
from keras.optimizers import Adam
from models.model import UNet
from hparams import *
from librosa.util import find_files


def load_npz(target=None, first=None, subset='train'):
    npz_files = find_files(f'../musdb_npz/{subset}', ext='npz')[:first]
    for file in npz_files:
        npz = np.load(file)
        assert npz['mix'].shape == npz[target].shape
        yield npz['mix'], npz[target]


def sampling(mix_mag, target_mag):
    X, y = [], []
    for mix, target in zip(mix_mag, target_mag):
        starts = np.random.randint(0, mix.shape[1] - PATCH_SIZE[0], (mix.shape[1] - PATCH_SIZE[0]) // SAMPLE_STRIDE)
        for start in starts:
            end = start + PATCH_SIZE[0]
            X.append(mix[1:, start:end, np.newaxis])
            y.append(target[1:, start:end, np.newaxis])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


if __name__ == '__main__':
    mix_mag, target_mag = zip(*load_npz(target='accompaniment', first=-1))
    model = UNet()
    model.compile(optimizer="adam", loss="mean_absolute_error")
    for e in range(100):
        print(f'Epoch {e}')
        X, y = sampling(mix_mag, target_mag)
        model.fit(X, y, batch_size=BATCH[0], verbose=1, validation_split=0.01)
        model.save('vocals_ln.h5', overwrite=True)
