import numpy as np
from librosa.core import istft, load, stft, magphase
import soundfile as sf
import keras as keras
from models.model2 import MBBlock

if __name__ == '__main__':
    SAMPLE_RATE, WINDOW_SIZE, HOP_LENGTH = 8192, 1024, 768
    # load test audio and convert to mag/phase
    mix_wav, _ = load("../mix.wav", sr=SAMPLE_RATE)
    mix_wav_mag, mix_wav_phase = magphase(stft(mix_wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH))

    START = 0
    END = START + 128

    # load saved model

    model = keras.models.load_model('vocals_mbb.h5',
                                    custom_objects={
                                        "MBBBlock": MBBlock
                                    })
    # model = keras.models.load_model('../models/vocal_20.h5')

    # predict and write into file
    X = mix_wav_mag[1:].reshape(1, 512, 128, 1)
    y = model.predict(X, batch_size=32)

    target_pred_mag = np.vstack((np.zeros((128,)), y.reshape(512, 128)))
    print(target_pred_mag.shape)

    print(mix_wav)
    sf.write(f'acc_mbb_3.wav', istft(
        target_pred_mag * mix_wav_phase
        , win_length=WINDOW_SIZE, hop_length=HOP_LENGTH), SAMPLE_RATE)

    # sf.write(f'../wav_files/mix_downsampled.wav', istft(
    #     mix_wav_mag * mix_wav_phase
    #     , win_length=WINDOW_SIZE, hop_length=HOP_LENGTH), SAMPLE_RATE)
    #
    # sf.write(f'../wav_files/vocals_downsampled.wav', istft(
    #     vocal_wav_mag * vocal_wav_phase
    #     , win_length=WINDOW_SIZE, hop_length=HOP_LENGTH), SAMPLE_RATE)
