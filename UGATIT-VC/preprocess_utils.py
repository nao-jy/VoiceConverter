import librosa
import librosa.display
import numpy as np
import configparser
from tqdm import tqdm
import matplotlib.pyplot as plt
from stft import TacotronSTFT
import glob
import torch

# config
config_ini = configparser.ConfigParser()
config_ini.read('config.ini', encoding='utf-8')
sr = int(config_ini["DEFAULT"]["sampling_rate"])
window_size = int(config_ini["DEFAULT"]["window_size"])
shift_size = int(config_ini["DEFAULT"]["shift_size"])
n_mels = int(config_ini["DEFAULT"]["n_mels"])
fmax = int(config_ini["DEFAULT"]["fmax"])

#TODO: config
stft = TacotronSTFT(mel_fmax=8000)

def read_wav_np(path):
    wav, _ = librosa.load(path, sr=sr)

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    if np.max(wav) > 1:
        wav = wav / abs(np.max(wav))
    
    if np.min(wav) < -1:
        wav = wav / abs(np.min(wav))

    return wav

#Load all wav file
#return value will be aligned between -1 to 1
def load_wavs(wav_dir):
    wavs = list()
    files = glob.glob(wav_dir + "/**.wav")
    for file in tqdm(files):
        wav = read_wav_np(file)
        wav = torch.from_numpy(wav).unsqueeze(0)
        wavs.append(wav)
    return wavs


def wavs2Mels(wavs):
    mels = list()
    for wav in tqdm(wavs):
        mel = stft.mel_spectrogram(wav)
        mels.append(mel)
    return mels

if __name__ == "__main__":
    wavs = load_wavs("target")
    mels = wavs2Mels(wavs)

    print(mels[0].shape)
    print(mels[1].shape)
