import librosa
import librosa.display
import numpy as np
import os
import sys
import soundfile as sf
import configparser
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.style as ms

# config
config_ini = configparser.ConfigParser()
config_ini.read('config.ini', encoding='utf-8')
sr = int(config_ini["DEFAULT"]["sampling_rate"])
window_size = int(config_ini["DEFAULT"]["window_size"])
shift_size = int(config_ini["DEFAULT"]["shift_size"])
n_mels = int(config_ini["DEFAULT"]["n_mels"])
fmax = int(config_ini["DEFAULT"]["fmax"])

#Load all wav file
#return value will be aligned between -1 to 1
def load_wavs(wav_dir):
    wavs = list()
    for file in tqdm(os.listdir(wav_dir)):
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr=sr)
        wavs.append(wav)
    return wavs


def wavs2Mels(wavs):
    mels = list()
    for wav in tqdm(wavs):
        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=window_size, hop_length=shift_size, power=2.0, n_mels=n_mels, fmax=fmax)
        mels.append(mel)
    return mels

def Mels2wavs(mels):
    wavs = list()
    for index, mel in tqdm(enumerate(mels)):
        wav = librosa.feature.inverse.mel_to_audio(M=mel, sr=sr, n_fft=window_size, hop_length=shift_size, power=2.0, fmax=fmax)
        #normalize
        minus_min = -min(wav)
        plus_max = max(wav)
        if plus_max < minus_min:
            wav = wav / minus_min
        else:
            wav = wav / plus_max
        wavs.append(wav)
        #for debug
        #sf.write("402/" + str(index) + ".wav", wav, samplerate=sr)
    return wavs

def write_wavs(wavs, path):
    for index, wav in tqdm(enumerate(wavs)):
        sf.write(path + "/" + str(index) + ".wav", wav, samplerate=sr)

if __name__ == "__main__":
    wavs = load_wavs("401")
    mels = wavs2Mels(wavs)

    print(mels[0].shape)
    print(mels[1].shape)

    wavs = Mels2wavs(mels)
    write_wavs(wavs, "402")
