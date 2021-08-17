from model import Generator, Discriminator
import torch
import preprocess_utils as utls
from make_dataset import DataSet
import pickle
import librosa
import configparser
import soundfile as sf
import time
import matplotlib.pyplot as plt
import numpy as np
from vocoder.infer import VocGANInfer
from tqdm import tqdm

# config
config_ini = configparser.ConfigParser()
config_ini.read('config.ini', encoding='utf-8')
sr = int(config_ini["DEFAULT"]["sampling_rate"])
window_size = int(config_ini["DEFAULT"]["window_size"])
shift_size = int(config_ini["DEFAULT"]["shift_size"])
n_mels = int(config_ini["DEFAULT"]["n_mels"])
fmax = int(config_ini["DEFAULT"]["fmax"])

class CycleGANInfer(object):
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator_A2B = Generator().to(self.device)
        checkPoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator_A2B.load_state_dict(state_dict=checkPoint['model_genA2B_state_dict'])
        self.generator_A2B.eval()
        self.source_mean = checkPoint['source_mean']
        self.source_std = checkPoint['source_std']
        self.target_mean = checkPoint['target_mean']
        self.target_std = checkPoint['target_std']
    
    def infer(self, mel):
        x = (mel - self.source_mean) / self.source_std
        x = torch.unsqueeze(x, 0).to(self.device).float()
        # t1 = time.time()
        # dummy_mask = torch.ones_like(x)
        mel_conv = self.generator_A2B(x)
        # t2 = time.time()
        # print(t2-t1)
        mel_conv = torch.squeeze(mel_conv, 0)
        mel_conv = (mel_conv * self.target_std) + self.target_mean
        mel_conv = torch.squeeze(mel_conv, 0)
        return mel_conv

if __name__ == "__main__":
    vc = CycleGANInfer("CycleGAN_CheckPoint")
    vocoder = VocGANInfer("vocoder/3420.pt")

    # vocoder = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    # vocoder = vocoder.remove_weightnorm(vocoder)
    # vocoder = vocoder.to('cuda')
    # vocoder.eval()

    wavs = utls.load_wavs("source")
    mels = utls.wavs2Mels(wavs)
    for i, mel in tqdm(enumerate(mels)):
        with torch.no_grad():
            mel_conv = vc.infer(mel)
            audio = vocoder.infer(mel_conv)

        # audio_numpy = audio[0].data.cpu().numpy()
        sf.write("target_test/" + str(i).zfill(3) + ".wav", audio, samplerate=22050)