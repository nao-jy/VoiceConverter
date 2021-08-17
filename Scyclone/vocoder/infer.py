from vocoder.model import ModifiedGenerator
import soundfile as sf
import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
from vocoder.stft import TacotronSTFT

def read_wav_np(path):
    wav, _ = librosa.load(path, sr=22050)

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    return wav

class VocGANInfer(object):
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = ModifiedGenerator(80, 4).to(self.device)
        checkPoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkPoint['model_g'])
        self.generator.eval(inference=True)
    
    def infer(self, mel):
        # mel = torch.from_numpy(mel).clone()
        with torch.no_grad():
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.to(self.device)
            audio = self.generator.inference(mel)

            audio = audio.squeeze(0)
            audio = audio.squeeze()
            audio = audio[:-(256*10)]
            audio = audio.cpu().detach().numpy()

                #normalize
            minus_min = -min(audio)
            plus_max = max(audio)
            if plus_max < minus_min:
                audio = audio / minus_min
            else:
                audio = audio / plus_max

            return audio

if __name__ == "__main__":
    vgan = VocGANInfer("vctk_pretrained_model_3180.pt")
    wav = read_wav_np("VOICEACTRESS100_001.wav")
    wav = torch.from_numpy(wav).unsqueeze(0)
    stft = TacotronSTFT(mel_fmax=8000)
    mel = stft.mel_spectrogram(wav)
    wav = vgan.infer(mel)
    sf.write("target_test.wav", wav, samplerate=22050)

    # fig, ax = plt.subplots()
    # S_dB = librosa.power_to_db(mel, ref=np.max)
    # img = librosa.display.specshow(S_dB, x_axis='time',
    #                         y_axis='mel', sr=22050,
    #                         fmax=8000, ax=ax)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectrogram')

    # plt.show()