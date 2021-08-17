import preprocess_utils as utls
import pickle
import torch
import configparser
import numpy as np
import random

config_ini = configparser.ConfigParser()
config_ini.read('config.ini', encoding='utf-8')
data_width = int(config_ini["DEFAULT"]["data_width"])

class DataSet:
    def __init__(self, wavs_path):

        wavs = utls.load_wavs(wav_dir=wavs_path)
        mels = utls.wavs2Mels(wavs)

        #melを繋げる
        self.data = torch.cat(mels, 2)

        #標準化
        self.mean, self.std = torch.std_mean(self.data, 2, keepdim=True)
        self.data = (self.data - self.mean) / self.std

    def getitem(self):
        start_index = random.randint(0, self.data.shape[2] - data_width - 1)
        mel = self.data[:, : ,start_index : start_index + data_width]
        # mask = np.ones(mel.shape)
        # mask_length = int(mask.shape[1] * random.randint(0, 50) * 0.01)
        # mask_zeros = np.zeros((mask.shape[0], mask_length))
        # mask_start_index = random.randint(0, mel.shape[1] - mask_length - 1)
        # mask[: ,mask_start_index : mask_start_index + mask_length] = mask_zeros
        # mel = np.multiply(mel, mask)
        # ret = np.stack([mel, mask])

        mel = torch.unsqueeze(mel, 0)
        return mel
    
    def getmask(self):
        mask = torch.ones((self.data.shape[1], data_width))
        mask_length = int(mask.shape[1] * random.randint(0, 25) * 0.01) * 2
        mask_zeros = torch.zeros((mask.shape[0], mask_length))
        # mask_start_index = random.randint(0, data_width - mask_length - 1)
        mask_start_index = int((data_width - mask_length) / 2)
        mask[: ,mask_start_index : mask_start_index + mask_length] = mask_zeros
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)
        return mask
    
if __name__ == "__main__":
    s = DataSet("source")
    t = DataSet("target")
    item = t.getitem()
    mask = t.getmask()
    # print(torch.cat((item * mask, mask), 1).tolist()[0][1])
    # with open('stats.pickle', 'wb') as f:
    #     pickle.dump([s.mean.to('cpu').detach().numpy().copy(), s.std.to('cpu').detach().numpy().copy(), t.mean.to('cpu').detach().numpy().copy(), t.std.to('cpu').detach().numpy().copy()], f) 