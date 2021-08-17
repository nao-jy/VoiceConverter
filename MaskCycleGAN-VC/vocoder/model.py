import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ResStack(nn.Module):
    def __init__(self, channel, dilation=1):
        super(ResStack, self).__init__()

        self.block = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(dilation),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=dilation)),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )
           

        self.shortcut = nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))
            

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.block[2])
        nn.utils.remove_weight_norm(self.block[4])
        nn.utils.remove_weight_norm(self.shortcut)

class ModifiedGenerator(nn.Module):
    def __init__(self, mel_channel, n_residual_layers, ratios=[4, 4, 2, 2, 2, 2], mult=256, out_band=1):
        super(ModifiedGenerator, self).__init__()
        self.mel_channel = mel_channel

        self.start = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mel_channel, mult * 2, kernel_size=7, stride=1))
        )

        r = ratios[0]
        self.upsample_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult,
                                                    kernel_size=r * 2, stride=r,
                                                    padding=r // 2 + r % 2,
                                                    output_padding=r % 2)))
        self.res_stack_1 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])

        r = ratios[1]
        mult = mult // 2
        self.upsample_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult,
                                                    kernel_size=r * 2, stride=r,
                                                    padding=r // 2 + r % 2,
                                                    output_padding=r % 2)))
        self.res_stack_2 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])

        r = ratios[2]
        mult = mult // 2
        self.upsample_3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult,
                                                    kernel_size=r * 2, stride=r,
                                                    padding=r // 2 + r % 2,
                                                    output_padding=r % 2)))

        self.skip_upsample_1 = nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel, mult,
                                                                       kernel_size=64, stride=32,
                                                                       padding=16,
                                                                       output_padding=0))
        self.res_stack_3 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])

        r = ratios[3]
        mult = mult // 2
        self.upsample_4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult,
                                                    kernel_size=r * 2, stride=r,
                                                    padding=r // 2 + r % 2,
                                                    output_padding=r % 2)))

        self.skip_upsample_2 = nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel, mult,
                                                                       kernel_size=128, stride=64,
                                                                       padding=32,
                                                                       output_padding=0))
        self.res_stack_4 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])

        r = ratios[4]
        mult = mult // 2
        self.upsample_5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult,
                                                    kernel_size=r * 2, stride=r,
                                                    padding=r // 2 + r % 2,
                                                    output_padding=r % 2)))

        self.skip_upsample_3 = nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel, mult,
                                                                       kernel_size=256, stride=128,
                                                                       padding=64,
                                                                       output_padding=0))
        self.res_stack_5 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])


        r = ratios[5]
        mult = mult // 2
        self.upsample_6 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult,
                                                    kernel_size=r * 2, stride=r,
                                                    padding=r // 2 + r % 2,
                                                    output_padding=r % 2)))

        self.skip_upsample_4 = nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel, mult,
                                                                       kernel_size=512, stride=256,
                                                                       padding=128,
                                                                       output_padding=0))
        self.res_stack_6 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])

        self.out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mult, out_band, kernel_size=7, stride=1)),
            nn.Tanh(),
        )
        self.apply(weights_init)
    
    def forward(self, mel):
        mel = (mel + 5.0) / 5.0  # roughly normalize spectrogram
        # Mel Shape [B, num_mels, T] -> torch.Size([3, 80, 10])
        x = self.start(mel)  # [B, dim*2, T] -> torch.Size([3, 512, 10])

        x = self.upsample_1(x)
        x = self.res_stack_1(x)  # [B, dim, T*4] -> torch.Size([3, 256, 40])

        x = self.upsample_2(x)
        x = self.res_stack_2(x)  # [B, dim/2, T*16] -> torch.Size([3, 128, 160])
        # out1 = self.sub_out_1(x)  # [B, 1, T*16] -> torch.Size([3, 1, 160])

        x = self.upsample_3(x)
        x = x + self.skip_upsample_1(mel)
        x = self.res_stack_3(x)  # [B, dim/4, T*32] -> torch.Size([3, 64, 320])
        # out2 = self.sub_out_2(x)  # [B, 1, T*32] -> torch.Size([3, 1, 320])

        x = self.upsample_4(x)
        x = x + self.skip_upsample_2(mel)
        x = self.res_stack_4(x)  # [B, dim/8, T*64] -> torch.Size([3, 32, 640])
        # out3 = self.sub_out_3(x)  # [B, 1, T*64] -> torch.Size([3, 1, 640])

        x = self.upsample_5(x)
        x = x + self.skip_upsample_3(mel)
        x = self.res_stack_5(x)  # [B, dim/16, T*128] -> torch.Size([3, 16, 1280])
        # out4 = self.sub_out_4(x)  # [B, 1, T*128] -> torch.Size([3, 1, 1280])

        x = self.upsample_6(x)
        x = x + self.skip_upsample_4(mel)
        x = self.res_stack_6(x)  # [B, dim/32, T*256] -> torch.Size([3, 8, 2560])

        out = self.out(x)  # [B, 1, T*256] -> torch.Size([3, 1, 2560])

        return out #out1, out2, out3, out4, out

    def eval(self, inference=False):
        super(ModifiedGenerator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)


    def inference(self, mel):
        hop_length = 256
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(mel.device)
        mel = torch.cat((mel, zero), dim=2)

        audio = self.forward(mel)
        return audio

  
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, ndf = 16, n_layers = 3, downsampling_factor = 4, disc_out = 512):
        super(Discriminator, self).__init__()
        discriminator = nn.ModuleDict()
        discriminator["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            nn.utils.weight_norm(nn.Conv1d(1, ndf, kernel_size=15, stride=1)),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, disc_out)

            discriminator["layer_%d" % n] = nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                )),
                nn.LeakyReLU(0.2, True),
            )
        nf = min(nf * 2, disc_out)
        discriminator["layer_%d" % (n_layers + 1)] = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(nf, disc_out, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(0.2, True),
        )

        discriminator["layer_%d" % (n_layers + 2)] = nn.utils.weight_norm(nn.Conv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        ))
        self.discriminator = discriminator

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for key, module in self.discriminator.items():
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_D = 3, ndf = 16, n_layers = 3, downsampling_factor = 4, disc_out = 512):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = Discriminator(
                ndf, n_layers, downsampling_factor, disc_out
            )

        self.downsample = nn.AvgPool1d(downsampling_factor, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results

if __name__=="__main__":
    x = torch.randn(3, 80, 10)  # (B, channels, T).
    generator = ModifiedGenerator(80, 4)
    discriminator = MultiScaleDiscriminator()
    audio = generator(x)
    d = discriminator(audio)