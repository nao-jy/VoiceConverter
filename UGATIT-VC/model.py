import torch.nn as nn
import torch
from torchsummary import summary
from torch.nn.parameter import Parameter

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        return x1 * torch.sigmoid(x2)

class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out

class ResnetAdaILNBlock(nn.Module):
    def __init__(self):
        super(ResnetAdaILNBlock, self).__init__()
        self.res1 = nn.Sequential(nn.Conv2d(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1, bias=False), GLU())
        self.norm1 = adaILN(256)
        self.res2 = nn.Sequential(nn.Conv2d(in_channels=256,
                                            out_channels=256,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1, bias=False))
        self.norm2 = adaILN(256)

    def forward(self, x, gamma, beta):
        out = self.res1(x)
        out = self.norm1(out, gamma, beta)
        out = self.res2(out)
        out = self.norm2(out, gamma, beta)
        return out + x

class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.res = nn.Sequential(nn.Conv2d(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1, bias=False),
                                nn.InstanceNorm2d(num_features=512,
                                                    affine=True),
                                GLU(),
                                nn.Conv2d(in_channels=256,
                                            out_channels=256,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1, bias=False),
                                nn.InstanceNorm2d(num_features=256,
                                                    affine=True))

    def forward(self, x):
        out = x + self.res(x)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # GLUはチャネル数を半分にする事に注意
        # 2D Conv Layer 
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                             out_channels=128,
                                             kernel_size=[5, 5],
                                             stride=1,
                                             padding=[2, 2], bias=False),
                                   GLU())

        self.downSample1 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                   out_channels=256,
                                                   kernel_size=5,
                                                   stride=2,
                                                   padding=2, bias=False),
                                        nn.InstanceNorm2d(num_features=256,
                                                          affine=True),
                                        GLU())
        
        self.downSample2 = nn.Sequential(nn.Conv2d(in_channels=128,
                                                   out_channels=512,
                                                   kernel_size=5,
                                                   stride=2,
                                                   padding=2, bias=False),
                                        nn.InstanceNorm2d(num_features=512,
                                                          affine=True),
                                        GLU())
        
        # Encoder Residual Blocks
        self.resBlock1 = ResnetBlock()
        self.resBlock2 = ResnetBlock()
        self.resBlock3 = ResnetBlock()
        self.resBlock4 = ResnetBlock()
        self.resBlock5 = ResnetBlock()
        self.resBlock6 = ResnetBlock()

        # Class Activation Map
        self.gap_fc = nn.Linear(256, 1, bias=False)
        self.gmp_fc = nn.Linear(256, 1, bias=False)
        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        self.FC = nn.Sequential(
                  nn.Linear(256, 256, bias=False),
                  nn.ReLU(True),
                  nn.Linear(256, 256, bias=False),
                  nn.ReLU(True))
        self.gamma = nn.Linear(256, 256, bias=False)
        self.beta = nn.Linear(256, 256, bias=False)

        # Decoder Residual Blocks
        self.resBlock7 = ResnetAdaILNBlock()
        self.resBlock8 = ResnetAdaILNBlock()
        self.resBlock9 = ResnetAdaILNBlock()
        self.resBlock10 = ResnetAdaILNBlock()
        self.resBlock11 = ResnetAdaILNBlock()
        self.resBlock12 = ResnetAdaILNBlock()
        
        # UpSample Layer
        self.upSample1 = nn.Sequential(nn.Conv2d(in_channels=256,
                                                 out_channels=1024,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.PixelShuffle(2),
                                       nn.InstanceNorm2d(num_features=256, affine=True),
                                       GLU())
        self.upSample2 = nn.Sequential(nn.Conv2d(in_channels=128,
                                                 out_channels=512,
                                                 kernel_size=[7, 5],
                                                 stride=1,
                                                 padding=[3, 2], bias=False),
                                       nn.PixelShuffle(2),
                                       nn.InstanceNorm2d(num_features=128, affine=True),
                                       GLU())

        self.lastConvLayer = nn.Conv2d(in_channels=64,
                                       out_channels=1,
                                       kernel_size=[5, 5],
                                       stride=1,
                                       padding=[2, 2], bias=False)

    def forward(self, input):
        #(Batch_size, channel_num, height, width)

        conv1 = self.conv1(input)
        downsample1 = self.downSample1(conv1)
        downsample2 = self.downSample2(downsample1)

        residual_layer_1 = self.resBlock1(downsample2)
        residual_layer_2 = self.resBlock2(residual_layer_1)
        residual_layer_3 = self.resBlock3(residual_layer_2)
        residual_layer_4 = self.resBlock4(residual_layer_3)
        residual_layer_5 = self.resBlock5(residual_layer_4)
        residual_layer_6 = self.resBlock6(residual_layer_5)


        gap = torch.nn.functional.adaptive_avg_pool2d(residual_layer_6, 1)
        gap_logit = self.gap_fc(gap.view(residual_layer_6.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = residual_layer_6 * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(residual_layer_6, 1)
        gmp_logit = self.gmp_fc(gmp.view(residual_layer_6.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = residual_layer_6 * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        cam = torch.cat([gap, gmp], 1)
        cam = self.relu(self.conv1x1(cam))

        heatmap = torch.sum(cam, dim=1, keepdim=True)

        x_ = torch.nn.functional.adaptive_avg_pool2d(cam, 1)
        x_ = self.FC(x_.view(x_.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        residual_layer_7 = self.resBlock7(cam, gamma, beta)
        residual_layer_8 = self.resBlock8(residual_layer_7, gamma, beta)
        residual_layer_9 = self.resBlock9(residual_layer_8, gamma, beta)
        residual_layer_10 = self.resBlock10(residual_layer_9, gamma, beta)
        residual_layer_11 = self.resBlock11(residual_layer_10, gamma, beta)
        residual_layer_12 = self.resBlock12(residual_layer_11, gamma, beta)

        upsample_layer_1 = self.upSample1(residual_layer_12)
        upsample_layer_2 = self.upSample2(upsample_layer_1)

        output = self.lastConvLayer(upsample_layer_2)
        # print(downsample1.size())
        # print(downsample2.size())
        # print(reshape2dto1d.size())
        # print(conv2dto1d_layer.size())
        # print(residual_layer_1.size())
        # print(conv1dto2d_layer.size())
        # print(reshape1dto2d.size())
        # print(upsample_layer_1.size())
        # print(upsample_layer_2.size())
        # print(output.size())

        return output, cam_logit, heatmap

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf * 2, kernel_size=4, stride=2, padding=1, bias=True)),
                 GLU(),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2 * 2, kernel_size=4, stride=2, padding=1, bias=True)),
                      GLU(),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2 * 2, kernel_size=4, stride=1, padding=1, bias=True)),
                  GLU(),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=1, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w

if __name__ == '__main__':
    #For dimensional debug
    generator = Generator()
    discriminator = Discriminator(input_nc=1, n_layers=6)

    # Goutput, _, _ = generator.forward(torch.randn(1, 2, 80, 64))
    # print(Goutput.size())
    # Doutput, _, _ = discriminator.forward(Goutput)
    # print(Doutput.size())
    summary(generator, input_size=(1, 80, 64))
    # summary(discriminator, input_size=(1, 80, 64))