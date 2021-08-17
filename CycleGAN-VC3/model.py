import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        return x1 * torch.sigmoid(x2)

class TFAN_1D(nn.Module):
    def __init__(self, channel_input):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm1d(channel_input, affine=False)

        channel_hidden = 128
        kernel_size = 5
        pad_size = kernel_size // 2

        self.hidden_layer = nn.Sequential(
            nn.Conv1d(channel_input, channel_hidden, kernel_size=kernel_size, padding=pad_size),
            nn.ReLU(),
            nn.Conv1d(channel_hidden, channel_hidden, kernel_size=kernel_size, padding=pad_size),
            nn.ReLU(),
            nn.Conv1d(channel_hidden, channel_hidden, kernel_size=kernel_size, padding=pad_size),
            nn.ReLU()
        )
        self.gamma = nn.Conv1d(channel_hidden, channel_input, kernel_size=kernel_size, padding=pad_size)
        self.beta = nn.Conv1d(channel_hidden, channel_input, kernel_size=kernel_size, padding=pad_size)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.size()[1:], mode='nearest')

        segmap = torch.squeeze(segmap, 1)

        actv = self.hidden_layer(segmap)

        gamma = self.gamma(actv)
        beta = self.beta(actv)

        out = normalized * (1 + gamma) + beta

        return out

class TFAN_2D(nn.Module):

    def __init__(self, channel_input):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(channel_input, affine=False)

        channel_hidden = 128
        kernel_size = 5
        pad_size = kernel_size // 2

        self.hidden_layer = nn.Sequential(
            nn.Conv2d(1, channel_hidden, kernel_size=kernel_size, padding=pad_size),
            nn.ReLU(),
            nn.Conv2d(channel_hidden, channel_hidden, kernel_size=kernel_size, padding=pad_size),
            nn.ReLU(),
            nn.Conv2d(channel_hidden, channel_hidden, kernel_size=kernel_size, padding=pad_size),
            nn.ReLU()
        )
        self.gamma = nn.Conv2d(channel_hidden, channel_input, kernel_size=kernel_size, padding=pad_size)
        self.beta = nn.Conv2d(channel_hidden, channel_input, kernel_size=kernel_size, padding=pad_size)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        actv = self.hidden_layer(segmap)

        gamma = self.gamma(actv)
        beta = self.beta(actv)

        out = normalized * (1 + gamma) + beta

        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # GLUはチャネル数を半分にする事に注意
        # 2D Conv Layer 
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                             out_channels=128,
                                             kernel_size=[5, 15],
                                             stride=1,
                                             padding=[2, 7]),
                                   GLU())

        self.downSample1 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                   out_channels=256,
                                                   kernel_size=5,
                                                   stride=2,
                                                   padding=2),
                                        nn.InstanceNorm2d(num_features=256,
                                                          affine=True),
                                        GLU())
        
        self.downSample2 = nn.Sequential(nn.Conv2d(in_channels=128,
                                                   out_channels=512,
                                                   kernel_size=5,
                                                   stride=2,
                                                   padding=2),
                                        nn.InstanceNorm2d(num_features=512,
                                                          affine=True),
                                        GLU())

        # 2D -> 1D Conv
        #reshapeはforwardで
        self.conv2dto1dLayer = nn.Sequential(nn.Conv1d(in_channels=2560,
                                                       out_channels=256,
                                                       kernel_size=1,
                                                       stride=1,
                                                       padding=0),
                                             nn.InstanceNorm1d(num_features=256,
                                                               affine=True))
        
        # Residual Blocks
        self.resBlock1 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       nn.InstanceNorm1d(num_features=512,
                                                         affine=True),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       nn.InstanceNorm1d(num_features=256,
                                                         affine=True))
        self.resBlock2 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       nn.InstanceNorm1d(num_features=512,
                                                         affine=True),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       nn.InstanceNorm1d(num_features=256,
                                                         affine=True))
        self.resBlock3 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       nn.InstanceNorm1d(num_features=512,
                                                         affine=True),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       nn.InstanceNorm1d(num_features=256,
                                                         affine=True))
        self.resBlock4 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       nn.InstanceNorm1d(num_features=512,
                                                         affine=True),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       nn.InstanceNorm1d(num_features=256,
                                                         affine=True))
        self.resBlock5 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       nn.InstanceNorm1d(num_features=512,
                                                         affine=True),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       nn.InstanceNorm1d(num_features=256,
                                                         affine=True))
        self.resBlock6 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       nn.InstanceNorm1d(num_features=512,
                                                         affine=True),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       nn.InstanceNorm1d(num_features=256,
                                                         affine=True))
        # 1D -> 2D Conv
        #reshapeはforwardで
        self.conv1dto2dLayer = nn.Sequential(nn.Conv1d(in_channels=256,
                                         out_channels=2560,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0))
        
        self.tfan1 = TFAN_1D(2560)
        
        # UpSample Layer
        self.upSample1 = nn.Sequential(nn.Conv2d(in_channels=256,
                                                 out_channels=1024,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2),
                                       nn.PixelShuffle(2))
        self.tfan2 = TFAN_2D(256)
        self.upSample2 = nn.Sequential(GLU(),
                                       nn.Conv2d(in_channels=128,
                                                 out_channels=512,
                                                 kernel_size=[11, 5],
                                                 stride=1,
                                                 padding=[5, 2]),
                                       nn.PixelShuffle(2))
        self.tfan3 = TFAN_2D(128)
        self.lastConvLayer = nn.Sequential(GLU(),
                                           nn.Conv2d(in_channels=64,
                                           out_channels=1,
                                           kernel_size=[5, 15],
                                           stride=1,
                                           padding=[2, 7]))

    def forward(self, input):
        #(Batch_size, channel_num, height, width)

        conv1 = self.conv1(input)
        downsample1 = self.downSample1(conv1)
        downsample2 = self.downSample2(downsample1)
        reshape2dto1d = downsample2.view(downsample2.size(0), 2560, 1, -1)
        reshape2dto1d = reshape2dto1d.squeeze(2)

        conv2dto1d_layer = self.conv2dto1dLayer(reshape2dto1d)
        residual_layer_1 = self.resBlock1(conv2dto1d_layer) + conv2dto1d_layer
        residual_layer_2 = self.resBlock2(residual_layer_1) + residual_layer_1
        residual_layer_3 = self.resBlock3(residual_layer_2) + residual_layer_2
        residual_layer_4 = self.resBlock4(residual_layer_3) + residual_layer_3
        residual_layer_5 = self.resBlock5(residual_layer_4) + residual_layer_4
        residual_layer_6 = self.resBlock6(residual_layer_5) + residual_layer_5
        conv1dto2d_layer = self.conv1dto2dLayer(residual_layer_6)
        conv1dto2d_layer = self.tfan1(conv1dto2d_layer, input)
        reshape1dto2d = conv1dto2d_layer.unsqueeze(2)
        reshape1dto2d = reshape1dto2d.view(reshape1dto2d.size(0), 256, 20, -1)

        upsample_layer_1 = self.upSample1(reshape1dto2d)
        upsample_layer_1 = self.tfan2(upsample_layer_1, input)
        upsample_layer_2 = self.upSample2(upsample_layer_1)
        upsample_layer_2 = self.tfan3(upsample_layer_2, input)

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

        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                             out_channels=128,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   GLU())
        
        self.downSample1 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                   out_channels=256,
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1),
                                         nn.InstanceNorm2d(num_features=256,
                                                           affine=True),
                                         GLU())

        self.downSample2 = nn.Sequential(nn.Conv2d(in_channels=128,
                                                   out_channels=512,
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1),
                                         nn.InstanceNorm2d(num_features=512,
                                                           affine=True),
                                         GLU())
        self.downSample3 = nn.Sequential(nn.Conv2d(in_channels=256,
                                                   out_channels=1024,
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1),
                                         nn.InstanceNorm2d(num_features=1024,
                                                           affine=True),
                                         GLU())

        self.downSample4 = nn.Sequential(nn.Conv2d(in_channels=512,
                                                   out_channels=1024,
                                                   kernel_size=[1, 5],
                                                   stride=1,
                                                   padding=[0, 2]),
                                         nn.InstanceNorm2d(num_features=1024,
                                                           affine=True),
                                         GLU())
        self.outputConvLayer = nn.Sequential(nn.Conv2d(in_channels=512,
                                                       out_channels=1,
                                                       kernel_size=[1, 3],
                                                       stride=1,
                                                       padding=[0, 1]))
    
    def forward(self, input):
        conv1 = self.conv1(input)

        downsample1 = self.downSample1(conv1)

        downsample2 = self.downSample2(downsample1)

        downsample3 = self.downSample3(downsample2)

        downsample4 = self.downSample4(downsample3)

        output = self.outputConvLayer(downsample4)

        return output

if __name__ == '__main__':
    #For dimensional debug
    generator = Generator()
    discriminator = Discriminator()

    Goutput = generator.forward(torch.randn(1, 1, 80, 64))
    Doutput = discriminator.forward(Goutput)
    # print(Goutput.size())
    # print(Doutput.size())
    summary(generator, input_size=(1, 80, 64))