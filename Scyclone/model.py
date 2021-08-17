import torch.nn as nn
import torch
from torchsummary import summary

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        return x1 * torch.sigmoid(x2)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # GLUはチャネル数を半分にする事に注意
        # 2D Conv Layer 
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=80,
                                             out_channels=256,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0),
                                   nn.LeakyReLU(0.01))

        # Residual Blocks
        self.resBlock1 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2),
                                       nn.LeakyReLU(0.01),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2))

        self.resBlock2 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2),
                                       nn.LeakyReLU(0.01),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2))

        self.resBlock3 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2),
                                       nn.LeakyReLU(0.01),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2))

        self.resBlock4 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2),
                                       nn.LeakyReLU(0.01),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2))

        self.resBlock5 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2),
                                       nn.LeakyReLU(0.01),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2))

        self.resBlock6 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2),
                                       nn.LeakyReLU(0.01),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2))

        self.resBlock7 = nn.Sequential(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2),
                                       nn.LeakyReLU(0.01),
                                       GLU(),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2))

        self.lastConvLayer = nn.Sequential(nn.Conv1d(in_channels=256,
                                             out_channels=80,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0),
                                             nn.LeakyReLU(0.01))

    def forward(self, input):
        #(Batch_size, channel_num, height, width)
        input = torch.squeeze(input, 1)

        conv1 = self.conv1(input)

        residual_layer_1 = self.resBlock1(conv1) + conv1
        residual_layer_2 = self.resBlock2(residual_layer_1) + residual_layer_1
        residual_layer_3 = self.resBlock3(residual_layer_2) + residual_layer_2
        residual_layer_4 = self.resBlock4(residual_layer_3) + residual_layer_3

        # residual_layer_5 = self.resBlock5(torch.cat([residual_layer_4, residual_layer_3], 1)) + residual_layer_4
        # residual_layer_6 = self.resBlock6(torch.cat([residual_layer_5, residual_layer_2], 1)) + residual_layer_5
        # residual_layer_7 = self.resBlock7(torch.cat([residual_layer_6, residual_layer_1], 1)) + residual_layer_6
        residual_layer_5 = self.resBlock5(residual_layer_4) + residual_layer_4
        residual_layer_6 = self.resBlock6(residual_layer_5) + residual_layer_5
        residual_layer_7 = self.resBlock7(residual_layer_6) + residual_layer_6

        output = self.lastConvLayer(residual_layer_7)

        return torch.unsqueeze(output, 1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(nn.utils.spectral_norm(
                                   nn.Conv1d(in_channels=80,
                                             out_channels=256,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)),
                                   nn.LeakyReLU(0.2))

        # Residual Blocks
        self.resBlock1 = nn.Sequential(nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2)),
                                       nn.LeakyReLU(0.2),
                                       GLU(),
                                       nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2)))

        self.resBlock2 = nn.Sequential(nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2)),
                                       nn.LeakyReLU(0.2),
                                       GLU(),
                                       nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2)))

        self.resBlock3 = nn.Sequential(nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2)),
                                       nn.LeakyReLU(0.2),
                                       GLU(),
                                       nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2)))

        self.resBlock4 = nn.Sequential(nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2)),
                                       nn.LeakyReLU(0.2),
                                       GLU(),
                                       nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2)))

        self.resBlock5 = nn.Sequential(nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2)),
                                       nn.LeakyReLU(0.2),
                                       GLU(),
                                       nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2)))

        self.resBlock6 = nn.Sequential(nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                                 out_channels=512,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2)),
                                       nn.LeakyReLU(0.2),
                                       GLU(),
                                       nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2)))

        self.lastConvLayer = nn.Sequential(nn.utils.spectral_norm(nn.Conv1d(in_channels=256,
                                             out_channels=1,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)),
                                             nn.LeakyReLU(0.2))

    def forward(self, input):
        #(Batch_size, channel_num, height, width)
        input = torch.squeeze(input, 1)

        conv1 = self.conv1(input)

        residual_layer_1 = self.resBlock1(conv1) + conv1
        residual_layer_2 = self.resBlock2(residual_layer_1) + residual_layer_1
        residual_layer_3 = self.resBlock3(residual_layer_2) + residual_layer_2
        residual_layer_4 = self.resBlock4(residual_layer_3) + residual_layer_3
        residual_layer_5 = self.resBlock5(residual_layer_4) + residual_layer_4
        residual_layer_6 = self.resBlock6(residual_layer_5) + residual_layer_5

        output = torch.mean(self.lastConvLayer(residual_layer_6), dim=2)

        return output

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,
#                                              out_channels=128,
#                                              kernel_size=3,
#                                              stride=1,
#                                              padding=1),
#                                    GLU())
        
#         self.downSample1 = nn.Sequential(nn.Conv2d(in_channels=64,
#                                                    out_channels=256,
#                                                    kernel_size=3,
#                                                    stride=2,
#                                                    padding=1),
#                                          nn.InstanceNorm2d(num_features=256,
#                                                            affine=True),
#                                          GLU())

#         self.downSample2 = nn.Sequential(nn.Conv2d(in_channels=128,
#                                                    out_channels=512,
#                                                    kernel_size=3,
#                                                    stride=2,
#                                                    padding=1),
#                                          nn.InstanceNorm2d(num_features=512,
#                                                            affine=True),
#                                          GLU())
#         self.downSample3 = nn.Sequential(nn.Conv2d(in_channels=256,
#                                                    out_channels=1024,
#                                                    kernel_size=3,
#                                                    stride=2,
#                                                    padding=1),
#                                          nn.InstanceNorm2d(num_features=1024,
#                                                            affine=True),
#                                          GLU())

#         self.downSample4 = nn.Sequential(nn.Conv2d(in_channels=512,
#                                                    out_channels=1024,
#                                                    kernel_size=[1, 5],
#                                                    stride=1,
#                                                    padding=[0, 2]),
#                                          nn.InstanceNorm2d(num_features=1024,
#                                                            affine=True),
#                                          GLU())
#         self.outputConvLayer = nn.Sequential(nn.Conv2d(in_channels=512,
#                                                        out_channels=1,
#                                                        kernel_size=[1, 3],
#                                                        stride=1,
#                                                        padding=[0, 1]))
    
#     def forward(self, input):
#         conv1 = self.conv1(input)

#         downsample1 = self.downSample1(conv1)

#         downsample2 = self.downSample2(downsample1)

#         downsample3 = self.downSample3(downsample2)

#         downsample4 = self.downSample4(downsample3)

#         output = self.outputConvLayer(downsample4)

#         return output

if __name__ == '__main__':
    #For dimensional debug
    generator = Generator()
    discriminator = Discriminator()

    Goutput = generator.forward(torch.randn(1, 1, 80, 64))
    Doutput = discriminator.forward(Goutput)
    print(Goutput.size())
    print(Doutput.size())
    # summary(discriminator, input_size=(1, 80, 64))