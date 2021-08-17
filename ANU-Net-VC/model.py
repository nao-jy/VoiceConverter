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

class AttentionGate(nn.Module):
    def __init__(self, input_channel, gate_channel):
        super(AttentionGate, self).__init__()
        self.inconv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.gate = nn.Conv2d(in_channels=gate_channel, out_channels=input_channel // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.midconv = nn.Conv2d(in_channels=input_channel // 2, out_channels=input_channel, kernel_size=1, stride=1, padding=0, bias=False)
    
    def forward(self, input, gate):
        gate = F.interpolate(gate, input.shape[2:], mode="bilinear" )

        x1 = self.inconv(input)
        x2 = self.gate(gate)
        x3 = F.relu(x1 + x2)
        x4 = torch.sigmoid(self.midconv(x3))

        return input * x4

class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(output_channel),
            nn.ReLU()
        )
    def forward(self, input):
        return self.conv(input)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        cnf = 32
        # Encoder
        self.x0_0 = ConvBlock(1, cnf)
        self.x1_0 = ConvBlock(cnf, cnf * 2)
        self.x2_0 = ConvBlock(cnf * 2, cnf * 4)
        self.x3_0 = ConvBlock(cnf * 4, cnf * 8)
        self.x4_0 = ConvBlock(cnf * 8, cnf * 16)

        self.x0_1 = ConvBlock(cnf + cnf * 2, cnf)
        self.x1_1 = ConvBlock(cnf * 2 + cnf * 4, cnf * 2)
        self.x2_1 = ConvBlock(cnf * 4 + cnf * 8, cnf * 4)
        self.x1_2 = ConvBlock(cnf * 4 + cnf * 2 + cnf * 2, cnf * 2)
        self.x0_3 = ConvBlock(cnf * 2 + cnf + cnf + cnf, cnf)

        self.x0_2 = ConvBlock(cnf + cnf + cnf * 2, cnf)

        # self.final = nn.Conv2d(cnf, 1, kernel_size=1)

        self.AG0_0 = AttentionGate(cnf, cnf * 2)
        self.AG0_1 = AttentionGate(cnf, cnf * 2)
        self.AG0_2 = AttentionGate(cnf, cnf * 2)
        self.AG0_3 = AttentionGate(cnf, cnf * 2)

        self.AG1_0 = AttentionGate(cnf * 2, cnf * 4)
        self.AG1_1 = AttentionGate(cnf * 2, cnf * 4)
        self.AG1_2 = AttentionGate(cnf * 2, cnf * 4)

        self.AG2_0 = AttentionGate(cnf * 4, cnf * 8)
        self.AG2_1 = AttentionGate(cnf * 4, cnf * 8)

        self.AG3_0 = AttentionGate(cnf * 8, cnf * 16)

        # Decoder
        self.x3_1 = ConvBlock(cnf * 16 + cnf * 8, cnf * 8)
        self.x2_2 = ConvBlock(cnf * 8 + cnf * 4 + cnf * 4, cnf * 4)
        self.x1_3 = ConvBlock(cnf * 4 + cnf * 2 + cnf * 2 + cnf * 2, cnf * 2)
        self.x0_4 = ConvBlock(cnf * 2 + cnf + cnf + cnf + cnf, 1)

        self.x3_1_A = ConvBlock(cnf * 16 + cnf * 8, cnf * 8)
        self.x2_2_A = ConvBlock(cnf * 8 + cnf * 4 + cnf * 4, cnf * 4)
        self.x1_3_A = ConvBlock(cnf * 4 + cnf * 2 + cnf * 2 + cnf * 2, cnf * 2)
        self.x0_4_A = ConvBlock(cnf * 2 + cnf + cnf + cnf + cnf, 1)

    def forward(self, input):
        downsample0_1 = nn.AdaptiveMaxPool2d([int(input.shape[2]) // 2, int(input.shape[3]) // 2])
        downsample1_2 = nn.AdaptiveMaxPool2d([int(input.shape[2]) // 4, int(input.shape[3]) // 4])
        downsample2_3 = nn.AdaptiveMaxPool2d([int(input.shape[2]) // 8, int(input.shape[3]) // 8])
        downsample3_4 = nn.AdaptiveMaxPool2d([int(input.shape[2]) // 16, int(input.shape[3]) // 16])
        upsample4_3 = nn.Upsample(size=[int(input.shape[2]) // 8, int(input.shape[3]) // 8], mode="bilinear")
        upsample3_2 = nn.Upsample(size=[int(input.shape[2]) // 4, int(input.shape[3]) // 4], mode="bilinear")
        upsample2_1 = nn.Upsample(size=[int(input.shape[2]) // 2, int(input.shape[3]) // 2], mode="bilinear")
        upsample1_0 = nn.Upsample(size=[int(input.shape[2]) // 1, int(input.shape[3]) // 1], mode="bilinear")

        # Encode
        x0_0_out = self.x0_0(input)
        x1_0_out = self.x1_0(downsample0_1(x0_0_out))
        x2_0_out = self.x2_0(downsample1_2(x1_0_out))
        x3_0_out = self.x3_0(downsample2_3(x2_0_out))
        x4_0_out = self.x4_0(downsample3_4(x3_0_out))

        # Decode
        AG0_0_out = self.AG0_0(x0_0_out, x1_0_out)
        AG1_0_out = self.AG1_0(x1_0_out, x2_0_out)
        AG2_0_out = self.AG2_0(x2_0_out, x3_0_out)
        AG3_0_out = self.AG3_0(x3_0_out, x4_0_out)

        x0_1_out = self.x0_1(torch.cat([AG0_0_out, upsample1_0(x1_0_out)], 1))
        x1_1_out = self.x1_1(torch.cat([AG1_0_out, upsample2_1(x2_0_out)], 1))
        x2_1_out = self.x2_1(torch.cat([AG2_0_out, upsample3_2(x3_0_out)], 1))
        x3_1_out = self.x3_1(torch.cat([AG3_0_out, upsample4_3(x4_0_out)], 1))
        x3_1_out_A = self.x3_1_A(torch.cat([AG3_0_out, upsample4_3(x4_0_out)], 1))

        AG0_1_out = self.AG0_1(x0_1_out, x1_1_out)
        AG1_1_out = self.AG1_1(x1_1_out, x2_1_out)
        AG2_1_out = self.AG2_1(x2_1_out, x3_1_out)        

        x0_2_out = self.x0_2(torch.cat([AG0_0_out, AG0_1_out, upsample1_0(x1_1_out)], 1))
        x1_2_out = self.x1_2(torch.cat([AG1_0_out, AG1_1_out, upsample2_1(x2_1_out)], 1))
        x2_2_out = self.x2_2(torch.cat([AG2_0_out, AG2_1_out, upsample3_2(x3_1_out)], 1))
        x2_2_out_A = self.x2_2_A(torch.cat([AG2_0_out, AG2_1_out, upsample3_2(x3_1_out_A)], 1))

        AG0_2_out = self.AG0_2(x0_2_out, x1_2_out)
        AG1_2_out = self.AG1_2(x1_2_out, x2_2_out)

        x0_3_out = self.x0_3(torch.cat([AG0_0_out, AG0_1_out, AG0_2_out, upsample1_0(x1_2_out)], 1))
        x1_3_out = self.x1_3(torch.cat([AG1_0_out, AG1_1_out, AG1_2_out, upsample2_1(x2_2_out)], 1))
        x1_3_out_A = self.x1_3(torch.cat([AG1_0_out, AG1_1_out, AG1_2_out, upsample2_1(x2_2_out_A)], 1))

        AG0_3_out = self.AG0_3(x0_3_out, x1_3_out)
        
        image = self.x0_4(torch.cat([AG0_0_out, AG0_1_out, AG0_2_out, AG0_3_out, upsample1_0(x1_3_out)], 1))
        attention = self.x0_4_A(torch.cat([AG0_0_out, AG0_1_out, AG0_2_out, AG0_3_out, upsample1_0(x1_3_out_A)], 1))
        attention = torch.sigmoid(attention)

        # softmax_ = nn.Softmax(dim=1)
        # attention = softmax_(attention)

        output = input * (1 - attention)
        output = torch.cat([output, image * attention], dim=1)

        return torch.sum(output, dim=1, keepdim=True), attention


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

if __name__ == '__main__':
    #For dimensional debug
    generator = Generator()
    # discriminator = Discriminator()

    Goutput = generator.forward(torch.randn(1, 1, 80, 128))
    # Doutput = discriminator.forward(Goutput)
    # print(Goutput.size())
    # print(Doutput.size())
    summary(generator, input_size=(1, 80, 64))