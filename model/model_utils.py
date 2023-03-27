import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssa import shunted_b, shunted_s, shunted_t


# CBR
class CBR(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, is_relu=True):
        super(CBR, self).__init__()

        self.is_relu = is_relu
        self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        if self.is_relu:
            out = self.relu(out)

        return out


# spatial_channel attention
class Spatial_Channel_Att(nn.Module):
    def __init__(self, in_channels):
        super(Spatial_Channel_Att, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(in_channels, in_channels // 2), nn.ReLU(),
                                 nn.Linear(in_channels // 2, in_channels))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        N, C, H, W = x.shape

        # channel
        attention_feature_max = F.adaptive_avg_pool2d(x, (1, 1)).view(N, C)
        channel_attention = F.softmax(self.mlp(attention_feature_max), dim=1).unsqueeze(2).unsqueeze(3)

        # spatial
        spatial_attention = torch.sigmoid(self.conv(x))

        out = x + x * channel_attention * spatial_attention

        return out


# Video_Encoder_part
class Encoder_Stage(nn.Module):

    def __init__(self, pretrained=True, model='s'):
        super(Encoder_Stage, self).__init__()

        if model == 'b':
            self.ssa = shunted_b(pretrained=pretrained)
        elif model == 's':
            self.ssa = shunted_s(pretrained=pretrained)
        elif model == 't':
            self.ssa = shunted_t(pretrained=pretrained)
        else:
            print('t, s, b')

        self.attention1 = Spatial_Channel_Att(64)
        self.attention2 = Spatial_Channel_Att(128)
        self.attention3 = Spatial_Channel_Att(256)
        self.attention4 = Spatial_Channel_Att(512)

    def forward(self, x):

        blocks = self.ssa(x)

        b1 = blocks[0]
        b2 = blocks[1]
        b3 = blocks[2]
        b4 = blocks[3]

        b1 = self.attention1(b1)
        b2 = self.attention2(b2)
        b3 = self.attention3(b3)
        b4 = self.attention4(b4)

        return [b1, b2, b3, b4]


# Video_Decoder_Part
class Decoder_Part(nn.Module):  # 0 1 2 3 4

    def __init__(self, in_channels, out_channels, is_Upsample=True):
        super(Decoder_Part, self).__init__()

        self.CBR1 = CBR(in_channels, in_channels, is_relu=False)
        self.CBR2 = CBR(in_channels, in_channels, is_relu=False)
        self.CBR3 = CBR(in_channels, out_channels)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.is_Upsample = is_Upsample

        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        out = self.CBR3(self.CBR2(self.CBR1(x))) + self.conv(x)

        if self.is_Upsample:
            out = self.Up_sample_2(out)

        return out


# -------------------- Time Moudle--------------------
class DEM(nn.Module):
    def __init__(self, in_channels):
        super(DEM, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.CBR1 = CBR(in_channels, in_channels)
        self.CBR2 = CBR(in_channels, in_channels)

    def Sub(self, frames):

        B, C, H, W = frames[0].size()

        for i in range(len(frames)):
            frames[i] = frames[i].flatten(2)

        tmp = []
        for i in range(len(frames)):
            if i == 0:
                x = torch.abs(frames[i] - frames[i])
            else:
                x = torch.abs(frames[i] - frames[i - 1])

            x = torch.softmax(x, dim=2)
            tmp.append(x)

        out = []
        A = torch.ones_like(tmp[0]).to(self.device) * 1

        for i in range(len(frames)):
            x = frames[i]
            x = x + x * (tmp[i] + A)
            out.append(x.reshape(B, C, H, W).contiguous())

        return out

    def forward(self, in_fs):  # (B, C, H, W)

        n = in_fs.size()[0]
        frames = []
        for i in range(n):
            f = in_fs[i, :, :, :].unsqueeze(0)
            frames.append(f)

        d_out1 = self.Sub(frames=frames)
        output1 = torch.cat(d_out1, dim=0)
        output1 = self.CBR1(output1 + in_fs)

        frames = []
        for i in range(n):
            f = output1[i, :, :, :].unsqueeze(0)
            frames.append(f)

        frames = frames[::-1]
        d_out2 = self.Sub(frames=frames)
        d_out2 = d_out2[::-1]

        output2 = torch.cat(d_out2, dim=0)
        output2 = self.CBR2(output2 + output1)

        return output2


# decoder_stage
class Decoder_Stage(nn.Module):
    def __init__(self, in_channels, out_channels, is_Upsample=True, is_DEM=True):
        super(Decoder_Stage, self).__init__()

        self.is_DEM = is_DEM

        self.DEM = DEM(in_channels=out_channels)
        self.CBR = CBR(in_channels, in_channels)
        self.decoder = Decoder_Part(in_channels, out_channels, is_Upsample=is_Upsample)

    def forward(self, x, block=None):

        if block == None:
            x = x
        else:
            x = self.CBR(x + block)

        out = self.decoder(x)

        if self.is_DEM:
            out = self.DEM(out)

        return out


class Output_Stage(nn.Module):
    def __init__(self, in_channels):
        super(Output_Stage, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1)

        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, outs):
        output = self.Up_sample_2(self.conv(outs))

        return output


class FBFM(nn.Module):
    def __init__(self, in_channels):
        super(FBFM, self).__init__()

        self.CBR1 = CBR(in_channels, in_channels, is_relu=False)
        self.CBR2 = CBR(in_channels, in_channels, is_relu=False)
        self.CBR3 = CBR(in_channels, in_channels, is_relu=False)
        self.CBR4 = CBR(in_channels, in_channels, is_relu=False)
        self.CBR5 = CBR(in_channels, in_channels)

    def forward(self, x1, x2, x3=None):

        if x3 == None:
            x = self.CBR1(x1 + x2)

            B, C, H, W = x.size()

            s1 = torch.softmax(self.CBR2(x + x1).flatten(2), dim=2).reshape(B, C, H, W).contiguous()
            s2 = torch.softmax(self.CBR3(x + x2).flatten(2), dim=2).reshape(B, C, H, W).contiguous()

            out1 = x1 + x1 * s1
            out2 = x2 + x2 * s2

            output = self.CBR5(out2 + out1)

        else:
            x = self.CBR1(x1 + x2 + x3)

            B, C, H, W = x.size()

            s1 = torch.softmax(self.CBR2(x + x1).flatten(2), dim=2).reshape(B, C, H, W).contiguous()
            s2 = torch.softmax(self.CBR3(x + x2).flatten(2), dim=2).reshape(B, C, H, W).contiguous()
            s3 = torch.softmax(self.CBR4(x + x3).flatten(2), dim=2).reshape(B, C, H, W).contiguous()

            out1 = x1 + x1 * s1
            out2 = x2 + x2 * s2
            out3 = x3 + x3 * s3

            output = self.CBR5(out2 + out1 + out3)

        return output
