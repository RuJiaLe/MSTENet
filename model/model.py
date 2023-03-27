import torch
import torch.nn as nn
from .model_utils import Decoder_Stage, Output_Stage, Encoder_Stage, FBFM, CBR, Spatial_Channel_Att


class Model(nn.Module):

    def __init__(self, pretrained=True):
        super(Model, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # --------------------编码阶段--------------------
        self.encoder1 = Encoder_Stage(pretrained=pretrained, model='s')
        self.encoder2 = Encoder_Stage(pretrained=pretrained, model='s')

        self.attention1 = Spatial_Channel_Att(64)
        self.attention2 = Spatial_Channel_Att(128)
        self.attention3 = Spatial_Channel_Att(256)
        self.attention4 = Spatial_Channel_Att(512)

        # --------------------Decoder_Stage--------------------
        self.stream1_3 = Decoder_Stage(512, 256)
        self.stream1_2 = Decoder_Stage(256, 128)
        self.stream1_1 = Decoder_Stage(128, 64)
        self.stream1_0 = Decoder_Stage(64, 64)

        self.stream2_3 = Decoder_Stage(512, 256)
        self.stream2_2 = Decoder_Stage(256, 128)
        self.stream2_1 = Decoder_Stage(128, 64)
        self.stream2_0 = Decoder_Stage(64, 64)
        #
        self.stream3_3 = Decoder_Stage(512, 256)
        self.stream3_2 = Decoder_Stage(256, 128)
        self.stream3_1 = Decoder_Stage(128, 64)
        self.stream3_0 = Decoder_Stage(64, 64)

        # --------------------Output_Stage阶段--------------------
        self.output_stream3 = Output_Stage(64)
        self.output_stream2 = Output_Stage(64)
        self.output_stream1 = Output_Stage(64)

        # --------------------Fuse_Stage阶段--------------------
        self.CBR1 = CBR(in_channels=64, out_channels=128, stride=2)
        self.CBR2 = CBR(in_channels=128, out_channels=256, stride=2)
        self.CBR3 = CBR(in_channels=256, out_channels=512, stride=2)

        self.en_fbfm_3 = FBFM(64)
        self.en_fbfm_2 = FBFM(128)
        self.en_fbfm_1 = FBFM(256)
        self.en_fbfm_0 = FBFM(512)

        self.de_fbfm_3 = FBFM(256)
        self.de_fbfm_2 = FBFM(128)
        self.de_fbfm_1 = FBFM(64)
        self.de_fbfm_0 = FBFM(64)

    def forward(self, frames):
        # --------------------编码阶段--------------------
        blocks1 = self.encoder1(frames)
        blocks2 = self.encoder2(frames)

        block3_1 = self.attention1(self.en_fbfm_3(blocks1[0], blocks2[0], torch.zeros_like(blocks1[0]).to(self.device)))
        block3_2 = self.attention2(self.en_fbfm_2(blocks1[1], blocks2[1], self.CBR1(block3_1)))
        block3_3 = self.attention3(self.en_fbfm_1(blocks1[2], blocks2[2], self.CBR2(block3_2)))
        block3_4 = self.attention4(self.en_fbfm_0(blocks1[3], blocks2[3], self.CBR3(block3_3)))

        blocks3 = [block3_1, block3_2, block3_3, block3_4]

        # --------------------解码阶段--------------------
        out_stream1_3 = self.stream1_3(blocks1[-1])
        out_stream2_3 = self.stream2_3(blocks2[-1])
        out_stream3_3 = self.stream3_3(blocks3[-1])

        out_stream1_2 = self.stream1_2(out_stream1_3, blocks1[2])
        out_stream2_2 = self.stream2_2(out_stream2_3, blocks2[2])
        out_stream3_2 = self.stream3_2(self.de_fbfm_3(out_stream1_3, out_stream2_3, out_stream3_3), blocks3[2])

        out_stream1_1 = self.stream1_1(out_stream1_2, blocks1[1])
        out_stream2_1 = self.stream2_1(out_stream2_2, blocks2[1])
        out_stream3_1 = self.stream3_1(self.de_fbfm_2(out_stream1_2, out_stream2_2, out_stream3_2), blocks3[1])

        out_stream1 = self.stream1_0(out_stream1_1, blocks1[0])
        out_stream2 = self.stream2_0(out_stream2_1, blocks2[0])
        out_stream3 = self.stream3_0(self.de_fbfm_1(out_stream1_1, out_stream2_1, out_stream3_1), blocks3[0])

        out = self.de_fbfm_0(out_stream1, out_stream2, out_stream3)

        # --------------------第一输出阶段--------------------
        output1 = self.output_stream1(out_stream1)

        output2 = self.output_stream2(out_stream2)

        output3 = self.output_stream3(out)

        return torch.sigmoid(output1), torch.sigmoid(output2), torch.sigmoid(output3)
