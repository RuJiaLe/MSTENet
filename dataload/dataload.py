from torch.utils.data import Dataset
import os
import random
from PIL import Image
import numpy as np


class VideoDataset(Dataset):
    def __init__(self, root_dir="", train_set_list=None, training=True, transforms=None, clip_len=4):
        super(VideoDataset, self).__init__()

        self.root_dir = root_dir
        self.clip_len = clip_len
        self.training = training
        self.transforms = transforms
        self.train_set_list = train_set_list
        self.frames = []

        for train_set in self.train_set_list:
            video_root = os.path.join(root_dir, train_set).replace('\\', '/')
            sequence_list = sorted(os.listdir(video_root))

            for sequence in sequence_list:
                sequence_info = self.get_frame_list(train_set, sequence)
                self.frames += self.get_clips(sequence_info)

    def get_frame_list(self, train_set, sequence):
        image_path_root = os.path.join(self.root_dir, train_set, sequence, "Imgs").replace('\\', '/')
        frame_list = sorted(os.listdir(image_path_root))
        sequence_info = []

        for i in range(len(frame_list)):
            image_path = os.path.join(self.root_dir, train_set, sequence, "Imgs", frame_list[i]).replace('\\', '/')

            frame_name = frame_list[i].split('.')[0]
            gt_name = frame_name + '.png'
            gt_path = os.path.join(self.root_dir, train_set, sequence, "ground-truth", gt_name).replace('\\', '/')

            if os.path.exists(gt_path):
                frame_info = {"image_path": image_path,
                            "gt_path": gt_path}

                sequence_info.append(frame_info)

        return sequence_info

    def get_clips(self, sequence_info):

        clips = []

        length = len(sequence_info)

        if length < self.clip_len:

            sequence_info *= (self.clip_len) // length

            sequence_info += sequence_info[0: self.clip_len % length]

            clips.append(sequence_info)

        else:
            for i in range(int(length / self.clip_len)):
                clips.append(sequence_info[self.clip_len * i: self.clip_len * (i + 1)])

            finish = self.clip_len * (int(length / self.clip_len))

            if finish < len(sequence_info):
                clips.append(sequence_info[length - self.clip_len: length])

        return clips


    def get_frame(self, frame_info):
        image_path = frame_info["image_path"]
        image = Image.open(image_path).convert("RGB")
        gt_path = frame_info["gt_path"]
        gt = Image.open(gt_path).convert("L")

        sample = {"image": image, "gt": gt, "not_gt": gt, "path": image_path}

        return sample

    def __getitem__(self, idx):
        frame = self.frames[idx]

        frame_output = []

        if self.training and random.randint(0, 1):
            frame = frame[::-1]

        for i in range(len(frame)):
            item = self.get_frame(frame[i])
            frame_output.append(item)

        frame_output = self.transforms(frame_output)

        return frame_output

    def __len__(self):
        return len(self.frames)
