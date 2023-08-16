import os
import cv2
import sys
import pdb
import six
import glob
import time
import lmdb
import torch
import random
import pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")
global kernel_sizes 

class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, dataset='phoenix2014', num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb", image_scale_big=1.0, image_scale_mid=160, image_scale_small=96, kernel_size=1, allowable_vid_length=16):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.allowable_vid_length = allowable_vid_length
        global kernel_sizes 
        kernel_sizes = kernel_size
        self.dataset = dataset
        self.image_scale_big = image_scale_big
        self.image_scale_mid = image_scale_mid
        self.image_scale_small = image_scale_small # not implemented for read_features()
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = dict([*filter(lambda x: isinstance(x[0], str) or x[0] < 10, self.inputs_list.items())])
        print(mode, len(self))
        self.data_aug = self.transform()
        self.resize_small = video_augmentation.Resize(self.image_scale_small)
        self.resize_mid = video_augmentation.Resize(self.image_scale_mid)
        self.resize_big = video_augmentation.Resize(self.image_scale_big)
        self.to_tensor = video_augmentation.ToTensor()

    def __getitem__(self, idx):
        if self.data_type == "video":   #only implemented for video type with big and small images
            input_data, label, _ = self.read_video(idx)
            input_data, input_data_mid, input_data_small, label = self.normalize(input_data, label)
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data, input_data_mid, input_data_small, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        elif self.data_type == "lmdb":
            input_data, label, _ = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]['original_info']

    def read_video(self, index):
        # load file info
        fi = self.inputs_list[index]
        if 'phoenix' in self.dataset:
            img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])  
        elif self.dataset == 'CSL':
            img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'] + "/*.jpg")
        elif self.dataset == 'CSL-Daily':
            img_folder = os.path.join(self.prefix, fi['folder'])
        img_list = sorted(glob.glob(img_folder))

        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, fi

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video_small = self.resize_small(video)
        video_mid = self.resize_mid(video)
        video = self.resize_big(video)
        video = self.to_tensor(video)
        video_mid = self.to_tensor(video_mid)
        video_small = self.to_tensor(video_small)
        video = video.float() / 127.5 - 1
        video_mid = video_mid.float() / 127.5 - 1
        video_small = video_small.float() / 127.5 - 1
        return video, video_mid, video_small, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                #video_augmentation.Resize(self.image_scale),
                video_augmentation.TemporalRescale(0.2),
                #video_augmentation.ToTensor(),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                #video_augmentation.Resize(self.image_scale),
                #video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, video_mid, video_small, label, info = list(zip(*batch))
        # not pad video, but only process them into the same length
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid) , -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
            padded_video_mid = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid) , -1, -1, -1),
                )
                , dim=0)
                for vid in video_mid]
            padded_video_mid = torch.stack(padded_video_mid)
            padded_video_small = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid) , -1, -1, -1),
                )
                , dim=0)
                for vid in video_small]
            padded_video_small = torch.stack(padded_video_small)
        else:
            padded_video = torch.tensor(video)
            padded_video_mid = torch.tensor(video_mid)
            padded_video_small = torch.tensor(video_small)
        video_length = torch.LongTensor([len(vid)  for vid in video])

        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, padded_video_mid, padded_video_small, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, padded_video_mid, padded_video_small, video_length, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        print
        data[1]
        pdb.set_trace()
