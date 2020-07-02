#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#@Filename : dataloader
#@Date : 2020-07-02-15-22
#@Project : anogan
#@Author : seungmin

import torch.utils.data as data
from torchvision import transforms
from PIL import Image

### Dataloader 작성
##
def make_datapath_list():

    train_img_list = list()

    for img_idx in range(200):
        img_path = "./data/img_78/img_7_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

        img_path = "./data/img_78/img_8_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

    return train_img_list

## 이미지 전처리 클래스
class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)

## 이미지 데이터셋 클래스, pytorch 데이터셋 클래스 상속
class GAN_Img_Dataset(data.Dataset):

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        # 이미지 갯수 리턴
        return len(self.file_list)

    def __getitem__(self, index):
        # 전처리 한 이미지의 텐서 형태 취득
        img_path = self.file_list[index]
        img = Image.open(img_path)
        # 이미지 전처리
        img_transformed = self.transform(img)

        return img_transformed