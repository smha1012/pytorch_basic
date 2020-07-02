#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#@Filename : anogan
#@Date : 2020-07-02-14-03
#@Project : anogan
#@Author : seungmin

import random
import math
import time
import pandas as pd
import numpy as np
### from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
### from torchvision import transforms
import matplotlib.pyplot as plt

from model.model import Generator, Discriminator
from dataloader import make_datapath_list, ImageTransform, GAN_Img_Dataset

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


### generator 동작확인
G = Generator(z_dim=20, image_size=64)

input_z = torch.randn(1, 20)
input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)

fake_images = G(input_z)

img_transformed = fake_images[0][0].detach().numpy()
plt.imshow(img_transformed, 'gray')
#plt.show()
#plt.close()

### discriminator 동작확인
D = Discriminator(z_dim=20, image_size=64)

input_z = torch.randn(1, 20)
input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
fake_images = G(input_z)

d_out = D(fake_images)
print(d_out)
### sigmoid 함수로 출력 변환
print(nn.Sigmoid()(d_out[0]))
### feature
print(d_out[1].shape)

### DataLoader 인스턴스 생성 및 동작 확인
## 이미지 파일 리스트 작성
train_img_list=make_datapath_list()

## 데이터셋 작성
mean = (0.5,)
std = (0.5,)
train_dataset=GAN_Img_Dataset(
    file_list=train_img_list, transform=ImageTransform(mean, std))

## DataLoader 작성
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

## 동작 확인
batch_iterator = iter(train_dataloader) # iterator
images = next(batch_iterator)  # 첫번째 이미지
print(images.size())  # torch.Size([64, 1, 64, 64])

### 네트워크 초기화
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G.apply(weights_init)
D.apply(weights_init)

print("Network initialized.")

### train 함수 작성
def train_model(G, D, dataloader, num_epochs):
    ## cuda 제어
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용디바이스：", device)

    ## 최적화 파라미터 설정
    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    ## Loss function
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    ## 기타 파라미터
    z_dim = 20
    mini_batch_size = 64

    ## 네트워트 -> GPU
    G.to(device)
    D.to(device)

    G.train()  # モデルを訓練モードに
    D.train()  # モデルを訓練モードに

    ## 네트워크가 어느정도 고정이 되면 가속화
    torch.backends.cudnn.benchmark = True

    ## 이미지 갯수
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    ## 이터레이션 카운트 시작
    iteration = 1
    logs = []

    ## 루프
    for epoch in range(num_epochs):

        # 루프 시작시간
        t_epoch_start = time.time()
        epoch_g_loss = 0.0  # 손실합
        epoch_d_loss = 0.0  # 손실합

        print('-------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-------------')
        print('（train）')

        # Dataloader에서 minibatch 크기 만큼 떼어내는 루프
        for images in dataloader:

            # --------------------
            # 1. Discriminator 학습
            # --------------------
            # minibatch 크기가 1이면, batch normalization에서 에러를 발생하므로 제어함
            if images.size()[0] == 1:
                continue

            # GPU를 사용하는 경우 GPU로 데이터 송신
            images = images.to(device)

            # 정답라벨과 가짜라벨을 작성
            mini_batch_size = images.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device) # torch.full(size, fill_value); Returns a tensor of size 'size' filled with 'fill_value'
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            # 진짜 이미지인지 판정
            d_out_real, _ = D(images)

            # 가짜 이미지를 생성하고 판정
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images)

            # 손실함수 계산
            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            # Backpropagation
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # --------------------
            # 2. Generator 학습
            # --------------------
            # 가짜 이미지를 생성하여 판정
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images)

            # 손실함수 계산
            g_loss = criterion(d_out_fake.view(-1), label_real)

            # Backpropagation
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            g_loss.backward()
            g_optimizer.step()

            # --------------------
            # 3. 기록
            # --------------------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        # epoch 당 손실과 정답률
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}'.format(
            epoch, epoch_d_loss / batch_size, epoch_g_loss / batch_size))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    print("총 이터레이션 수: ", iteration)

    return G, D

num_epochs = 300
G_update, D_update = train_model(
    G, D, dataloader=train_dataloader, num_epochs=num_epochs)