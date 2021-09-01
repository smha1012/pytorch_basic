#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : trainer
# @Date : 2021-09-01-09-17
# @Project : pytorch_basic
# @Author : seungmin

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

import os
from datetime import datetime
import shutil
import numpy as np
import matplotlib.pyplot as plt
from utils import utils

from model.mymodel import MyModel

print("#########################################")
print("#       TRAINER TEMPLATE.  v1.0         #")
print("#                                       #")
print("#                   -Promedius Inc.-    #")
print("#                                       #")
print("#########################################")

# 그래프가 학습이 진행됨에 따라 어느정도 고정이 되면, 학습 속도를 빠르게 진행하기 위한 옵션.
cudnn.benchmark = True

# mymodel.yaml 파일을 모델 가중치와 세트로 저장하기 위한 함수.
def _save_config_file(model_checkpoints_folder, model_name):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    shutil.copy('./config/' + model_name + '.yaml', os.path.join(model_checkpoints_folder, model_name + '.yaml'))

# 최초, 또는 가장 최근 학습된 가중치와 yaml 파일은 일단 ./weight/checkpoint 폴더에 저장됨.
# 이후에 이뤄지는 실험은 이전보다 최신이므로, 전에 저장된 checkpoint 이하 파일들은 ./weight/exeperiments 이하로 timestamp 와 함께 copy 됨.
def _copy_to_experiment_dir(model_checkpoints_folder, model_name):
    now_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    new_exp_dir = os.path.join('./weights/experiments', model_name + '_checkpoints', now_time)
    if not os.path.exists(new_exp_dir):
        os.makedirs(new_exp_dir)
    for src in os.listdir(model_checkpoints_folder):
        shutil.copy(os.path.join(model_checkpoints_folder, src), new_exp_dir)

# train loop 에 해당하는 클래스
# multi-gpu 구현되지 않음. (나중에 추가 예정)
class Trainer(object):

    # model_dict 는 model 디렉토리 안의 파일명을 key, 그 안에 구현된 model class 명을 value 값으로 함.
    # 실험해야 할 model 파일이 늘어나면 이 구조로 dictionary 안에 담고, mymodel.yaml 에서 사용자가 model 종류를 직접 설정함.
    def __init__(self, dataset, base_model, config):
        self.dataset = dataset
        self.base_model = base_model
        self.config = config
        self.device = self._get_device()
        self.loss = nn.CrossEntropyLoss()
        self.model_dict = {"mymodel": MyModel}

    # cuda / cpu 선택
    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    # 사용자가 mymodel.yaml 에서 설정한 모델명이 실제 구현되어있는지, 또는 오타인지 한번 더 확인함.
    def _get_model(self):
        try:
            model = self.model_dict[self.base_model]
            return model
        except:
            raise ("Invalid model name. Pass one of the model dictionary.")

    # train loop 안에서 epoch 마다 실행 될 validation 로직.
    def _validate(self, epoch, net, criterion, valid_loader, best_acc):

        # 모델, 즉, net 을 .eval() 모드로 바꿈.
        net.eval()

        # validation 손실값을 담을 empty array 준비.
        h = np.array([])

        valid_loss = 0
        correct = 0
        total = 0

        # validation steps
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                h = np.append(h, loss.item())

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                utils.progress_bar(i, len(valid_loader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
                                   % (valid_loss / (i + 1), 100. * correct / total, correct, total))

            valid_loss = np.mean(h)

        # Save checkpoint.
        model_checkpoints_folder = os.path.join('./weights', 'checkpoints')

        # 해당 epoch 에서 정확도를 계산.
        # 최초 epoch 의 모델은 반드시 저장. 이후 epoch 부터는 바로 이전까지의 epoch 들에서 나온 acc 들과 비교하여
        # best 일때만 해당 epoch 모델을 저장함.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(model_checkpoints_folder, 'model.pth'))
            best_acc = acc

        # 다음 epoch 을 위해 다시 모델을 .train() 모드로 전환함.
        net.train()
        return valid_loss, best_acc

    # train 또는 inference 에서 모델 가중치를 불러오기 위한 함수.
    # mymodel.yaml 에서 resume 이 None 일경우, 가장 최신 checkpoint 를 불러옴.
    # 이전 모델을 불러오고 싶은 경우, 해당 폴더명(timestamp)을 resume 에 설정함.
    def _load_pre_trained_weights(self, model):
        best_acc = 0
        start_epoch = 0
        if self.config['resume'] is not None:
            try:
                checkpoints_folder = os.path.join('./weights/experiments', str(self.base_model) + '_checkpoints')
                checkpoint = torch.load(os.path.join(checkpoints_folder, self.config['resume'],'model.pth'))
                model.load_state_dict(checkpoint['net'])
                best_acc = checkpoint['acc']
                start_epoch = checkpoint['epoch']
                print('\n==> Resuming from checkpoint..')
            except FileNotFoundError:
                print("\nPre-trained weights not found. Training from scratch.")
        else:
            print("\nTraining from scratch.")
        return model, best_acc, start_epoch

    # train loop
    # train.py 의 main 함수에서 본 메서드를 실행.
    def train(self):
        # train.py 의 main 함수에서 데이터를 가져옴.
        train_loader, test_loader = self.dataset.get_data_loaders()

        model = self._get_model()
        model = model(**self.config['model'])
        model, best_acc, start_epoch = self._load_pre_trained_weights(model)
        model = model.to(self.device)
        model.train()

        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3fM\n' % parameters)

        criterion = self.loss.to(self.device)
        ## optimizer = optim.Adam(model.parameters(), 3e-4) #, weight_decay=eval(self.config['weight_decay']))
        optimizer = optim.SGD(model.parameters(), lr=self.config['learning_rate'], momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

        # save config file
        model_checkpoints_folder = os.path.join('./weights', 'checkpoints')
        _save_config_file(model_checkpoints_folder, str(self.base_model))

        history = {}
        history['train_loss'] = []
        history['valid_loss'] = []

        epochs = self.config['epochs']
        for e in range(start_epoch, start_epoch+epochs):
            h = np.array([])

            train_loss = 0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                h = np.append(h, loss.item())

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                utils.progress_bar(i, len(train_loader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
                                   % (train_loss / (i + 1), 100. * correct / total, correct, total))

            train_loss = np.mean(h)
            valid_loss, best_acc = self._validate(e, model, criterion, test_loader, best_acc)

            print('epoch [{}/{}], train_loss:{:.4f}, valid_loss:{:.4f}\n'.format(e + 1, start_epoch+epochs, train_loss, valid_loss))

            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)

            plt.figure(figsize=(15, 10))
            plt.plot(history['train_loss'], linewidth=2.0)
            plt.plot(history['valid_loss'], linewidth=2.0)
            plt.title('model loss.')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'valid'], loc='upper right')
            plt.savefig('./weights/checkpoints/loss.png')
            plt.close()

        print("--------------")
        print("Done training.")

        # copy and save trained model with config to experiments dir.
        _copy_to_experiment_dir(model_checkpoints_folder, str(self.base_model))

        print("--------------")
        print("All files saved.")