#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : train
# @Date : 2021-09-01-09-05
# @Project : pytorch_basic
# @Author : seungmin

import yaml

from train.trainer import Trainer
from utils.dataloader import MyTrainSetWrapper

def main(model_name):
    # yaml 로드
    config = yaml.load(open("./config/" + str(model_name) + ".yaml", "r"), Loader=yaml.FullLoader)
    trainset = MyTrainSetWrapper(**config['train'])

    # Trainer 클래스 초기화. train 실행.
    downstream = Trainer(trainset, model_name, config)
    downstream.train()

if __name__ == "__main__":
    main("mymodel")