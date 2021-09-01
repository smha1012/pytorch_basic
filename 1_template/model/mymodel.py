#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : model
# @Date : 2021-09-01-09-06
# @Project : pytorch_basic
# @Author : seungmin

import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    # 실험하고 싶은 model 종류가 여러가지 일 때, 유용한 클래스.
    # dictionary 를 구현하지 않고, class 별로 파일을 따로 만들어도 상관없음.
    def __init__(self, base_model):
        super(MyModel, self).__init__()
        self.model_dict = {"my_model_class_1": models.resnet50(pretrained=False),
                            "my_model_class_2": models.resnet152(pretrained=False)}

        mymodel = self._get_basemodel(base_model)
        self.features = nn.Sequential(*list(mymodel.children())[:-1])

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file.")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        return h