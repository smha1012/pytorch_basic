#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : inference
# @Date : 2021-09-01-09-05
# @Project : pytorch_basic
# @Author : seungmin

import os, yaml, itertools

import torch
import numpy as np

import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import confusion_matrix

from utils.dataloader import MyTestSetWrapper

from model.mymodel import MyModel

## load model
def _get_model(base_model):
    model_dict = {"mymodel": MyModel}

    try:
        model = model_dict[base_model]
        return model
    except:
        raise ("Invalid model name. Pass one of the model dictionary.")

def _load_weights(model, load_from, base_model):
    try:
        checkpoints_folder = os.path.join('./weights/experiments', str(base_model) + '_checkpoints')
        checkpoint = torch.load(os.path.join(checkpoints_folder, load_from, 'model.pth'))
        model.load_state_dict(checkpoint['net'])
        print('\n==> Resuming from checkpoint..')
    except FileNotFoundError:
        print("\nWeights for inference not found.")
    return model

def _load_weights_from_recent(model):
    try:
        checkpoints_folder = os.path.join('./weights', 'checkpoints')
        checkpoint = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
        model.load_state_dict(checkpoint['net'])
        print('\n==> Resuming from checkpoint..')
    except FileNotFoundError:
        print("\nWeights for inference not found.")
    return model

## roc curve, confusion matrix
def plt_roc(test_y, probas_y, plot_micro=False, plot_macro=False):
    assert isinstance(test_y, list) and isinstance(probas_y, list), 'the type of input must be list'
    skplt.metrics.plot_roc(test_y, probas_y, plot_micro=plot_micro, plot_macro=plot_macro,
                           classes_to_plot=0,
                           figsize=(15,15))
    plt.savefig('roc_auc_curve.png')
    plt.close()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    refence:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()

## main
def main(model_name):
    checkpoints_folder = os.path.join('./weights', 'checkpoints')
    print(os.listdir(checkpoints_folder))
    config = yaml.load(open(checkpoints_folder + '/' + str(model_name) + ".yaml", "r"), Loader=yaml.FullLoader)
    device = config['inference_device']
    print(device)

    ## get class names
    classes_txt = os.listdir(config['train']['train_path'])

    testset = MyTestSetWrapper(**config['test'])

    ## model load
    # model topology
    model = _get_model(model_name)
    model = model(**config['model'])

    # model weight
    if config['resume'] != "None":
        model = _load_weights(model, config['resume'], model_name)
        model = model.to(device)
    else:
        model = _load_weights_from_recent(model)
        model = model.to(device)
    model.eval()

    ## test loader
    test_loader = testset.get_test_loaders()

    correct = 0
    total = 0

    pred_y = []
    test_y = []
    probas_y = []
    myclass = [0, 1]

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)

            probas_y.extend(outputs.data.cpu().numpy().tolist())
            pred_y.extend(outputs.data.cpu().max(1, keepdim=True)[1].numpy().flatten().tolist())
            test_y.extend(labels.data.cpu().numpy().flatten().tolist())

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            print("\nLabel: {} / :Logit: {}".format(labels, predicted))
            print("Predicted: ", " ".join('%5s' % classes_txt[predicted[j]].split('.')[0] for j in range(config['test']['batch_size'])))

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy of the network on the {} test images: {:.4f}".format(100, 100 * correct / total))

    confusion = confusion_matrix(pred_y, test_y)
    plot_confusion_matrix(confusion,
                          classes=myclass,
                          title='Confusion matrix')
    plt_roc(test_y, probas_y)


if __name__ == "__main__":
    main("mymodel")