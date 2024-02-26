import time
import os

import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# 忽略烦人的红色提示
import warnings

warnings.filterwarnings("ignore")

from torchvision import models
import torch.optim as optim

from torchvision import transforms

from torchvision import datasets

from torch.utils.data import DataLoader

import multiprocessing

import MyCBAM
import final

def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)


    #在不同的神经网络中，会用到不同的预处理参数

    # 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

    # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

    dataset_dir = 'data_split'
    train_path = os.path.join(dataset_dir, 'train')
    test_path = os.path.join(dataset_dir, 'val')
    print('训练集路径', train_path)
    print('测试集路径', test_path)



    # 载入训练集
    train_dataset = datasets.ImageFolder(train_path, train_transform)

    # 载入测试集
    test_dataset = datasets.ImageFolder(test_path, test_transform)

    class_names = train_dataset.classes
    n_class = len(class_names)
    idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}
    #%%

    BATCH_SIZE = 16

    # 训练集的数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=0  #windows系统这里一定要设置成单线程！！！
                             )

    # 测试集的数据加载器
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=0  #windows系统这里一定要设置成单线程！！！
                            )

    images, labels = next(iter(train_loader))

    # 自搭resnet34模型
    #载入预训练权重参数
    model=final.ResNet.resnet34()
    model_weight_path = "./resnet34-pretrain.pth"
    misssing_keys,unexpected_keys=model.load_state_dict(torch.load(model_weight_path),strict=False)
    inchannel = model.fc.in_features
    model.fc=nn.Linear(inchannel,n_class)

    #训练用模型
    # model=models.resnet50(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, n_class)
    # optimizer = optim.Adam(model.fc.parameters(),lr=0.1)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练轮次 Epoch
    EPOCHS = 20
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # 遍历每个 EPOCH
    done=0
    for epoch in tqdm(range(EPOCHS)):

        model.train()
        avg_loss=0
        total_num=0
        for images, labels in train_loader:  # 获取训练集的一个 batch，包含数据和标注
            images = images.to(device)
            labels = labels.to(device)

            reconstructions = model.forward(images)  # 前向预测，获得当前 batch 的预测结果
            loss = criterion(reconstructions, labels)  # 比较预测结果和标注，计算当前 batch 的交叉熵损失函数
            avg_loss+=loss.item()
            total_num+=1
            optimizer.zero_grad()
            loss.backward()  # 损失函数对神经网络权重反向传播求梯度
            optimizer.step()  # 优化更新神经网络权重
        print('\n本轮训练的平均损失值为 {:.3f} %'.format(avg_loss/total_num))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader):  # 获取测试集的一个 batch，包含数据和标注
            images = images.to(device)
            labels = labels.to(device)
            classes = model.forward(images)  # 前向预测，获得当前 batch 的预测置信度
            _, preds = torch.max(classes, 1)  # 获得最大置信度对应的类别，作为预测结果
            total += labels.size(0)
            correct += (preds == labels).sum()  # 预测正确样本个数

        print('测试集上的准确率为 {:.3f} %'.format(100 * correct / total))

if __name__ == '__main__':
    main()





