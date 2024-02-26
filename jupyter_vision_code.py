import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

# 胶囊网络基本层
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Non_local注意力块
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

        # output embeding和Batch norm
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        # 相当于计算value
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # 相当于计算query
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # 相当于计算key
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # 计算attention map
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        # output
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        # 残差连接
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


# 注意力CBAM模块
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

        self.attention_block = CBAM(out_channel)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention_block(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.attention_block = CBAM(out_channel)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention_block(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.nl1 = _NonLocalBlockND(64)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.nl2 = _NonLocalBlockND(128)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.nl3 = _NonLocalBlockND(256)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.nl4 = _NonLocalBlockND(512)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.nl1(x)

        x = self.layer2(x)
        x = self.nl2(x)

        x = self.layer3(x)
        x = self.nl3(x)

        x = self.layer4(x)
        x = self.nl4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    def resnet34(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnet34-333f7ec4.pth
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

    def resnet50(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnet50-19c8e357.pth
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

    def resnet101(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

    def resnext50_32x4d(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
        groups = 32
        width_per_group = 4
        return ResNet(Bottleneck, [3, 4, 6, 3],
                      num_classes=num_classes,
                      include_top=include_top,
                      groups=groups,
                      width_per_group=width_per_group)

    def resnext101_32x8d(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
        groups = 32
        width_per_group = 8
        return ResNet(Bottleneck, [3, 4, 23, 3],
                      num_classes=num_classes,
                      include_top=include_top,
                      groups=groups,
                      width_per_group=width_per_group)


# train and test part

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
from sklearn.metrics import classification_report
import numpy as np


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    # 在不同的神经网络中，会用到不同的预处理参数

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
    idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
    # %%

    BATCH_SIZE = 16

    # 训练集的数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=8  # windows系统这里一定要设置成单线程！！！
                              )

    # 测试集的数据加载器
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=8  # windows系统这里一定要设置成单线程！！！
                             )

    images, labels = next(iter(train_loader))

    # 自搭resnet34模型
    # 载入预训练权重参数
    model = ResNet.resnet34()
    model_weight_path = "./resnet34-pretrain.pth"
    misssing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)
    inchannel = model.fc.in_features
    model.fc = nn.Linear(inchannel, n_class)

    # 训练用模型
    # model = torchvision.models.resnet50(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, n_class)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练轮次 Epoch
    EPOCHS = 120
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # 遍历每个 EPOCH
    done = 0
    for epoch in tqdm(range(EPOCHS)):

        model.train()
        avg_loss = 0
        total_num = 0
        for images, labels in train_loader:  # 获取训练集的一个 batch，包含数据和标注
            images = images.to(device)
            labels = labels.to(device)

            reconstructions = model.forward(images)  # 前向预测，获得当前 batch 的预测结果
            loss = criterion(reconstructions, labels)  # 比较预测结果和标注，计算当前 batch 的交叉熵损失函数
            avg_loss += loss.item()
            total_num += 1
            optimizer.zero_grad()
            loss.backward()  # 损失函数对神经网络权重反向传播求梯度
            optimizer.step()  # 优化更新神经网络权重
        print('\n本轮训练的平均损失值为 {:.3f} '.format(avg_loss / total_num))

    model.eval()
    with torch.no_grad():
        all_preds = []  # 用于收集所有预测标签
        all_labels = []  # 用于收集所有真实标签
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader):  # 获取测试集的一个 batch，包含数据和标注
            images = images.to(device)
            labels = labels.to(device)
            classes = model.forward(images)  # 前向预测，获得当前 batch 的预测置信度
            _, preds = torch.max(classes, 1)  # 获得最大置信度对应的类别，作为预测结果
            total += labels.size(0)
            correct += (preds == labels).sum()  # 预测正确样本个数
            # 保存预测和实际标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        print('测试集上的准确率为 {:.3f} %'.format(100 * correct / total))
        # 计算宏精确度、宏召回率和宏 F1 值
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']

        print('宏精确率 (Macro-Precision): {:.3f}'.format(macro_precision))
        print('宏召回率 (Macro-Recall): {:.3f}'.format(macro_recall))
        print('宏 F1 值 (Macro-F1): {:.3f}'.format(macro_f1))


if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

# 胶囊网络基本层
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Non_local注意力块
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

        # output embeding和Batch norm
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        # 相当于计算value
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # 相当于计算query
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # 相当于计算key
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # 计算attention map
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        # output
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        # 残差连接
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


# 注意力CBAM模块
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

        self.attention_block = CBAM(out_channel)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention_block(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.attention_block = CBAM(out_channel)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention_block(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.nl1 = _NonLocalBlockND(64)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.nl2 = _NonLocalBlockND(128)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.nl3 = _NonLocalBlockND(256)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.nl4 = _NonLocalBlockND(512)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.nl1(x)

        x = self.layer2(x)
        x = self.nl2(x)

        x = self.layer3(x)
        x = self.nl3(x)

        x = self.layer4(x)
        x = self.nl4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    def resnet34(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnet34-333f7ec4.pth
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

    def resnet50(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnet50-19c8e357.pth
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

    def resnet101(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

    def resnext50_32x4d(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
        groups = 32
        width_per_group = 4
        return ResNet(Bottleneck, [3, 4, 6, 3],
                      num_classes=num_classes,
                      include_top=include_top,
                      groups=groups,
                      width_per_group=width_per_group)

    def resnext101_32x8d(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
        groups = 32
        width_per_group = 8
        return ResNet(Bottleneck, [3, 4, 23, 3],
                      num_classes=num_classes,
                      include_top=include_top,
                      groups=groups,
                      width_per_group=width_per_group)


# train and test part

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
from sklearn.metrics import classification_report
import numpy as np


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    # 在不同的神经网络中，会用到不同的预处理参数

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
    idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
    # %%

    BATCH_SIZE = 16

    # 训练集的数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=8  # windows系统这里一定要设置成单线程！！！
                              )

    # 测试集的数据加载器
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=8  # windows系统这里一定要设置成单线程！！！
                             )

    images, labels = next(iter(train_loader))

    # 自搭resnet34模型
    # 载入预训练权重参数
    model = ResNet.resnet34()
    model_weight_path = "./resnet34-pretrain.pth"
    misssing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)
    inchannel = model.fc.in_features
    model.fc = nn.Linear(inchannel, n_class)

    # 训练用模型
    # model = torchvision.models.resnet50(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, n_class)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练轮次 Epoch
    EPOCHS = 120
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # 遍历每个 EPOCH
    done = 0
    for epoch in tqdm(range(EPOCHS)):

        model.train()
        avg_loss = 0
        total_num = 0
        for images, labels in train_loader:  # 获取训练集的一个 batch，包含数据和标注
            images = images.to(device)
            labels = labels.to(device)

            reconstructions = model.forward(images)  # 前向预测，获得当前 batch 的预测结果
            loss = criterion(reconstructions, labels)  # 比较预测结果和标注，计算当前 batch 的交叉熵损失函数
            avg_loss += loss.item()
            total_num += 1
            optimizer.zero_grad()
            loss.backward()  # 损失函数对神经网络权重反向传播求梯度
            optimizer.step()  # 优化更新神经网络权重
        print('\n本轮训练的平均损失值为 {:.3f} '.format(avg_loss / total_num))

    model.eval()
    with torch.no_grad():
        all_preds = []  # 用于收集所有预测标签
        all_labels = []  # 用于收集所有真实标签
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader):  # 获取测试集的一个 batch，包含数据和标注
            images = images.to(device)
            labels = labels.to(device)
            classes = model.forward(images)  # 前向预测，获得当前 batch 的预测置信度
            _, preds = torch.max(classes, 1)  # 获得最大置信度对应的类别，作为预测结果
            total += labels.size(0)
            correct += (preds == labels).sum()  # 预测正确样本个数
            # 保存预测和实际标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        print('测试集上的准确率为 {:.3f} %'.format(100 * correct / total))
        # 计算宏精确度、宏召回率和宏 F1 值
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']

        print('宏精确率 (Macro-Precision): {:.3f}'.format(macro_precision))
        print('宏召回率 (Macro-Recall): {:.3f}'.format(macro_recall))
        print('宏 F1 值 (Macro-F1): {:.3f}'.format(macro_f1))


if __name__ == '__main__':
    main()
