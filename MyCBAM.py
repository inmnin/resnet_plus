import torch
import torch.nn as nn
import torch.nn.functional as F
import math



#注意力CBAM模块
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
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
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out




#主体resnet结构

#resnet18和resnet34的基本残差结构
class BasicBlock(nn.Module):
    expansion = 1 #每一个残差结构的各个卷积层的卷积核的个数一不一样

    #初始化函数
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
    #几个参数的含义分别是：输入特征矩阵深度，输出特征矩阵深度，步数，和下采样实体(None和notNone不是布尔变量，而是对应着两种具体的下采样方法：layer方法和Squential方法)
     super(BasicBlock,self).__init__()
     #第一个卷积层
     self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
     self.bn1=nn.BatchNorm2d(out_channel)
     self.relu=nn.ReLU(inplace=True)

     #第二层
     self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1,bias=False)
     self.bn2=nn.BatchNorm2d(out_channel)
     self.downsample=downsample

    #正向传播过程
    def forward(self,x): #输入特征矩阵
        identity=x  #捷径（shortcut，这就是下采样里面的那个东西）上的输出值
        if self.downsample is not None:
            identity=self.downsample(x)

        #第一个卷积层
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        #第二个卷积层
        out=self.conv2(out)
        out=self.bn2(out)

        #下采样加和
        out += identity
        out=self.relu(out)

        return out

#resnet50,101,152的基本残差结构
class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        #第一个卷积层
        self.conv1 = nn.Conv2d(in_channel=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        #第二个卷积层
        self.conv2= nn.Conv2d(in_channel=out_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        #第三个（最终）卷积层
        self.conv3=nn.Conv2d(in_channel=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, padding=1,
                               bias=False)
        self.bn3=nn.BatchNorm2d(out_channel)

        #一般组件
        self.relu=nn.ReLu(inplace=True) #这个inplace是干嘛的
        self.downsample=downsample

    def forward(self,x):
        indentity=x
        if self.downsample is not None:
            indentity=self.downsample(x)

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        out+=indentity
        out=self.relu(out)

        return out

#构造resnet
class ResNet(nn.Module):
    def __init__(self,block,block_num,classes_num=1000,include_top=True):
        # 其中block_num是一个向量，记录了各个主层中的卷积层的个数
        # classes_num记录了分类任务个一共有几个类别
        # include_top在以后扩展resnet时会用到
        super(ResNet,self).__init__()
        self.include_top=include_top
        self.in_channel=64 #这个是经过了一开始的池化层之后的特征矩阵的深度

        #初始卷积层
        self.conv1=nn.Conv2d(3,self.in_channel,kernel_size=7,stride=2,padding=3,bias=False) #输入的深度一般为RGB图像，所以in_channel设置为3（对应RGB三色通道）
        self.bn1=nn.BatchNorm2d(self.in_channel)
        self.relu=nn.ReLU()

        #最大池化层
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        #各个主层
        self.layer1=self._make_layer(block,64,block_num[0])
        self.layer2=self._make_layer(block,128,block_num[1],stride=2)
        self.layer3=self._make_layer(block,256,block_num[2],stride=2)
        self.layer4=self._make_layer(block,512,block_num[3],stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512*block.expansion,classes_num)

        #对整个神经网络进行初始化
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')



    def _make_layer(self,block,channel,block_num,stride=1):  #创建层函数
        #channel表示一个主层中第一个卷积层的卷积核个数（在resnet50以上的版本中，最后一个卷积层的卷积核个数是前面卷积层的卷积核个数的四倍）
        #block_num表示一个主层中有几个残差结构
        downsample=None
        if stride != 1 or self.in_channel != channel*block_num:
            downsample=nn.Sequential(nn.Conv2d(self.in_channel,channel*block.expansion,kernel_size=1,stride=stride,bias=False),nn.BatchNorm2d(channel*block.expansion))

        layer=[]
        layer.append(block(self.in_channel,channel,stride=stride,downsample=downsample))
        self.in_channel = channel*block.expansion
        # layer.append(CBAM(channel))  #添加注意力CBAM块

        for _ in range(1,block_num):
            layer.append(block(self.in_channel,channel))
        # layer.append(CBAM(channel))  #添加注意力CBAM块
        return nn.Sequential(*layer)

    #整个神经网络的正向传播
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x,1)
            x = self.fc(x)
        return x

    def resnet34(classes_num=1000,include_top=True):
        return ResNet(BasicBlock,[3,4,6,3],classes_num=classes_num,include_top=include_top)

    def resnet50(classes_num=1000,include_top=True):
        return ResNet(Bottleneck,[3,4,23,3],classes_num=classes_num,include_top=include_top)







