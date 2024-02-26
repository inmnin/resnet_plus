import torch.nn as nn
import torch

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

        for _ in range(1,block_num):
            layer.append(block(self.in_channel,channel))

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







