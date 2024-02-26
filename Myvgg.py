import torch
import torch.nn as nn

#vgg网络参数配置，一个数字表示一个有多少个卷积核的卷积层，一个字母表示一个池化层
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_features(cfg:list):
    layers=[]
    in_channels=3
    for v in cfg:
        if v =='M':
            layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            cvd2=nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            layers+=[cvd2,nn.ReLU(True)]
            in_channels=v

    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self,features,class_num=1000,init_weight=False):
        super(VGG,self).__init__()
        self.features=features
        self.classifier=nn.Sequential(
            nn.Dropout(p=0.5),  #随机失活50%的神经元  失活并不能使channel数减少，只是使一些channel失去作用而已
            nn.Linear(512*7*7,2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(True),
            nn.Linear(2048,class_num)
            )
        if init_weight:
            self._intialize_weights()


    def _intialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

                if isinstance(m,nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias,0)  #在做最终分类判断前，全连接层的偏置项bias一定不可以忽略

    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1) #展平处理    start_dim指的是在哪个维度开始处理，由于在实际训练过程中，第0个维度是batch相关信息，所以要从第1个维度开始展平
        x=self.classifier(x)
        return x

def vgg(model_name="vgg16",**kwargs): #第二个参数是一个可变长字典，可用于指定分类个数，以及是否初始化权重
    try:
        cfg=cfgs[model_name]
    except:
        print("Warning: model number {} not in current versions!".format(model_name))
        exit(-1)

    model=VGG(make_features(cfg),**kwargs)
    return model
