import torch
import torch.nn as nn

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU,  Dropout, \
    Sequential, Module

import math

#__all__ = ['MobileFacenet', 'mfacenet']

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Bottleneck(Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.res_layer = Sequential(
            #pw
            Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            BatchNorm2d(inp * expansion),
            PReLU(inp * expansion),
            # ReLU(inplace=True),

            #dw
            Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            BatchNorm2d(inp * expansion),
            PReLU(inp * expansion),
            # ReLU(inplace=True),

            #pw-linear
            Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            BatchNorm2d(oup)
        )

    def forward(self, x):
        if self.connect: #Residual
            return x + self.res_layer(x)
        else: #DResidual
            return self.res_layer(x)

class ConvBlock(Module): 
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = Sequential( Conv2d(inp, oup, k, s, p, groups=inp, bias=False), BatchNorm2d(oup))
        else:
            self.conv = Sequential( Conv2d(inp, oup, k, s, p, bias=False), BatchNorm2d(oup))
        if not linear:
            self.prelu = PReLU(oup)
    def forward(self, x):
        x = self.conv(x)
#        print('ConvBlock:{}'.format(x.size()))
        if self.linear:
            return x
        else:
            return self.prelu(x)

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 1, 2], #Dresidual
    [2, 64, 4, 1], #residual
    [4, 128, 1, 2], #Dresidual
    [2, 128, 6, 1], #residual
    [4, 128, 1, 2], #Dresidual
    [2, 128, 2, 1]  #residual
]


Mobilefacenet_ICCV_bottleneck_setting = [
    # t, c , n ,s
    [1, 64, 2, 1], #residual
    [2, 64, 1, 2], #Dresidual
    [2, 64, 8, 1], #residual
    [4, 128, 1, 2], #Dresidual
    [2, 128, 16, 1], #residual
    [4, 128, 1, 2], #Dresidual
    [2, 128, 4, 1]  #residual
]

#Mobilenetv2_bottleneck_setting = [
#    # t, c, n, s
#    [1, 16, 1, 1],
#    [6, 24, 2, 2],
#    [6, 32, 3, 2],
#    [6, 64, 4, 2],
#    [6, 96, 3, 1],
#    [6, 160, 3, 2],
#    [6, 320, 1, 1],
#]

class MobileFacenet(Module):
    def __init__(self, input_size, bottleneck_setting=Mobilefacenet_bottleneck_setting, feature_dim =128):
        super(MobileFacenet, self).__init__()
        self.setting_size=len(bottleneck_setting)
#        self.input = Sequential( Conv2d(3, 64, 3, 2, 1, bias=False), BatchNorm2d(64),PReLU(64))
        self.input = ConvBlock(3, 64, 3, 2, 1)
        if self.setting_size <=6 :
           self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)


        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0) # equivalent to conv_6_sep (insightface: fmobilefacenet)

        if input_size[0] == 112:
           self.linear7 = ConvBlock(512, 512, (7, 7), 1, 0, dw=True, linear=True)#gdc: conv_6_dw
        elif input_size[0] == 144:
           print('in 144')
           self.linear7 = ConvBlock(512, 512, (9, 9), 1, 0, dw=True, linear=True)#gdc: conv_6_dw
        else:
           print('in 244')
           self.linear7 = ConvBlock(512, 512, (14, 14), 1, 0, dw=True, linear=True)#gdc: conv_6_dw

#        self.output = Sequential( ConvBlock(512, 128, 1, 1, 0, linear=True), Flatten())
        self.output = Sequential( Flatten(),
                                  Linear(512 , feature_dim),
                                  BatchNorm1d(feature_dim))


        self._initialize_weights()

    def _make_layer(self, block, setting):
        modules = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    modules.append(block(self.inplanes, c, s, t))
                else:
                    modules.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return Sequential(*modules)

    def forward(self, x):

        x = self.input(x)
#        print("x:{}".format(x.size()))
#        print('setting_size:{}'.format(self.setting_size))
        if self.setting_size <=6 :
           x = self.dw_conv1(x)
        x = self.blocks(x)

        x = self.conv2(x)

        x = self.linear7(x)
#        print('linear7:{}'.format(x.size()))
        x = self.output(x)
#        print('output:{}'.format(x.size()))

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

#def Mobilenetv2(feature_dim=512, **kwargs):
#    """Constructs a Mobilefacenet.

#    Args:

#    """
#    model = MobileFacenet(bottleneck_setting = Mobilenetv2_bottleneck_setting)
#
#    return model

def mfacenet(input_size, feature_dim=256, **kwargs):
    """Constructs a Mobilefacenet.

    Args:

    """
    model = MobileFacenet(input_size = input_size, bottleneck_setting = Mobilefacenet_bottleneck_setting, feature_dim=feature_dim)

    return model

def mfacenet_iccv(input_size, feature_dim=512, **kwargs):
    """Constructs a Mobilefacenet.

    Args:

    """
    model = MobileFacenet(input_size = input_size, bottleneck_setting = Mobilefacenet_ICCV_bottleneck_setting, feature_dim=feature_dim)

    return model

def getmodel_byname(model_name, input_size=(112, 112)):
    if model_name == "mfacenet":
        print("[PARAM] Model type is : " + model_name)
        model = mfacenet(input_size=input_size)

    # elif model_name == "resnet34":
    #     print("[PARAM] Model type is : " + model_name)
    #     model = resnet34()

    else:
        print("Model type not recognized: Exiting")
        exit()
    return model


if __name__ == "__main__":
    from torch.autograd import Variable
#because of depthwise in linear 7 therefore, it need two images for batch norm.
    input = Variable(torch.FloatTensor(2, 3, 112, 112))
#    net = MobileFacenet()
    net = mfacenet(input_size=[112, 112], feature_dim=256)
    print(net)
#    net1= nn.Sequential(*list(net.children())[:-1#])
#    print(net1)
    x = net(input)
    print(x)
    print(net.output.parameters())
