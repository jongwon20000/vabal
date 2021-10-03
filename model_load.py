import torch
from models import *

def model_loader(model_name):
        
    if model_name is 'RESNET18':
        net = ResNet18()
        input_chs = [64, 128, 256, 512]
        
    elif model_name is 'VGG16':
        net = VGG(model_name)
        input_chs = [64, 128, 256, 512, 512]
        
    else:
        raise NameError('UNKNOWN MODEL NAME')
        
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()        
        
    return (net, input_chs)

