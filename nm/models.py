import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights,MobileNet_V3_Small_Weights
from torchvision.models.quantization.mobilenetv2 import MobileNet_V2_QuantizedWeights
from torchvision.models.quantization.mobilenetv3 import MobileNet_V3_Large_QuantizedWeights



def getModels():
    modelMean=[0.485, 0.456, 0.406]
    modelStd=[0.229, 0.224, 0.225]
    modelList=[
        {
            'name':'EfficientNet_B0',
            'model':torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1),
            'resize':256,
            'crop':224,
            'mean':modelMean,
            'std':modelStd,
            'params':5.3,
            'gflops':0.39,
        },
        {
            'name':'EfficientNet_B1',
            'model':torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1),
            'resize':256,
            'crop':240,
            'mean':modelMean,
            'std':modelStd,
            'params':7.8,
            'gflops':0.69,
        },
        {
            'name':'EfficientNet_B2',
            'model':torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1),
            'resize':288,
            'crop':288,
            'mean':modelMean,
            'std':modelStd,
            'params':9.1,
            'gflops':1.09,
        },
        {
            'name':'EfficientNet_B3',
            'model':torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1),
            'resize':320,
            'crop':300,
            'mean':modelMean,
            'std':modelStd,
            'params':12.2,
            'gflops':1.83,
        },
        {
            'name':'EfficientNet_B4',
            'model':torchvision.models.efficientnet_b4(weights=torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1),
            'resize':384,
            'crop':380,
            'mean':modelMean,
            'std':modelStd,
            'params':19.3,
            'gflops':4.39,
        },
        {
            'name':'EfficientNet_B5',
            'model':torchvision.models.efficientnet_b5(weights=torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1),
            'resize':456,
            'crop':456,
            'mean':modelMean,
            'std':modelStd,
            'params':30.4,
            'gflops':10.27,
        },
        {
            'name':'EfficientNet_B6',
            'model':torchvision.models.efficientnet_b6(weights=torchvision.models.EfficientNet_B6_Weights.IMAGENET1K_V1),
            'resize':528,
            'crop':528,
            'mean':modelMean,
            'std':modelStd,
            'params':43,
            'gflops':19.07,
        },
        {
            'name':'EfficientNet_B7',
            'model':torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1),
            'resize':600,
            'crop':600,
            'mean':modelMean,
            'std':modelStd,
            'params':66.3,
            'gflops':37.75,
        },
        {
            'name':'EfficientNet_V2_S',
            'model':torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
            'resize':384,
            'crop':384,
            'mean':modelMean,
            'std':modelStd,
            'params':21.5,
            'gflops':8.37,
        }, 
        {
            'name':'EfficientNet_V2_M',
            'model':torchvision.models.efficientnet_v2_m(weights=torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1),
            'resize':480,
            'crop':480,
            'mean':[0.5, 0.5, 0.5],
            'std':[0.5, 0.5, 0.5],
            'params':54.1,
            'gflops':24.58,
        }, 
        {
            'name':'EfficientNet_V2_L',
            'model':torchvision.models.efficientnet_v2_l(weights=torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1),
            'resize':600,
            'crop':600,
            'mean':[0.5, 0.5, 0.5],
            'std':[0.5, 0.5, 0.5],
            'params':118.5,
            'gflops':56.08,
        }, 
        {
            'name':'ConvNeXt_Base',
            'model':torchvision.models.convnext_base(weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
            'resize':232,
            'crop':224,
            'mean':modelMean,
            'std':modelStd,
            'params':88.6,
            'gflops':15.36,
        },
        {
            'name':'ConvNeXt_Large',
            'model':torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1),
            'resize':232,
            'crop':224,
            'mean':modelMean,
            'std':modelStd,
            'params':197.8,
            'gflops':34.36,
        },
        {
            'name':'ConvNeXt_Small',
            'model':torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1),
            'resize':230,
            'crop':224,
            'mean':modelMean,
            'std':modelStd,
            'params':50.2,
            'gflops':8.68,
        },
        {
            'name':'ConvNeXt_Tiny',
            'model':torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
            'resize':236,
            'crop':224,
            'mean':modelMean,
            'std':modelStd,
            'params':28.6,
            'gflops':4.46,
        },
        {
            'name':'MobileNet_V2',
            'model':torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1),
            'resize':256,
            'crop':224,
            'mean':modelMean,
            'std':modelStd,
            'params':3.5,
            'gflops':0.3,
        },
        {
            'name':'MobileNet_V3_Large',
            'model':torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
            'resize':256,
            'crop':224,
            'mean':modelMean,
            'std':modelStd,
            'params':5.5,
            'gflops':0.22,
        },          
    ]

    return modelList
