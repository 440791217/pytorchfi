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

from loaders import ImageNetLoader
import mysys
from tqdm import tqdm
import os
import json
import nm.models

# # 定义评估函数
# def evaluate(model, val_loader, device):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in tqdm(val_loader, desc="Evaluating"):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)          
#             total += labels.size(0)
#             for out in outputs:
#                 print(out.tolist())
#             # for l in labels:
#             #     print(l.item())
#             # print('size',labels.size(0))
#             correct += (predicted == labels).sum().item()
#     accuracy = 100 * correct / total
#     return accuracy

# 定义评估函数
def evaluate(model, val_loader, device,fname):
    model.eval()
    dataList=[]
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 将输出转换为置信度（概率分布）
            confidences = torch.softmax(outputs, dim=1)
            # print(confidences[i])
            # 找出每个样本的最大置信度及其对应的类别索引
            max_confidences, predicted = torch.max(confidences, dim=1)
            confList=confidences.tolist()
            total += labels.size(0)
            for i in range(len(max_confidences)):
                data={
                    'label':labels[i].item(),
                    'predicted':predicted[i].item(),
                    'max_confidences':round(max_confidences[i].item(),4),
                    'confidences':confList[i],
                }
                dataList.append(data)
                # print(data)
            # with open(os.path.join('out','tmp','f.txt'),'w') as wf:
            #     json.dump(dataList,wf,indent=2)
            # 打印每个样本的最大置信度和预测类别
            # for i in range(len(max_confidences)):
            #     print(f"Sample {i}: Predicted class {predicted[i].item()}, Confidence {max_confidences[i].item():.4f}")
            correct += (predicted == labels).sum().item()
    with open(os.path.join('out','tmp','{}.json'.format(fname)),'w') as wf:
        json.dump(dataList,wf,indent=2)
    accuracy = 100 * correct / total
    return accuracy


def eval():
#     weights_list=[]
#     # MobileNetV2=['MobileNetV2',torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1),'cuda']
#     # weights_list.append(MobileNetV2)
#     # MobileNetV3_Large=['MobileNetV3_Large',torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1),'cuda']
#     # weights_list.append(MobileNetV3_Large)
#     # MobileNetV3_Large=['MobileNetV3_Large',torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2),'cuda']
#     # weights_list.append(MobileNetV3_Large)
#     # MobileNetV3_Small=['MobileNetV3_Small',torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1),'cuda']
#     # weights_list.append(MobileNetV3_Small)
#     # MobieNetV2_Q=torchvision.models.quantization.mobilenet_v2(weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1,quantize=True)
#     # MobileNetV3_Large_Q=torchvision.models.quantization.mobilenet_v3_large(weights=MobileNet_V3_Large_QuantizedWeights.IMAGENET1K_QNNPACK_V1,quantize=True)
    
#     #################efficientnet
#     transform={
#         'resize':256,
#         'crop':224,
#         'mean':[0.485, 0.456, 0.406],
#         'std':[0.229, 0.224, 0.225]
#     }
#     EfficientNet_B0=['EfficientNet_B0',torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1),'cuda',transform]
#     weights_list.append(EfficientNet_B0)
# #
#     transform={
#         'resize':256,
#         'crop':240,
#         'mean':[0.485, 0.456, 0.406],
#         'std':[0.229, 0.224, 0.225]
#     }    
#     EfficientNet_B1=['EfficientNet_B1',torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1),'cuda',transform]
#     weights_list.append(EfficientNet_B1)
# #
#     transform={
#         'resize':288,
#         'crop':288,
#         'mean':[0.485, 0.456, 0.406],
#         'std':[0.229, 0.224, 0.225]
#     }  
#     EfficientNet_B2=['EfficientNet_B2',torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1),'cuda',transform]
#     weights_list.append(EfficientNet_B2)
#     #
#     transform={
#         'resize':320,
#         'crop':300,
#         'mean':[0.485, 0.456, 0.406],
#         'std':[0.229, 0.224, 0.225]
#     }  
#     EfficientNet_B3=['EfficientNet_B3',torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1),'cuda',transform]
#     weights_list.append(EfficientNet_B3)
#     #
#     transform={
#         'resize':384,
#         'crop':380,
#         'mean':[0.485, 0.456, 0.406],
#         'std':[0.229, 0.224, 0.225]
#     }  
#     EfficientNet_B4=['EfficientNet_B4',torchvision.models.efficientnet_b4(weights=torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1),'cuda',transform]
#     weights_list.append(EfficientNet_B4)
#     #
#     transform={
#         'resize':456,
#         'crop':456,
#         'mean':[0.485, 0.456, 0.406],
#         'std':[0.229, 0.224, 0.225]
#     }  
#     EfficientNet_B5=['EfficientNet_B5',torchvision.models.efficientnet_b5(weights=torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1),'cuda',transform]
#     weights_list.append(EfficientNet_B5)
#     #
#     transform={
#         'resize':528,
#         'crop':528,
#         'mean':[0.485, 0.456, 0.406],
#         'std':[0.229, 0.224, 0.225]
#     }  
#     EfficientNet_B6=['EfficientNet_B6',torchvision.models.efficientnet_b6(weights=torchvision.models.EfficientNet_B6_Weights.IMAGENET1K_V1),'cuda',transform]
#     weights_list.append(EfficientNet_B6)
#     #
#     transform={
#         'resize':600,
#         'crop':600,
#         'mean':[0.485, 0.456, 0.406],
#         'std':[0.229, 0.224, 0.225]
#     }  
#     EfficientNet_B7=['EfficientNet_B7',torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1),'cuda',transform]
#     weights_list.append(EfficientNet_B7)

#     ##################efficientnetv2
#     transform={
#         'resize':384,
#         'crop':384,
#         'mean':[0.485, 0.456, 0.406],
#         'std':[0.229, 0.224, 0.225]
#     }      
#     EfficientNet_V2_S=['EfficientNet_V2_S',torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),'cuda',transform]
#     weights_list.append(EfficientNet_V2_S)
#     transform={
#         'resize':480,
#         'crop':480,
#         'mean':[0.485, 0.456, 0.406],
#         'std':[0.229, 0.224, 0.225]
#     }  
#     EfficientNet_V2_M=['EfficientNet_V2_M',torchvision.models.efficientnet_v2_m(weights=torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1),'cuda',transform]
#     weights_list.append(EfficientNet_V2_M)
#     transform={
#         'resize':480,
#         'crop':480,
#         'mean':[0.5, 0.5, 0.5],
#         'std':[0.5, 0.5, 0.5]
#     }  
#     EfficientNet_V2_L=['EfficientNet_V2_L',torchvision.models.efficientnet_v2_l(weights=torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1),'cuda']
#     weights_list.append(EfficientNet_V2_L)
    models=nm.models.getModels()
    for item in models:
        train_loader, val_loader = ImageNetLoader.loader(root=mysys.ImageNetRoot,num_workers=4,batch_size=32,
                                                         resize=item['resize'],crop=item['crop'],mean=item['mean'],std=item['std'])
        device = torch.device("cuda")
        # 加载预训练的MobileNetV2模型
        model=item['model']
        model.to(device)
        # 评估模型准确率
        fname=item['name']
        accuracy = evaluate(model, val_loader, device,fname)
        print("Accuracy of the {} on the ImageNet validation set: {}%".format(fname,accuracy))


if __name__ == '__main__':
    freeze_support()
    eval()
