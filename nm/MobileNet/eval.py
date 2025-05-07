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
def evaluate(model, val_loader, device):
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
    with open(os.path.join('out','tmp','f.json'),'w') as wf:
        json.dump(dataList,wf,indent=2)
    accuracy = 100 * correct / total
    return accuracy


def eval():
    train_loader, val_loader = ImageNetLoader.loader(root=mysys.ImageNetRoot,num_workers=4,batch_size=64)
    weights_list=[]
    # MobileNetV2=['MobileNetV2',torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1),'cuda']
    # weights_list.append(MobileNetV2)
    # MobileNetV3_Large=['MobileNetV3_Large',torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1),'cuda']
    # weights_list.append(MobileNetV3_Large)
    MobileNetV3_Large=['MobileNetV3_Large',torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2),'cuda']
    weights_list.append(MobileNetV3_Large)
    # MobileNetV3_Small=['MobileNetV3_Small',torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1),'cuda']
    # weights_list.append(MobileNetV3_Small)
    # MobieNetV2_Q=torchvision.models.quantization.mobilenet_v2(weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1,quantize=True)
    # MobileNetV3_Large_Q=torchvision.models.quantization.mobilenet_v3_large(weights=MobileNet_V3_Large_QuantizedWeights.IMAGENET1K_QNNPACK_V1,quantize=True)
# ['MobieNetV2_Q',MobieNetV2_Q,'cpu'],['MobileNetV3_Large_Q',MobileNetV3_Large_Q,'cpu']
    # weights_list=[MobileNetV2,MobileNetV3_Large,MobileNetV3_Small]
    for weightsModel in weights_list:
        if weightsModel[2]=='cuda':
            device = torch.device("cuda")
        else:
            device =  torch.device("cpu")
        # 加载预训练的MobileNetV2模型
        model=weightsModel[1]
        model.to(device)
        model.eval()
        # 评估模型准确率
        accuracy = evaluate(model, val_loader, device)
        print("Accuracy of the {} on the ImageNet validation set: {}%".format(weightsModel[0],accuracy))


if __name__ == '__main__':
    freeze_support()
    eval()
