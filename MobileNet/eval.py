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


# 定义评估函数
def evaluate(model, val_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def eval():
    train_loader, val_loader = ImageNetLoader.loader(root=mysys.ImageNetRoot)
    MobileNetV2=['MobileNetV2',torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1),'cuda']
    MobileNetV3_Large=['MobileNetV3_Large',torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1),'cuda']
    MobileNetV3_Small=['MobileNetV3_Small',torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1),'cuda']
    # MobieNetV2_Q=torchvision.models.quantization.mobilenet_v2(weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1,quantize=True)
    # MobileNetV3_Large_Q=torchvision.models.quantization.mobilenet_v3_large(weights=MobileNet_V3_Large_QuantizedWeights.IMAGENET1K_QNNPACK_V1,quantize=True)
# ['MobieNetV2_Q',MobieNetV2_Q,'cpu'],['MobileNetV3_Large_Q',MobileNetV3_Large_Q,'cpu']
    weights_list=[MobileNetV2,MobileNetV3_Large,MobileNetV3_Small]
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
