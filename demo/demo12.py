#神经网络例程
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
# 加载数据集
train_dataset = torchvision.datasets.ImageFolder(root='path_to_imagenet/train', transform=transform)
val_dataset = torchvision.datasets.ImageFolder(root='path_to_imagenet/val', transform=transform)
 
# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

model = mobilenet_v2(pretrained=True)  # 使用预训练的MobileNet V2模型
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1000)  # 如果需要调整最后一层输出到1000个类别（ImageNet有1000个类别）

criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # SGD优化器