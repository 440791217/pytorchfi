import torch
import torchvision
from torchvision import transforms





# 指定 ImageNet 数据集的根目录
def loader(root):
    # 定义数据预处理操作
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载训练集
    train_dataset = torchvision.datasets.ImageNet(root=root, split='train', transform=transform)

    # 加载验证集
    val_dataset = torchvision.datasets.ImageNet(root=root, split='val', transform=transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False,num_workers=4)
    return  train_loader,val_loader

if __name__=='__main__':
    train,val=loader()
    # 遍历训练集
    for images, labels in train:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break
