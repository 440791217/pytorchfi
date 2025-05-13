import torch
import torchvision
from torchvision import transforms





# 指定 ImageNet 数据集的根目录
def loader(root,num_workers=8,batch_size=32,resize=None,crop=None,mean=None,std=None):
    # 定义数据预处理操作
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 加载训练集
    train_dataset = torchvision.datasets.ImageNet(root=root, split='train', transform=transform)

    # 加载验证集
    val_dataset = torchvision.datasets.ImageNet(root=root, split='val', transform=transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    return  train_loader,val_loader

if __name__=='__main__':
    train,val=loader()
    # 遍历训练集
    for images, labels in train:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break
