import torch
import torch.nn as nn
import torch.optim as optim
import mysys
from loaders import ImageNetLoader
import os
from tqdm import tqdm
import datetime

class MobileNetV1(nn.Module):
    def __init__(self, class_num=1000):
        super(MobileNetV1, self).__init__()

        class ConvBnReluBlock(nn.Module):
            def __init__(self, in_c, out_c, stride=1):
                super().__init__()
                self.conv = nn.Conv2d(in_c, out_c, 3, stride, padding=1, bias=False)
                self.bn = nn.BatchNorm2d(out_c)
                self.relu = nn.ReLU6(inplace=True)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        class DepthwiseSeparableConvBlock(nn.Module):
            def __init__(self, in_c, out_c, stride=1):
                super().__init__()
                self.depthwise_conv = nn.Conv2d(in_c, in_c, 3, stride, padding=1, groups=in_c, bias=False)
                self.bn1 = nn.BatchNorm2d(in_c)
                self.relu1 = nn.ReLU6(inplace=True)
                self.pointwise_conv = nn.Conv2d(in_c, out_c, 1, stride=1, padding=0, bias=False)
                self.bn2 = nn.BatchNorm2d(out_c)
                self.relu2 = nn.ReLU6(inplace=True)

            def forward(self, x):
                x = self.depthwise_conv(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.pointwise_conv(x)
                x = self.bn2(x)
                x = self.relu2(x)
                return x

        self.model = nn.Sequential(
            ConvBnReluBlock(3, 32, 2),
            DepthwiseSeparableConvBlock(32, 64, 1),
            DepthwiseSeparableConvBlock(64, 128, 2),
            DepthwiseSeparableConvBlock(128, 128, 1),
            DepthwiseSeparableConvBlock(128, 256, 2),
            DepthwiseSeparableConvBlock(256, 256, 1),
            DepthwiseSeparableConvBlock(256, 512, 2),
            DepthwiseSeparableConvBlock(512, 512, 1),
            DepthwiseSeparableConvBlock(512, 512, 1),
            DepthwiseSeparableConvBlock(512, 512, 1),
            DepthwiseSeparableConvBlock(512, 512, 1),
            DepthwiseSeparableConvBlock(512, 512, 1),
            DepthwiseSeparableConvBlock(512, 1024, 2),
            DepthwiseSeparableConvBlock(1024, 1024, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, class_num, bias=True)
        )

    def forward(self, x):
        return self.model(x)
    
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

def predict():
    # 实例化模型
    model = MobileNetV1(class_num=10)
    # 随机生成输入数据
    input_data = torch.randn(1, 3, 224, 224)
    # 前向传播
    output = model(input_data)
    print(output.shape)


def train():
    start_time = datetime.datetime.now()
    print(f"Training started at: {start_time}")
    root = mysys.ImageNetRoot
    train_loader, val_loader = ImageNetLoader.loader(root=root)
    # 初始化模型
    model = MobileNetV1()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 检查是否有之前保存的检查点
    checkpoint_path = os.path.join(mysys.modelsPath, 'MobileNetV1', 'checkpoint.pth')
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch + 1}")

    # 训练模型
    num_epochs = 300
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        # 使用 tqdm 为训练循环添加进度条
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # 更新进度条
                pbar.set_postfix({'Loss': loss.item()})
                pbar.update(1)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}')

        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        checkpoint_dir = os.path.join(mysys.modelsPath, 'MobileNetV1')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pth'))
        print(f"Checkpoint saved at epoch {epoch + 1}")

    end_time = datetime.datetime.now()
    print(f"Training ended at: {end_time}")
    print(f"Total training time: {end_time - start_time}")
    # 保存最终模型参数
    model_path = os.path.join(mysys.modelsPath, 'MobileNetV1')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_path = os.path.join(model_path, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 加载模型参数示例
    loaded_model = MobileNetV1()
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.to(device)
    loaded_model.eval()

    # 简单测试加载后的模型
    test_image = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        test_output = loaded_model(test_image)
    print(f"Output shape of the loaded model: {test_output.shape}")

def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(mysys.modelsPath, 'MobileNetV1/model.pth')
    # 加载模型参数示例
    loaded_model = MobileNetV1()
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.to(device)
    loaded_model.eval()

    root = mysys.ImageNetRoot
    train_loader, val_loader = ImageNetLoader.loader(root=root)
    # 评估加载后的模型准确率
    loaded_accuracy = evaluate(loaded_model, val_loader, device)
    print("Accuracy of the loaded {} on the ImageNet validation set: {}%".format('MobileNetV1', loaded_accuracy))


if __name__ == '__main__':
    train()
    # eval()
    