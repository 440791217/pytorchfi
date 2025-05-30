import torch
import torchvision
import mysys
import os
rootpath = os.path.join(mysys.datasetsPath,'CIFAR10')

data=torchvision.datasets.CIFAR10(
    root=rootpath,
    train=True,
    download=True,
)


data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4)