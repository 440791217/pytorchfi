import torch
import torchvision


def loadDataSet(rootpath = 'D:\GitHub\pytorchfi\dataset\CIFAR100'):

    data=torchvision.datasets.CIFAR10(
        root=rootpath,
        train=True,
        download=True,
    )


    data_loader = torch.utils.data.DataLoader(data,
                                            batch_size=4,
                                            shuffle=True,
                                            num_workers=4)
    return data_loader, data.classes, data

if __name__=='__main__':
    loadDataSet()