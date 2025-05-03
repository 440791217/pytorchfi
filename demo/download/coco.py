from torch.utils.data import DataLoader

dataset = CustomDataset(root_dir="path/to/images", annFile="path/to/annotations.json")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)