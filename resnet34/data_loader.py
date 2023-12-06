import os 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

Labels = ["ND"]

class BinaryClassificationDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader):
        super(BinaryClassificationDataset, self).__init__()
        self.annRoot = annRoot
        self.imgRoot = imgRoot
        self.split = split
        self.transform = transform
        self.loader = loader
        self.LabelNames = Labels.copy()

        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        # gtPath = os.path.join(self.annRoot, "T_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols=self.LabelNames + ["Filename"])
        self.imgPaths = gt["Filename"].values
        self.labels = gt[self.LabelNames].values
    
    def __len__(self):
        return (len(self.imgPaths))

    def __getitem__(self, index):
        path = self.imgPaths[index]
        img = self.loader(os.path.join(self.imgRoot, self.split, path))
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index, ]
        return img, int(label), path

if __name__ == "__main__":
    from torch.utils.data import dataloader
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_data = BinaryClassificationDataset(annRoot="/data/qh/Sewer/annotations", imgRoot="/data/qh/Sewer", split="Train", transform=transform)
    print(train_data.labels)

