import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader


DefectLabels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK", "VA", "ND"]
WaterLabels = ["0%<5%","5-15%","15-30%","30%<="]
                
class MultiTaskDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, waterIntervals = [5, 15, 30], onlyDefects=False, inferce=False):
    # def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, waterIntervals = [5, 15, 30], onlyDefects=True, inferce=False):
        super(MultiTaskDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.inferce = inferce
        self.transform = transform
        self.loader = loader
        self.onlyDefects = onlyDefects
        self.waterIntervals = waterIntervals

        
        self.defect_LabelNames = DefectLabels.copy()
        self.defect_LabelNames.remove("VA")
        self.defect_LabelNames.remove("ND")

        self.water_LabelNames = WaterLabels.copy()

        self.defect_num_classes = len(self.defect_LabelNames)
        self.water_num_classes = len(self.water_LabelNames)

        self.loadAnnotations()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        # gtPath = os.path.join(self.annRoot, "T_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = self.defect_LabelNames + ["Filename", "Defect", "WaterLevel"])
        
        if self.inferce:
            ptPath = './binary_results/pred_v.csv'
            pt = pd.read_csv(ptPath, sep=',', encoding='utf-8', usecols=['Filename', 'ND'])
            filename = pt[pt['ND'] == 0]['Filename']
            # filename = pt['Filename']
            gt = gt.loc[pt['Filename'].isin(filename)]
            # gt = gt[gt['Filename' == pt['Filename'] & pt['ND'] == 0]]

        if self.split == 'Train':
            gt = gt[gt["Defect"] == 1]

        self.imgPaths = gt["Filename"].values

        # Get defect labels
        self.defect_labels = gt[self.defect_LabelNames].values
        
        # Get water labels
        self.water_labels = gt["WaterLevel"].values

        if len(self.waterIntervals) > 0:
            self.water_num_classes = len(self.waterIntervals)+1
            self.water_labels[self.water_labels < self.waterIntervals[0]] = 0
            self.water_labels[self.water_labels >= self.waterIntervals[-1]] = self.water_num_classes-1
            for idx in range(1, len(self.waterIntervals)):
                self.water_labels[(self.water_labels >= self.waterIntervals[idx-1]) & (self.water_labels < self.waterIntervals[idx])] = idx
        else:
            uniqueLevels = np.unique(self.labels)
            self.water_num_classes = len(uniqueLevels)
            for idx, level in enumerate(uniqueLevels):
                self.water_labels[self.water_labels == level] = idx

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, self.split, path))
        if self.transform is not None:
            img_transform = self.transform(img)
            img = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])(img)

        defect_target = torch.tensor(self.defect_labels[index, :], dtype=torch.float)
        water_target = torch.tensor(self.water_labels[index], dtype=torch.long)

        return [img, img_transform], [defect_target, water_target], path


class MultiLabelDataset(Dataset):
    # def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, onlyDefects=False):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, onlyDefects=True):
        super(MultiLabelDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = loader

        self.LabelNames = DefectLabels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)

        self.loadAnnotations()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        # gtPath = os.path.join(self.annRoot, "T_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = self.LabelNames + ["Filename", "Defect"])

        if self.onlyDefects:
            gt = gt[gt["Defect"] == 1]

        self.imgPaths = gt["Filename"].values
        self.labels = gt[self.LabelNames].values
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, self.split, path))
        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(self.labels[index, :], dtype=torch.float)

        return img, target, path


class WaterLevelDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, waterIntervals = [5, 15, 30]):
        super(WaterLevelDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split
        self.waterIntervals = waterIntervals

        self.transform = transform
        self.loader = loader

        self.LabelNames = WaterLabels.copy()
        self.num_classes = len(self.LabelNames)

        self.loadAnnotations()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["Filename", "WaterLevel"])

        self.imgPaths = gt["Filename"].values
        self.labels = gt["WaterLevel"].values

        if len(self.waterIntervals) > 0:
            self.num_classes = len(self.waterIntervals)+1
            self.labels[self.labels < self.waterIntervals[0]] = 0
            self.labels[self.labels >= self.waterIntervals[-1]] = self.num_classes-1
            for idx in range(1, len(self.waterIntervals)):
                self.labels[(self.labels >= self.waterIntervals[idx-1]) & (self.labels < self.waterIntervals[idx])] = idx
        else:
            uniqueLevels = np.unique(self.labels)
            self.num_classes = len(uniqueLevels)
            for idx, level in enumerate(uniqueLevels):
                self.labels[self.labels == level] = idx
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, self.split, path))
        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(self.labels[index], dtype=torch.long)

        return img, target, path
