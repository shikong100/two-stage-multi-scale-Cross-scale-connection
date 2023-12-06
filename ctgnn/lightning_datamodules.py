import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataloader import MultiLabelDataset, WaterLevelDataset, MultiTaskDataset



class MultiTaskDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, workers=4, ann_root="/mnt/data0/qh/Sewer/annotations", data_root="/mnt/data0/qh/Sewer", train_transform = None, eval_transform = None, only_defects=False):
    # def __init__(self, batch_size=32, workers=4, ann_root="/mnt/data0/qh/Sewer/annotations", data_root="/mnt/data0/qh/Sewer/Train", train_transform = None, eval_transform = None, only_defects=True):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root
        self.only_defects = only_defects

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        # split dataset
        if stage == 'fit':
            self.train_dataset = MultiTaskDataset(self.ann_root, self.data_root, split="Train", transform=self.train_transform, onlyDefects=self.only_defects)
            self.val_dataset = MultiTaskDataset(self.ann_root, self.data_root, split="Valid", transform=self.eval_transform, onlyDefects=self.only_defects)
        if stage == 'test':
            self.test_dataset = MultiTaskDataset(self.ann_root, self.data_root, split="Test", transform=self.eval_transform, onlyDefects=self.only_defects)

        self.defect_num_classes = self.train_dataset.defect_num_classes
        self.water_num_classes = self.train_dataset.water_num_classes

        self.defect_LabelNames = self.train_dataset.defect_LabelNames
        self.water_LabelNames = self.train_dataset.water_LabelNames

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.workers, pin_memory=True) # pin_memory = True, 将数据复制到cuda中
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return test_dl


class MultiLabelDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, workers=4, ann_root="./annotations", data_root="../devdisk/Sewer/Train", train_transform = None, eval_transform = None, only_defects=False):
    # def __init__(self, batch_size=32, workers=4, ann_root="./annotations", data_root="../devdisk/Sewer/Train", train_transform = None, eval_transform = None, only_defects=True):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root
        self.only_defects = only_defects

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        # split dataset
        if stage == 'fit':
            self.train_dataset = MultiLabelDataset(self.ann_root, self.data_root, split="Train", transform=self.train_transform, onlyDefects=self.only_defects)
            self.val_dataset = MultiLabelDataset(self.ann_root, self.data_root, split="Valid", transform=self.eval_transform, onlyDefects=self.only_defects)
        if stage == 'test':
            self.test_dataset = MultiLabelDataset(self.ann_root, self.data_root, split="Test", transform=self.eval_transform, onlyDefects=self.only_defects)

        self.num_classes = self.train_dataset.num_classes
        self.LabelNames = self.train_dataset.LabelNames

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.workers, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return test_dl


class WaterLevelDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, workers=4, ann_root="./annotations", data_root="../devdisk/Sewer/Train", train_transform = None, eval_transform = None):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        # split dataset
        if stage == 'fit':
            self.train_dataset = WaterLevelDataset(self.ann_root, self.data_root, split="Train", transform=self.train_transform)
            self.val_dataset = WaterLevelDataset(self.ann_root, self.data_root, split="Valid", transform=self.eval_transform)
        if stage == 'test':
            self.test_dataset = WaterLevelDataset(self.ann_root, self.data_root, split="Test", transform=self.eval_transform)

        self.num_classes = self.train_dataset.num_classes
        self.LabelNames = self.train_dataset.LabelNames

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.workers, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return test_dl
