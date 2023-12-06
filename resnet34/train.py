import os
import torch
import torch.optim as optim
from torchvision import transforms, datasets, models
import torch.nn as nn
from model import resnet50, resnet34, resnet18
from utils import train_and_val,plot_acc,plot_loss, plot_f1
import numpy as np
from torch.utils.data import DataLoader
from data_loader import BinaryClassificationDataset
import os



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("using {} device.".format(device))

    if not os.path.exists('./weight1'):
        os.makedirs('./weight1')

    BATCH_SIZE = 128
    # BATCH_SIZE = 7
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #     "val": transforms.Compose([transforms.Resize(256),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # train_dataset = datasets.ImageFolder("./data/train/", transform=data_transform["train"])  # 训练集数据
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                            #    num_workers=4)  # 加载数据
    # len_train = len(train_dataset)
    # val_dataset = datasets.ImageFolder("./data/valid/", transform=data_transform["val"])  # 测试集数据
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    #                                          num_workers=4)  # 加载数据
    train_data = BinaryClassificationDataset(annRoot="/mnt/data0/qh/Sewer/annotations", imgRoot="/mnt/data0/qh/Sewer", transform=transform, split="Train")
    train_data.labels = train_data.labels.flatten()
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    # train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # train_loader.labels.flatten()
    # print(train_loader.labels)
    len_train = len(train_data)

    valid_data = BinaryClassificationDataset(annRoot="/mnt/data0/qh/Sewer/annotations", imgRoot="/mnt/data0/qh/Sewer", split="Valid", transform=transform)
    valid_data.labels = valid_data.labels.flatten()
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)
    # valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    len_val = len(valid_data)

    # len_val = len(val_dataset)

    # net = resnet50()
    net = resnet34()
    # 载入预训练模型
    model_weight_path = "./pre-model/resnet34-pre.pth"
    predict_model = torch.load(model_weight_path)
    net_weights = net.state_dict()
    del_key = []
    for key, _ in predict_model.items():
        if "fc" in key:
            del_key.append(key)
    for key in del_key:
        del predict_model[key]
    net.load_state_dict(predict_model, strict=False)

    loss_function = nn.CrossEntropyLoss()  # 设置损失函数
    # optimizer = optim.Adam(net.parameters(), lr=0.001)  # 设置优化器和学习率
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)  # 设置优化器和学习率
    
    epoch = 30

    history = train_and_val(epoch, net, train_loader, len_train, valid_loader, len_val, loss_function, optimizer, device)

    plot_loss(np.arange(0,epoch), history)
    plot_acc(np.arange(0,epoch), history)
    plot_f1(np.arange(0, epoch), history)
