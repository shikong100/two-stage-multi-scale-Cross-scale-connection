import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
from data_loader import BinaryClassificationDataset
from torch.utils.data import DataLoader



if __name__ == '__main__':
    # model = torch.load("./weight/best.pth")
    model = torch.load("./pre-model/best_val_f1.pth")

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    # class_correct = [0.] * 10
    class_correct = [0.] 
    # class_total = [0.] * 10
    class_total = [0.] 
    y_test, y_pred, y_imgPath = [], [], []
    X_test = []

    BATCH_SIZE = 256
    # BATCH_SIZE = 16
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # data_transform = transforms.Compose([transforms.Resize(256),
    #                                    transforms.CenterCrop(224),
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # val_dataset = datasets.ImageFolder("./data/valid/", transform=data_transform)  # 测试集数据
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    #                                              num_workers=2)  # 加载数据
    train_data = BinaryClassificationDataset(annRoot="/mnt/data0/qh/Sewer/annotations", imgRoot="/mnt/data0/qh/Sewer", transform=transform, split="Train")
    train_data.labels = train_data.labels.flatten()
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    valid_data = BinaryClassificationDataset(annRoot="/mnt/data0/qh/Sewer/annotations", imgRoot="/mnt/data0/qh/Sewer", split="Valid", transform=transform)
    valid_data.labels = valid_data.labels.flatten()
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # train_data = BinaryClassificationDataset(annRoot="/mnt/H/qh_data/Sewer/annotations", imgRoot="/mnt/H/qh_data/Sewer", transform=transform, split="")
    # train_data.labels = train_data.labels.flatten()
    # train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("begin run.....")
    with torch.no_grad():
        # for images, labels, imagePaths in valid_loader:
        for images, labels, imagePaths in train_loader:
            # X_test.extend([_ for _ in images])
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            # c = (predicted == labels).squeeze()
            # for i, label in enumerate(labels):
            #     class_correct[label] += c[i].item()
            #     class_total[label] += 1
            # y_imgPath.extend(imagePaths)
            imagePaths = list(imagePaths)
            predicted = list(predicted.numpy())
            # y_pred.extend(predicted.numpy())
            df = pd.DataFrame({
                "Filename": imagePaths,
                "ND": predicted
            })
            if os.path.exists('pred_t.csv'):
                df = pd.DataFrame({
                    "Filename": imagePaths,
                    "ND": predicted
                })
                df.to_csv('pred_t.csv', mode='a', columns=['Filename', 'ND'], header=False, index=False)
            else:
                df = pd.DataFrame({
                    "Filename": imagePaths,
                    "ND": predicted
                })
                df.to_csv('pred_t.csv', mode='w', index=False)

    print('finish run!')
            # y_test.extend(labels.cpu().numpy())

    # for i in range(2):
    # print(f"Acuracy of {classes[0]:5s}: {100 * class_correct[0] / class_total[0]:2.0f}%")
    # print(y_imgPath)
    # df = pd.DataFrame({
    #     "Filename": y_imgPath,
    #     "ND": y_pred
    # })
    
    # csv_file = 'pred_t_csv.csv'
    # csv_file = 'pred_v_csv.csv'
    # df.to_csv(csv_file, index=False)

    # ac = accuracy_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # cr = classification_report(y_test, y_pred, target_names=classes)
    # print("Accuracy is :", ac)
    # print(cr)


    # labels = pd.DataFrame(cm).applymap(lambda v: f"{v}" if v != 0 else f"")
    # plt.figure(figsize=(7, 5))
    # sns.heatmap(cm, annot=labels, fmt='s', xticklabels=classes, yticklabels=classes, linewidths=0.1)
    # plt.show()
