import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import sklearn.metrics as sm


def train_and_val(epochs, model, train_loader, len_train,val_loader, len_val,criterion, optimizer,device):

    run = wandb.init(
        project = "ResNet34-Sewer-BinaryClassification",
        entity = "qhproject",
        name = "resnet34+external1+pre-128-1",
        config = {
            "BATCH_SIZE": 128,
            "epoch": 30,
            "lr": 0.001,
            "image_size": 224,
            "net": "resnet34"
        }
    )

    # torch.cuda.empty_cache()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    val_F1normal = []
    best_acc = 0
    best_val_F1 = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        training_acc = 0
        pred_t = [] # 预测标签结果
        label_t = [] # 真实训练标签
        pred_v = [] # 预测验证标签结果
        label_v = [] # 预测真实标签
        with tqdm(total=len(train_loader)) as pbar:
            for image, label, imgName in train_loader:

                model.train()
                optimizer.zero_grad()
                image = image.to(device)
                label = label.to(device)
                l_t = list(label.cpu().numpy())
                label_t += l_t

                # forward
                output = model(image)
                
                loss = criterion(output, label)
                predict_t = torch.max(output, dim=1)[1]

                p_t = list(predict_t.cpu().numpy())
                pred_t += p_t

                # backward
                loss.backward()
                optimizer.step()  # update weight

                running_loss += loss.item()
                training_acc += torch.eq(predict_t, label).sum().item()

                pbar.update(1)

        model.eval()
        val_losses = 0
        validation_acc = 0
        # validation loop
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pb:
                for image, label, imgName in val_loader:
                    image = image.to(device)
                    label = label.to(device)

                    l_v = list(label.cpu().numpy())
                    label_v += l_v

                    output = model(image)

                    # loss
                    loss = criterion(output, label)
                    predict_v = torch.max(output, dim=1)[1]

                    p_v = list(predict_v.cpu().numpy())
                    pred_v += p_v

                    val_losses += loss.item()
                    validation_acc += torch.eq(predict_v, label).sum().item()
                    pb.update(1)

            # calculatio mean for each batch
            train_loss.append(running_loss / len_train)
            print(f'runing_loss/ len_train->{running_loss/len_train}')
        
            val_loss.append(val_losses / len_val)

            train_acc.append(training_acc / len_train)
            val_acc.append(validation_acc / len_val)

            v_f1 = sm.f1_score(label_v, pred_v, average='macro')
            val_F1normal.append(v_f1)

            torch.save(model, "./weight128-1/last.pth")
            if best_acc < (validation_acc / len_val):
                best_acc_path = "./weight128-1/" + "best_acc_" + str((validation_acc / len_val)) + ".pth"
                best_acc = validation_acc / len_val
                torch.save(model, best_acc_path)
            if best_val_F1 < v_f1:
                best_val_f1_path = "./weight128-1/" + "best_val_f1_" + str(v_f1) + ".pth"
                best_val_F1 = v_f1
                torch.save(model, best_val_f1_path)

            wandb.log({
                "Train Acc": (training_acc / len_train),
                "Val Acc": (validation_acc / len_val),
                "Train Loss": (running_loss / len_train),
                "Val Loss": (val_losses / len_val),
                "Val_F1": v_f1,
                "epoch": e
            })
            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Acc: {:.3f}..".format(training_acc / len_train),
                  "Val Acc: {:.3f}..".format(validation_acc / len_val),
                  "Train Loss: {:.3f}..".format(running_loss / len_train),
                  "Val Loss: {:.3f}..".format(val_losses / len_val),
                  "Val_F1: {:.3f}..".format(v_f1),
                  "Time: {:.2f}s".format((time.time() - since)))
    histor = {'train_loss': train_loss, 'val_loss': val_loss ,'train_acc': train_acc, 'val_acc': val_acc, 'val_F1normal': val_F1normal}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    run.finish()

    return histor

def plot_loss(x, history):
    plt.plot(x, history['val_loss'], label='val', marker='o')
    plt.plot(x, history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('./weight128-1/loss.png')
    plt.show()


def plot_acc(x, history):
    plt.plot(x, history['train_acc'], label='train_acc', marker='x')
    plt.plot(x, history['val_acc'], label='val_acc', marker='x')
    plt.title('Acc per epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('./weight128-1/acc.png')
    plt.show()

def plot_f1(x, history):
    plt.plot(x, history['val_F1normal'], label='val_F1normal', marker='x')
    plt.title('F1normal')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(),plt.grid()
    plt.savefig('./weight128-1/F1normal.png')
    plt.show()
