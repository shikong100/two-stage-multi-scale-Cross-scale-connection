import os
import wandb
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn

from torchvision import models as torch_models
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_datamodules import MultiTaskDataModule

from class_weight import positive_ratio, inverse_frequency, effective_samples, identity_weight, defect_CIW

import models.encoders as encoder_models
import models.decoders as decoder_models
from models.cont_defect import Cont
from models.supconloss import SupConLoss
from models.res2net import res2net50_26w_4s
from models.dblloss import ResampleLoss


class MTL_Model(pl.LightningModule):
    TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__ if
                                     name.islower() and not name.startswith("__") and callable(
                                         torch_models.__dict__[name]))
    MODEL_NAMES = TORCHVISION_MODEL_NAMES

    ENCODER_NAMES = sorted(name for name in encoder_models.__dict__ if
                           not name.startswith("__") and callable(encoder_models.__dict__[name]))
    DECODER_NAMES = sorted(name for name in decoder_models.__dict__ if
                           not name.startswith("__") and callable(decoder_models.__dict__[name]))

    def __init__(self, args, backbone="resnet18", encoder="ResNetBackbone", decoder="SimpleHead", task_classes=[],
                 task_class_type=[], task_class_weights=[], valid_tasks=["defect", "water"],
                 task_criterion_weighting="Equal", learning_rate=1e-2, momentum=0.9, weight_decay=0.0001, max_epochs=60,
                 **kwargs):
        super(MTL_Model, self).__init__()
        self.save_hyperparameters()

        self.valid_tasks = valid_tasks # '0':defect '1':water
        self.task_classes = task_classes # 17,4
        self.task_class_weights = task_class_weights
        self.task_class_type = task_class_type # 'ML', 'MC'
        self.task_criterion_weighting = task_criterion_weighting # 'Fixed'
        self.num_tasks = len(self.task_classes) # 2

        # if backbone in MTL_Model.TORCHVISION_MODEL_NAMES: # resnet50
        #     self.backbone = torch_models.__dict__[backbone]
        # else:
        #     raise ValueError("Got backbone {}, but no such backbone is in this codebase".format(backbone))

        # if encoder in MTL_Model.ENCODER_NAMES: # ResNetBackbone
        #     self.encoder = encoder_models.__dict__[encoder](backbone=self.backbone, n_tasks=self.num_tasks)
        # else:
        #     raise ValueError("Got encoder {}, but no such encoder is in this codebase".format(encoder))
        self.encoder = res2net50_26w_4s(pretrained=False)
        pre_path = './models/res2net50_26w_4s.pth'
        pred_model = torch.load(pre_path)
        net_weights = self.encoder.state_dict()
        del_key = []
        for key, _ in pred_model.items():
            if "fc" in key:
                del_key.append(key)
        for key in del_key:
            del pred_model[key]
        self.encoder.load_state_dict(pred_model, strict=False)

        if backbone == "resnet18" or backbone == "resnet34": # 硬编码共享参数
            input_channels = 512
        else:
            input_channels = 2048 # resnet50及以后的网络输出channel为2048

        if encoder == "MTAN": # 软编码共享
            list_input = True
        else:
            list_input = False

        if decoder in MTL_Model.DECODER_NAMES:
            if decoder == "SimpleHead":
                self.decoder = decoder_models.__dict__[decoder](n_tasks=self.num_tasks, num_classes=task_classes,
                                                                input_channels=input_channels,
                                                                decoder_channels=kwargs["decoder_channels"],
                                                                pool=kwargs["decoder_pool"], list_input=list_input)
            else:
                self.decoder = decoder_models.__dict__[decoder](n_tasks=self.num_tasks,                     num_classes=task_classes,
                                                                input_channels=input_channels,
                                                                decoder_channels=kwargs["decoder_channels"],
                                                                adj_mat_path=kwargs["adj_mat_path"],
                                                                gnn_head=kwargs["gnn_head"],
                                                                gnn_layers=kwargs["gnn_layers"],
                                                                gnn_channels=kwargs["gnn_channels"],
                                                                gnn_dropout=kwargs["gnn_dropout"],
                                                                gnn_residual=kwargs["gnn_residual"],
                                                                attention_heads=kwargs["gat_num_heads"],
                                                                bottleneck_channels=kwargs["bottleneck_channels"],
                                                                adj_normalization=kwargs["adj_normalization"],
                                                                pool=kwargs["decoder_pool"], list_input=list_input,
                                                                shared_bottleneck=kwargs["shared_bottleneck"],
                                                                shared_linear=kwargs["shared_linear"],
                                                                gnn_residual_act=kwargs["gnn_residual_act"])
        else:
            raise ValueError("Got decoder {}, but no such decoder is in this codebase".format(decoder))
        # 对比损失
        self.contrative_defect = Cont(feature_dim=2176, label_dim=17, z_dim=17)
        self.contrative_water = SupConLoss()
        #平衡损失
        # self.dbl_loss = ResampleLoss(use_sigmoid=True, 
        #                              reweight_func='rebalance', 
        #                              focal=dict(focal=True, balance_param=2.0, gamma=2),
        #                              logit_reg=dict(neg_scale=2.0, init_bias=0.05),
        #                              map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
        #                              loss_weight=1.0, freq_file=freq_file)

        if task_criterion_weighting == "Equal":
            self.register_buffer("task_weights", torch.ones([self.num_tasks, max_epochs]))
            self.weighted_loss_func = self.weighted_loss

        if task_criterion_weighting == "Fixed":
            weights = [x / sum(kwargs["task_weights_fixed"]) * self.num_tasks for x in kwargs["task_weights_fixed"]]
            weights_mat = np.ones((self.num_tasks, max_epochs))
            for idx in range(self.num_tasks):
                weights_mat[idx, :] *= weights[idx]
            self.register_buffer("task_weights", torch.tensor(weights_mat))
            self.weighted_loss_func = self.weighted_loss

        if task_criterion_weighting == "DWA":
            if len(kwargs["task_weights_fixed"]) > 0:
                weights = [x / sum(kwargs["task_weights_fixed"]) * self.num_tasks for x in kwargs["task_weights_fixed"]]
                weights_mat = np.ones((self.num_tasks, max_epochs))
                for idx in range(self.num_tasks):
                    weights_mat[idx, :] *= weights[idx]
                self.register_buffer("task_weights", torch.tensor(weights_mat))
            else:
                self.register_buffer("task_weights", torch.ones([self.num_tasks, max_epochs]))

            self.register_buffer("loss_buffer", torch.zeros([self.num_tasks, max_epochs]))
            self.dwa_temp = kwargs["dwa_temp"]
            self.weighted_loss_func = self.weighted_loss

        if task_criterion_weighting == "Uncertainty":
            self.logvars = nn.Parameter(torch.zeros([self.num_tasks]))
            self.weighted_loss_func = self.uncertainty_loss

        # Create task criterions with calculated weights. The weight argument acts as a per-class weight, unlike the pos_weight argument.
        # This can be used as a per class weight, by providing a VECTOR, which will be broadcasted to the batch size

        task_idx = 0
        if "defect" in valid_tasks:
            print(self.task_class_weights[task_idx])
            self.defect_criterion = torch.nn.BCEWithLogitsLoss(weight=self.task_class_weights[task_idx][0],
                                                               pos_weight=self.task_class_weights[task_idx][1])
            task_idx += 1
        if "water" in valid_tasks:
            self.water_criterion = torch.nn.CrossEntropyLoss(weight=self.task_class_weights[task_idx])
            task_idx += 1

        self.task_LUT = {"defect":0,
                         "water": 1}

        if kwargs["use_auxilliary"]:
            self.sum_loss = self.aux_loss
            self.main_weight = kwargs["main_weight"]
        else:
            self.sum_loss = self.main_loss
            self.main_weight = 1.0

    
    def aux_loss(self, losses, losses_pre, conloss):
        return self.main_weight * losses + (1 - self.main_weight) * losses_pre + conloss

    def main_loss(self, losses, losses_pre):
        return losses

    def weighted_loss(self, losses):
        return losses * self.task_weights[:, self.current_epoch]

    def uncertainty_loss(self, losses):
        return losses * torch.exp(
            -self.logvars) + 0.5 * self.logvars  # The regularization term can become negative (i.e. std dev < 1). Possible fix is log(1+ sigma) instad of log(sigma) - https://arxiv.org/abs/1805.06334

    def supconloss_defect(self, input_label, label_emb, feat_emb, embs, temp=1.0, sample_wise=False):
        # if sample_wise:
            # loss_func = SupConLoss(temperature=0.1)
            # return loss_func(torch.stack([label_emb, feat_emb], dim=1), input_label.float())

        features = torch.cat((label_emb, feat_emb))
        labels = torch.cat((input_label, input_label)).float()
        n_label = labels.shape[1]
        emb_labels = torch.eye(n_label).to(feat_emb.device)
        mask = torch.matmul(labels, emb_labels)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, embs),
            temp)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)
        mean_log_prob_neg = ((1.0-mask) * log_prob).sum(1) / ((1.0-mask).sum(1) + 1e-7)

        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss
    
    def supconloss_water(self, features, label):
        loss = self.contrative_water(features, label)
        return loss

    def forward(self, x):
        feat_vec, lay_feat = self.encoder(x)
        logits, logits_pre, feat = self.decoder(feat_vec, lay_feat)
        return logits, logits_pre, feat

    def train_function(self, x, y):
        x = torch.cat([x[0], x[1]], dim=0)

        y_hat, y_hat_pre, feat = self(x)
        bsz = y[0].shape[0]
        _, y_hat_d = torch.split(y_hat[0], [bsz, bsz], dim=0)
        _, y_hat_w = torch.split(y_hat[1], [bsz, bsz], dim=0)
        y_hat = [y_hat_d, y_hat_w]
        _, y_hat_pre_d = torch.split(y_hat_pre[0], [bsz, bsz], dim=0)
        _, y_hat_pre_w = torch.split(y_hat_pre[1], [bsz, bsz], dim=0)
        y_hat_pre = [y_hat_pre_d, y_hat_pre_w]

        # contrastive learning
        
        f1, f2 = torch.split(feat[0], [bsz, bsz], dim=0)
        contrive_defect = self.contrative_defect(y[0], f2)
        features_water = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # [B, 2, 128]
        
        losses = []
        losses_pre = []
        for idx, loss_name in enumerate(self.valid_tasks):
            losses.append(getattr(self, "{}_criterion".format(loss_name))(y_hat[idx], y[self.task_LUT[loss_name]]))
            losses_pre.append(
                getattr(self, "{}_criterion".format(loss_name))(y_hat_pre[idx], y[self.task_LUT[loss_name]]))
        losses = torch.stack(losses)
        losses_pre = torch.stack(losses_pre)

        weighted_losses = self.weighted_loss_func(losses)
        weighted_losses_pre = self.weighted_loss_func(losses_pre)

        # contrastive loss
        conloss = self.supconloss_defect(y[0], contrive_defect['label_emb'], contrive_defect['feat_emb'], contrive_defect['embs'])
        conloss_water = self.supconloss_water(features_water, y[1])

        conloss = [conloss, conloss_water]
        conloss = torch.stack(conloss)

        return losses, weighted_losses, weighted_losses_pre, conloss

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        losses, weighted_losses, weighted_losses_pre, conloss = self.train_function(x, y)

        total_loss = torch.sum(self.sum_loss(weighted_losses, weighted_losses_pre, conloss))
        wandb.log({'total_train_loss': total_loss})
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True,
                 prog_bar=True)  # Log individual losses and loss weights
        for idx, loss_name in enumerate(self.valid_tasks):
            wandb.log({'train_{}_loss'.format(loss_name): losses[idx],
                        'train_{}_wloss'.format(loss_name): weighted_losses[idx],
                        'train_{}_conloss'.format(loss_name): conloss[idx],})
            self.log('train_{}_loss'.format(loss_name), losses[idx], on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=True)  # Log individual losses and loss weights
            self.log('train_{}_wloss'.format(loss_name), weighted_losses[idx], on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=False)  # Log individual losses and loss weights
            self.log('train_{}_wloss_aux'.format(loss_name), weighted_losses_pre[idx], on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=False)  # Log individual losses and loss weights
            self.log('train_{}_conloss'.format(loss_name), conloss[idx], on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        self.weighted_losses = weighted_losses
        self.losses = losses

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        losses, weighted_losses, weighted_losses_pre, conloss = self.train_function(x, y)

        total_loss = torch.sum(self.sum_loss(weighted_losses, weighted_losses_pre, conloss))
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True,
                 prog_bar=False)  # Log individual losses and loss weights
        wandb.log({'total_val_loss': total_loss})
        for idx, loss_name in enumerate(self.valid_tasks):
            wandb.log({'val_{}_loss'.format(loss_name): losses[idx],
                        'val_{}_wloss'.format(loss_name): weighted_losses[idx],
                        'val_{}_conloss'.format(loss_name): conloss[idx],})
            self.log('val_{}_loss'.format(loss_name), losses[idx], on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=True)  # Log individual losses and loss weights
            self.log('val_{}_wloss'.format(loss_name), weighted_losses[idx], on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=False)  # Log individual losses and loss weights
            self.log('val_{}_wloss_aux'.format(loss_name), weighted_losses_pre[idx], on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=False)  # Log individual losses and loss weights
            self.log('val_{}_conloss'.format(loss_name), conloss[idx], on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        return total_loss

    def on_epoch_start(self):
        if self.task_criterion_weighting == "DWA":
            if self.current_epoch > 1:
                gamma = self.loss_buffer[:, self.current_epoch - 1] / self.loss_buffer[:, self.current_epoch - 2]
                exp_ind = torch.exp(gamma / self.dwa_temp)
                exp_sum = torch.sum(exp_ind)
                for idx in range(self.num_tasks):
                    self.task_weights[idx, self.current_epoch] = self.num_tasks * exp_ind[idx] / exp_sum

    def on_epoch_end(self):
        if self.task_criterion_weighting == "DWA":
            for idx, task in enumerate(self.valid_tasks):
                self.loss_buffer[idx, self.current_epoch] = self.trainer.logged_metrics["train_{}_loss".format(task)]

    def configure_optimizers(self):

        if self.task_criterion_weighting == "Uncertainty":
            grouped_parameters = [
                {"params": self.encoder.parameters(), 'lr': self.hparams.learning_rate},
                {"params": self.decoder.parameters(), 'lr': self.hparams.learning_rate},
                {"params": self.logvars, 'lr': self.hparams.weights_learning_rate},
            ]
        else:
            grouped_parameters = self.parameters()

        optim = torch.optim.SGD(grouped_parameters, lr=self.hparams.learning_rate, momentum=self.hparams.momentum,
                                weight_decay=self.hparams.weight_decay)

        if self.hparams.lr_schedule == "Step":
            if self.hparams.schedule_int == "epoch":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=self.hparams.learning_rate_steps,
                                                                 gamma=self.hparams.learning_rate_gamma)
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                                 milestones=[x * len(self.train_dataloader()) for x in
                                                                             self.hparams.learning_rate_steps],
                                                                 gamma=self.hparams.learning_rate_gamma)
        elif self.hparams.lr_schedule == "Cosine":
            if self.hparams.schedule_int == "epoch":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams.max_epochs,
                                                                       eta_min=self.hparams.learning_rate * 0.01)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams.max_epochs * len(
                    self.train_dataloader()), eta_min=self.hparams.learning_rate * 0.01)

        scheduler = {"scheduler": scheduler,
                     "interval": self.hparams.schedule_int,
                     "frequency": 1}

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=128, help="Size of the batch per GPU")
        parser.add_argument('--learning_rate', type=float, default=0.1)
        parser.add_argument('--learning_rate_gamma', type=float, default=0.01) # 调整学习率倍数
        # parser.add_argument('--learning_rate_steps', nargs='+', type=int, default=[20, 30]) # 在第几个step调整学习率
        parser.add_argument('--learning_rate_steps', nargs='+', type=int, default=[20]) # 在第几个step调整学习率
        parser.add_argument('--lr_schedule', type=str, default="Step", choices=["Step", "Cosine"]) # 学习率调整策略
        parser.add_argument('--schedule_int', type=str, default="epoch", choices=["epoch", "step"])
        parser.add_argument('--weights_learning_rate', type=float, default=0.025)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.0001)

        parser.add_argument('--backbone', type=str, default="resnet50", choices=MTL_Model.MODEL_NAMES)
        parser.add_argument('--encoder', type=str, default="ResNetBackbone", choices=MTL_Model.ENCODER_NAMES)
        parser.add_argument('--decoder', type=str, default="CTGNN", choices=MTL_Model.DECODER_NAMES)
        parser.add_argument('--decoder_channels', nargs='+', type=int, default=[])
        parser.add_argument('--decoder_pool', type=str, default="Avg", choices=["Avg", "Max", ""])
        parser.add_argument('--bottleneck_channels', type=int)
        parser.add_argument('--shared_bottleneck', action='store_true')
        parser.add_argument('--shared_linear', action='store_true')

        parser.add_argument('--gnn_head', type=str, default="", choices=["", "GCN", "GAT"])
        parser.add_argument('--gnn_layers', type=int, default=4)
        parser.add_argument('--gnn_channels', nargs='+', type=int, default=128)
        parser.add_argument('--gnn_dropout', type=float, default=0.0)
        parser.add_argument('--gnn_residual', action='store_true')
        parser.add_argument('--gnn_residual_act', help="When to apply activation, if any", type=str, default="Pre",
                            choices=["None", "Pre", "Post", "Both"])
        parser.add_argument('--gat_num_heads', type=int, default=8)

        parser.add_argument('--adj_mat_path', type=str, default="")
        parser.add_argument('--adj_normalization', type=str, default="Sym", choices=["", "Sym", "In"])

        parser.add_argument('--use_auxilliary', action='store_true')
        parser.add_argument('--main_weight', type=float, default=0.75)

        parser.add_argument('--valid_tasks', nargs='+', type=str, default=["defect", "water"])
        parser.add_argument('--class_weight', type=str, default="Effective",
                            choices=["None", "Inverse", "Positive", "Effective", "CIW"])
        parser.add_argument('--defect_weights', type=str, default="PosCIW", choices=["", "CIW", "PosCIW", "Both"])
        parser.add_argument('--effective_beta', type=float, default=0.9999)
        parser.add_argument('--task_weight', type=str, default="Equal",
                            choices=["Equal", "DWA", "Uncertainty", "Fixed"])
        parser.add_argument('--task_weights_fixed', nargs='+', type=float)
        parser.add_argument('--dwa_temp', type=float, default=2.0)
        parser.add_argument('--f2CIW_weights', type=str, default="PosWeight")
        parser.add_argument('--only_defects', type=bool, default=False, choices=[True, False])

        return parser

def main(args):
    args.seed = pl.seed_everything(args.seed)

    # Init data with transforms
    img_size = 224

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    dm = MultiTaskDataModule(batch_size=args.batch_size, workers=args.workers, ann_root=args.ann_root,
                             data_root=args.data_root, train_transform=train_transform, eval_transform=eval_transform, only_defects=args.only_defects)

    dm.prepare_data() # 只在1个GPU上训练，没用
    dm.setup("fit") # 训练阶段

    # Get the class weights per task
    if args.class_weight == "None": # 权重一致，都为1.0
        defect_weights = identity_weight(dm.train_dataset.defect_labels, dm.defect_num_classes)
        water_weights = identity_weight(dm.train_dataset.water_labels, dm.water_num_classes)

    elif args.class_weight == "Positive":
        defect_weights = positive_ratio(dm.train_dataset.defect_labels, dm.defect_num_classes)
        water_weights = positive_ratio(dm.train_dataset.water_labels, dm.water_num_classes)

    elif args.class_weight == "Inverse":
        defect_weights = inverse_frequency(dm.train_dataset.defect_labels, dm.defect_num_classes)
        water_weights = inverse_frequency(dm.train_dataset.water_labels, dm.water_num_classes)

    elif args.class_weight == "Effective":
        assert args.effective_beta < 1.0 and args.effective_beta >= 0.0, "The effective sampling beta need to be in the range [0,1) and not: {}".format(
            args.effective_beta)

        # 要看其他论文Class-Balanced Loss Based on Effective Number of Samples
        defect_weights = effective_samples(dm.train_dataset.defect_labels, dm.defect_num_classes, args.effective_beta)
        water_weights = effective_samples(dm.train_dataset.water_labels, dm.water_num_classes, args.effective_beta)

    defect_weights = [defect_weights, None]
    if args.defect_weights != "": # 对类重要性权重进行平均，使用PosCIW
        ciw_weights = defect_CIW(dm.train_dataset.defect_LabelNames)

        if args.defect_weights == "Both":
            defect_weights[1] = ciw_weights
        elif args.defect_weights == "CIW":
            defect_weights[0] = ciw_weights
            defect_weights[1] = None
        elif args.defect_weights == "PosCIW":
            defect_weights[0] = None
            defect_weights[1] = ciw_weights

    task_LUT = {"defect": 0,
                "water": 1}

    valid_tasks = args.valid_tasks # defect, water

    task_class_weights = [defect_weights, water_weights]
    # task_class_type = ["ML", "MC", "MC", "MC"]
    task_class_type = ["ML", "MC"] # ML 是多任务, MC 是多分类
    task_classes = [dm.defect_num_classes, dm.water_num_classes] # （17,4）

    # Only keep the values for the tasks used
    task_class_weights = [task_class_weights[task_LUT[task]] for task in valid_tasks]
    task_class_type = [task_class_type[task_LUT[task]] for task in valid_tasks] # ML，MC
    task_classes = [task_classes[task_LUT[task]] for task in valid_tasks] # 17,4

    if args.task_weight == "Fixed" or (args.task_weight == "DWA" and len(args.task_weights_fixed) > 0):
        assert len(args.task_weights_fixed) == len(
            args.valid_tasks), "The amount of supplied weights ({}) are not equal to the amount of valid tasks ({})".format(
            len(args.task_weights_fixed), len(args.valid_tasks))

    # Init our model
    light_model = MTL_Model(args, task_classes=task_classes, task_class_weights=task_class_weights,
                            task_class_type=task_class_type, task_criterion_weighting=args.task_weight, **vars(args))

    # train
    if len(args.decoder_channels):
        decoder_channels = "_".join([str(x) for x in args.decoder_channels])
    else:
        decoder_channels = ""
    model_name = args.backbone + "_" + args.encoder + "_" + args.decoder + decoder_channels + "_" + args.gnn_head

    prefix = "{}-{}-MTL-".format(args.task_weight, args.class_weight)
    print("-" * 15 + prefix + "-" * 15)

    if not os.path.isdir(args.log_save_dir):
        os.makedirs(args.log_save_dir)

    logger = TensorBoardLogger(save_dir=args.log_save_dir, name=model_name,
                               version=prefix + "version_" + str(args.log_version))

    logger_path = os.path.join(args.log_save_dir, model_name, prefix + "version_" + str(args.log_version))

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger_path),
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        save_last=True,
        verbose=False,
        monitor="val_loss",
        mode='min',
        prefix='',
        period=1
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    if args.use_deterministic:
        deterministic = True
        benchmark = False
    else:
        deterministic = False
        benchmark = True

    trainer = pl.Trainer.from_argparse_args(args, terminate_on_nan=True, benchmark=benchmark,
                                            deterministic=deterministic, max_epochs=args.max_epochs, logger=logger,
                                            callbacks=[checkpoint_callback, lr_monitor], gpus=[1, 2, 3])

    try:
        trainer.fit(light_model, dm)
        wandb.finish()
    except Exception as e:
        print(e)
        with open(os.path.join(logger_path, "error.txt"), "w") as f:
            f.write(str(e))


def run_cli():
    # add PROGRAM level args
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='qh')
    parser.add_argument('--ann_root', type=str, default='/mnt/data0/qh/Sewer/annotations')
    parser.add_argument('--data_root', type=str, default='/mnt/data0/qh/Sewer')
    parser.add_argument('--workers', type=int, default=4)
    # parser.add_argument('--log_save_dir', type=str, default="./logs")
    parser.add_argument('--log_save_dir', type=str, default="./log")
    parser.add_argument('--log_version', type=int, default=1)
    parser.add_argument('--use_deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=1234567890)
    # parser.add_argument("--gpus", type=list, default=[0, 1])

    # add TRAINER level args 训练级别参数
    parser = pl.Trainer.add_argparse_args(parser)

    # add MODEL level args 模型级别参数
    parser = MTL_Model.add_model_specific_args(parser)
    args = parser.parse_args()

    # args.workers = args.gpus * 6
    args.workers = 12
    # args.workers = 2

    if args.decoder == "SimpleHead":
        assert (args.gnn_head == "")
    else:
        assert (args.gnn_head != "")

    
    wandb.init(
        project="two-defect",
        config=args,
        name="TC-GCN-2.2-multi-scale"
    )
    main(args)


if __name__ == "__main__":
    run_cli()
