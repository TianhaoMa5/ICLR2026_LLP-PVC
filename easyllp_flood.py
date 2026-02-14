
from __future__ import print_function
import random

import time
import argparse
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, OrderedDict

from sklearn.metrics import f1_score
import copy

from WideResNet import WideResnet
from datasets.cifar_dir import get_train_loader, get_val_loader
from utils import  accuracy,setup_default_logging, AverageMeter, CurrentValueMeter, WarmupCosineLrScheduler
import tensorboard_logger
import torch.multiprocessing as mp

from LeNet import LeNet5,MLPDropIn

import torch
import torch.nn as nn

from torchvision import models
import os, csv


def set_model(args):
    if args.dataset in ['CIFAR10', 'CIFAR100','miniImageNet','SVHN']:

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.n_classes)
        model = WideResnet(
            n_classes=args.n_classes,
            k=args.wresnet_k,
            n=args.wresnet_n,
            proj=False
        )
       # model = WideResnet(n_classes=args.n_classes, k=args.wresnet_k, n=args.wresnet_n, proj=True)
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.n_classes)
        #model = resnet34(num_classes=10)
    else:
        model = LeNet5()
    #model = ResNet18CIFAR10(num_classes=args.n_classes)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)

        msg = model.load_state_dict(checkpoint, strict=False)


        assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
        print('loaded from checkpoint: %s' % args.checkpoint)
    model.train()
    model.cuda()

    if args.eval_ema:
        ema_model = copy.deepcopy(model).cuda().eval()
        for p in ema_model.parameters():
            p.requires_grad = False
    else:
        ema_model = None

    criteria_x = nn.CrossEntropyLoss().cuda()
    criteria_u = nn.CrossEntropyLoss(reduction='none').cuda()

    return model, criteria_x, criteria_u, ema_model



@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1 - ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)


def llp_loss(labels_proportion, y):
    x = torch.tensor(labels_proportion, dtype=torch.float64).cuda()

    # Ensure y is also double

    y = y.double()
    cross_entropy = torch.sum(-x * (torch.log(y) + 1e-7))

    return cross_entropy


def custom_loss(chunk, label_proportions, k=1.0, p=0.5):
    alpha = label_proportions[0][1]


    coeff_1 = k * (alpha - p) + p
    coeff_0 = k * (p - alpha) + (1 - p)


    target_0 = torch.zeros(chunk.size(0), 1).cuda()  # [batch_size, 2]
    target_1 = torch.ones(chunk.size(0), 1).cuda()  # [batch_size, 2]
    target_1=target_1.squeeze(-1)
    target_0=target_0.squeeze(-1)
    chunk=chunk.squeeze(-1)

    g_chunk_1 = nn.BCEWithLogitsLoss()(chunk, target_1.float())+1e-9
    g_chunk_0 = nn.BCEWithLogitsLoss()(chunk, target_0.float())+1e-9


    loss = max(0,coeff_1 * g_chunk_1) + max(0,coeff_0 * g_chunk_0)

    return loss.mean()
def double_hinge_loss(logits_u_w, lbs_u_real):

    logits_u_w = logits_u_w.squeeze(-1)
    lbs_u_real = lbs_u_real.squeeze(-1)
    m = logits_u_w * (2 * lbs_u_real - 1)

    #
    loss = torch.max(torch.zeros_like(m), 1 - m, m - 2)

    return loss.mean()


def thre_ema(thre, sum_values, ema):
    return thre * ema + (1 - ema) * sum_values


def weight_decay_with_mask(mask, initial_weight, max_mask_count):
    mask_count = mask.sum().item()
    weight_decay = max(0, 1 - mask_count / max_mask_count)
    return initial_weight * weight_decay


def train_one_epoch(epoch,
                    bagsize,
                    n_classes,
                    model,
                    ema_model,
                    prob_list,
                    criteria_x,
                    criteria_u,
                    optim,
                    lr_schdlr,
                    dltrain_u,
                    args,
                    n_iters,
                    logger,
                    samp_ran
                    ):
    model.train()
    loss_u_meter = AverageMeter()
    loss_easy_meter = AverageMeter()
    thre_meter = AverageMeter()
    kl_meter = AverageMeter()
    kl_hard_meter = AverageMeter()
    loss_contrast_meter = AverageMeter()
    # the number of correct pseudo-labels
    n_correct_u_lbs_meter = AverageMeter()
    # the number of confident unlabeled data
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()
    # the number of edges in the pseudo-label graph
    pos_meter = AverageMeter()
    samp_lb_meter, samp_p_meter = [], []
    for i in range(0, bagsize):
        x = CurrentValueMeter()
        y = CurrentValueMeter()
        samp_lb_meter.append(x)
        samp_p_meter.append(y)
    epoch_start = time.time()  # start time
    dl_u = iter(dltrain_u)
    n_iter = len(dltrain_u)

    for it in range(len(dltrain_u)):
        (var1, var2, var3, var4, var5) = next(dl_u)
        var1 = var1[0]
        # var2 = torch.stack(var2)
        # print(var2)
        # print(f'var1:{var1.shape};\n var2: {var2.shape};\n var3: {var3.shape};\n var4: {var4.shape}')
        length = len(var2[0])

        """
        pseudo_counter = Counter(selected_label.tolist())
        for i in range(args.n_classes):
            classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

        """
        ims_u_weak1 = var1

        imsw, labels_real, labels_idx = [], [], []  # $$$$$$$$$$$$$

        for i in range(length):
            imsw.append(ims_u_weak1[i])
            labels_real.append(var3[i])
            labels_idx.append(var4[i])
        ims_u_weak = torch.cat(imsw, dim=0)
        lbs_u_real = torch.cat(labels_real, dim=0)
        label_proportions = [[] for _ in range(length)]
        lbs_u_real = lbs_u_real.cuda()
        lbs_idx = torch.cat(labels_idx, dim=0)
        lbs_idx = lbs_idx.cuda()
        #ims_u_weak = ims_u_weak.permute(0, 2, 1, 3)


        positions = torch.nonzero(lbs_idx == 37821).squeeze()

        if positions.numel() != 0:
            head = positions - positions % bagsize
            rear = head + bagsize - 1
        for i in range(length):
            labels = []
            for j in range(n_classes):
                labels.append(var2[j][i])
            label_proportions[i].append(labels)
        proportion =[]
        for i in range(length):
            proportion.append(torch.tensor(label_proportions[i][0], dtype=torch.float64).cuda())
        proportion=torch.stack(proportion)
        prior=proportion.mean(dim=0)
        bag_weight = proportion*bagsize-(bagsize-1)*prior
        instance_weight = bag_weight.repeat_interleave(bagsize,dim=0)
        btu = ims_u_weak.size(0)
        bt = 0

        if args.dataset in ["MNIST", "FashionMNIST", "KMNIST"]:
            ims_u_weak = ims_u_weak.permute(0, 2, 1, 3)
        imgs = torch.cat([ims_u_weak], dim=0).cuda()
        logits = model(imgs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]  # 取真正的分类输出那个 tensor

        # logits_x = logits[:bt]
        logits_u_w = torch.split(logits[0:], btu)
        logits_u_w = logits_u_w[0]

        # feats_x = features[:bt]

        # loss_x = criteria_x(logits_x, lbs_x)

        chunk_size = len(logits_u_w) // length

        chunks = [logits_u_w[i * chunk_size:(i + 1) * chunk_size] for i in range(length)]


        loss_prop = torch.Tensor([]).cuda()
        loss_prop = loss_prop.double()

        loss_easy = torch.Tensor([]).cuda()
        loss_easy = loss_easy.double()


        loss_easy = loss_easy.mean()
        with torch.no_grad():
            # feats_x = feats_x.detach()

            probs = torch.softmax(logits_u_w, dim=1)


            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(args.thr).float()
            if positions.numel() != 0:
                for i in range(0, bagsize):
                    samp_lb_meter[i].update(lbs_u_guess[head + i].item())
                    samp_p_meter[i].update(scores[head + i].item())

        logits_u_w = logits_u_w.squeeze(-1)
        loss_easy = nn.CrossEntropyLoss()(logits_u_w,instance_weight).mean()
        loss=loss_easy
        b=0.3
        loss = (loss - b).abs() + b
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()

        if args.eval_ema:
            with torch.no_grad():
                ema_model_update(model, ema_model, args.ema_m)
        loss_easy_meter.update(loss_easy.item())
        loss_u_meter.update(loss_easy.item())
        mask_meter.update(mask.mean().item())
        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())

        if (it + 1) % n_iter == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)
            logger.info("{}-x{}-s{}, {} | epoch:{}, iter: {}.  loss_easy: {:.3f}."
                        "LR: {:.3f}. Time: {:.2f}".format(
                args.dataset, args.n_labeled, args.seed, args.exp_dir, epoch, it + 1, loss_easy_meter.avg,  lr_log, t))

            epoch_start = time.time()
            bagsize = getattr(args, "bagsize", getattr(args, "bag_size", None))
            if bagsize is None:
                raise ValueError("请在 args 中提供 bagsize 或 bag_size")

            exp_dir_name = os.path.basename(os.path.normpath(args.exp_dir))

            os.makedirs(args.exp_dir, exist_ok=True)
            csv_name = f"{args.dataset}_{exp_dir_name}_{bagsize}.csv"
            csv_path = os.path.join(args.exp_dir, csv_name)

            is_new = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(["epoch", "time_sec"])
                writer.writerow([epoch, round(t, 2)])

            logger.info(f"CSV path -> {os.path.abspath(csv_path)}")
    return loss_u_meter.avg,loss_easy_meter.avg, n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, mask_meter.avg, kl_meter.avg, kl_hard_meter.avg



def evaluate(model, ema_model, dataloader,dataset):
    model.eval()

    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    ema_top5_meter = AverageMeter()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []
    ema_all_preds = []
    ema_all_labels = []
    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            if dataset in ["MNIST", "FashionMNIST", "KMNIST"]:
                ims = ims.permute(0, 2, 1, 3)

            out = model(ims)
            if isinstance(out, (tuple, list)):
                logits = out[0]  # 取第一个作为 logits
            else:
                logits = out
            loss = torch.nn.CrossEntropyLoss()(logits, lbs)

            loss_meter.update(loss.item())

            scores = torch.softmax(logits, dim=1)
            preds = scores.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbs.cpu().numpy())

            top1, top5 = accuracy(scores, lbs, (1, 5))
            top1_meter.update(top1.item())
            top5_meter.update(top5.item())

            if ema_model is not None:
                ema_logits = ema_model(ims)
                ema_scores = torch.softmax(ema_logits, dim=1)
                ema_preds = ema_scores.argmax(dim=1)
                ema_all_preds.extend(ema_preds.cpu().numpy())
                ema_all_labels.extend(lbs.cpu().numpy())
                ema_top1, ema_top5 = accuracy(ema_scores, lbs, (1, 5))
                ema_top1_meter.update(ema_top1.item())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    ema_macro_f1 = f1_score(ema_all_labels, ema_all_preds, average='macro') if ema_model is not None else None

    return top1_meter.avg, ema_top1_meter.avg, top5_meter.avg, ema_top5_meter.avg, loss_meter.avg, macro_f1, ema_macro_f1

def main():
    parser = argparse.ArgumentParser(description='L^2P-AHIL')

    parser.add_argument('--root', default='./data', type=str, help='dataset directory')
    parser.add_argument('--wresnet-k', default=4, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=16, type=int,
                        help='depth of wide resnet')
    parser.add_argument('--dataset', type=str, default="SVHN",
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=100,
                        help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=10,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=500,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=1,
                        help='train batch size of bag samples')
    parser.add_argument('--bagsize', type=int, default=1024,
                        help='train bag size of samples')
    parser.add_argument('--n-imgs-per-epoch', type=int, default=50,
                        help='number of training images for each epoch')

    parser.add_argument('--eval-ema', default=False, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)

    parser.add_argument('--lam-u', type=float, default=1.,
                        help='c oefficient of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed for random behaviors, no seed if negtive')
    parser.add_argument(
        '--pl_method',
        type=str,
        default='cluster',
        choices=['random', 'cluster', 'alphafirst'],
        help='pseudo label selection method: random | cluster | alphafirst'
    )
    parser.add_argument('--temperature', default=0.2, type=float, help='softmax temperature')
    parser.add_argument('--low-dim', type=int, default=64)
    parser.add_argument('--lam-c', type=float, default=1,
                        help='coefficient of contrastive loss')
    parser.add_argument('--lam-p', type=float, default=2,
                        help='coefficient of proportion loss')
    parser.add_argument('--contrast-th', default=0.8, type=float,
                        help='pseudo label graph threshold')
    parser.add_argument('--thr', type=float, default=0.95,
                        help='pseudo label threshold')
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument(
        '--pi', default='10', type=str,
        help='Bag purity / concentration control for alphafirst and clusterbag. '
             'Smaller pi => more homogeneous bags (instances tend to come from the same label-proportion target or the same cluster); '
             'larger pi => more mixed bags.'
    )

    parser.add_argument('--queue_batch', type=float, default=5,
                        help='number of batches stored in memory bank')
    parser.add_argument('--exp-dir', default='easy_flood', type=str, help='experiment id')
    parser.add_argument('--checkpoint', default='', type=str, help='use pretrained model')
    parser.add_argument('--folds', default='2', type=str, help='number of dataset')
    args = parser.parse_args()

    logger, output_dir = setup_default_logging(args)
    logger.info(dict(args._get_kwargs()))

    tb_logger = tensorboard_logger.Logger(logdir=output_dir, flush_secs=2)
    samp_ran = 37821
    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    n_iters_per_epoch = args.n_imgs_per_epoch  # 1024

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.n_labeled}")

    model, criteria_x, criteria_u, ema_model = set_model(args)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    if args.pl_method == 'random':
        from datasets.cifar import get_train_loader
    elif args.pl_method == 'alphafirst':
        from datasets.cifar_dir import get_train_loader
    elif args.pl_method == 'cluster':
        from datasets.cifar_cluster2 import get_train_loader
    else:
        raise ValueError(f"Unknown pl_method: {args.pl_method}")
    dltrain_u, dataset_length, input_dim = get_train_loader(args,args.n_classes,
                                                            args.dataset, args.batchsize, args.bagsize, root=args.root,
                                                            method='DLLP',
                                                            supervised=False)
    dlval = get_val_loader(dataset=args.dataset, batch_size=64, num_workers=2, root=args.root,n_classes=args.n_classes)
    n_iters_all = len(dltrain_u) * args.n_epoches
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]





    optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum, nesterov=True)
    lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)

    # memory bank
    args.queue_size = 5120
    queue_feats = torch.zeros(args.queue_size, args.low_dim).cuda()
    queue_probs = torch.zeros(args.queue_size, args.n_classes).cuda()
    queue_ptr = 0

    # for distribution alignment
    prob_list = []

    train_args = dict(
        model=model,
        ema_model=ema_model,
        prob_list=prob_list,
        criteria_x=criteria_x,
        criteria_u=criteria_u,
        optim=optim,
        lr_schdlr=lr_schdlr,
        dltrain_u=dltrain_u,
        args=args,
        n_iters=n_iters_per_epoch,
        logger=logger
    )

    best_acc = -1
    best_acc_5 = -1
    best_epoch_5 = 0

    best_epoch = 0

    logger.info('-----------start training--------------')
    for epoch in range(args.n_epoches):
        loss_u,loss_prob, n_correct_u_lbs, n_strong_aug, mask_mean, num_pos, samp_lb = \
            train_one_epoch(epoch, bagsize=args.bagsize, n_classes=args.n_classes, **train_args, samp_ran=samp_ran,
                            )

        top1, ema_top1, top5, ema_top5, loss_test,macro,ema_macro = evaluate(model, ema_model, dlval,args.dataset)
        tb_logger.log_value('loss_u', loss_u, epoch)

        tb_logger.log_value('loss_prob', loss_prob, epoch)
        if (n_strong_aug == 0):
            tb_logger.log_value('guess_label_acc', 0, epoch)
        else:
            tb_logger.log_value('guess_label_acc', n_correct_u_lbs / n_strong_aug, epoch)
        tb_logger.log_value('test_acc', top1, epoch)
        tb_logger.log_value('mask', mask_mean, epoch)
        tb_logger.log_value('loss_test', loss_test, epoch)
        tb_logger.log_value('macro', macro, epoch)
        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch
        logger.info(
            "Epoch {}. Acc: {:.4f}. Ema-Acc: {:.4f}.Macro: {:.4f}. loss_test: {:.4f}. best_acc: {:.4f} in epoch{}".
            format(epoch, top1, ema_top1,macro,loss_test, best_acc, best_epoch ))

        if epoch % 5234512353454 == 13453:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'lr_scheduler': lr_schdlr.state_dict(),
                'prob_list': prob_list,
                'queue': {'queue_feats': queue_feats, 'queue_probs': queue_probs, 'queue_ptr': queue_ptr},
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_%02d.pth' % epoch))


if __name__ == '__main__':
    main()