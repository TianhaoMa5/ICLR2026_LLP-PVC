from __future__ import print_function
import random

import time
import argparse
import os
import sys

import numpy as np

import torch
import torch .nn as nn
import torch .nn .functional as F
from collections import Counter ,OrderedDict
from WideResNet import WideResnet
from datasets .cifar_cluster2 import get_train_loader ,get_val_loader
from utils import accuracy ,setup_default_logging ,AverageMeter ,CurrentValueMeter ,WarmupCosineLrScheduler
import tensorboard_logger
import torch .multiprocessing as mp
from LeNet import LeNet5 ,MLPDropIn
from torchvision import models
import os ,csv

import torch
import math





def cross_entropy_loss_torch (softmax_matrix ,onehot_labels ):

    log_softmax =torch .log (softmax_matrix +1e-12 )

    cross_entropy =-torch .sum (onehot_labels *log_softmax ,dim =1 )

    mean_loss =torch .mean (cross_entropy )
    return mean_loss


def normal(h, h_tilde, beta):
    return torch.exp(-(h - h_tilde)**2 / beta)

def calc_bag_entropy(probs):
    probs_class_normal = torch.nn.functional.normalize(probs, p=1, dim=1)
    return -torch.sum(probs_class_normal * torch.log
                      (probs_class_normal + 1e-8), dim=1)

def calc_instance_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

def calc_opt_entropy(nn): # nn表示每个类的数量
    return torch.log(nn + 1e-8)
def set_model (args ):
    if args .dataset in ['CIFAR10','SVHN','CIFAR100','miniImageNet']:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.n_classes)
        model =WideResnet (
        n_classes =args .n_classes ,
        k =args .wresnet_k ,
        n =args .wresnet_n ,
        proj =False
        )
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.n_classes)

        #model = resnet34(num_classes=10)
    else :
        model =LeNet5 ()
        #model = ResNet18CIFAR10(num_classes=args.n_classes)

    if args .checkpoint :
        checkpoint =torch .load (args .checkpoint )

        msg =model .load_state_dict (checkpoint ,strict =False )


        assert set (msg .missing_keys )=={"classifier.weight","classifier.bias"}
        print ('loaded from checkpoint: %s'%args .checkpoint )
    model .train ()
    model .cuda ()

    if args .eval_ema :
        if args .dataset in ['CIFAR10','SVHN','CIFAR100','miniImageNet']:
            ema_model =WideResnet (
            n_classes =args .n_classes ,
            k =args .wresnet_k ,
            n =args .wresnet_n ,
            proj =False
            )
            model =models .resnet18 (pretrained =True )
            model .fc =nn .Linear (model .fc .in_features ,args.n_classes )
            #ema_model = resnet34(num_classes=10)

        else :
            model =MLPDropIn ()
            #ema_model = ResNet18CIFAR10(num_classes=args.n_classes)

        for param_q ,param_k in zip (model .parameters (),ema_model .parameters ()):
            param_k .data .copy_ (param_q .detach ().data )# initialize
            param_k .requires_grad =False # not update by gradient for eval_net
        ema_model .cuda ()
        ema_model .eval ()
    else :
        ema_model =None

    criteria_x =nn .CrossEntropyLoss ().cuda ()
    criteria_u =nn .CrossEntropyLoss (reduction ='none').cuda ()

    return model ,criteria_x ,criteria_u ,ema_model



@torch .no_grad ()
def ema_model_update (model ,ema_model ,ema_m ):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    for param_train ,param_eval in zip (model .parameters (),ema_model .parameters ()):
        param_eval .copy_ (param_eval *ema_m +param_train .detach ()*(1 -ema_m ))

    for buffer_train ,buffer_eval in zip (model .buffers (),ema_model .buffers ()):
        buffer_eval .copy_ (buffer_train )


def llp_loss (labels_proportion ,y ):
    x =torch .tensor (labels_proportion ,dtype =torch .float64 ).cuda ()
    x =x .squeeze (0 )#

    # Ensure y is also double

    y =y .double ()
    cross_entropy =torch .sum (-x *(torch .log (y )+1e-7 ))
    mse_loss =torch .mean ((x -y )**2 )

    return cross_entropy


def custom_loss (probs ,lambda_val =1.0 ):
# probs is assumed to be a 2D tensor of shape (n, N_i)
# where n is the number of rows and N_i is the number of columns

# Compute the log of probs
    log_probs =torch .log (probs )

    # Multiply probs with log_probs element-wise
    product =-probs *log_probs

    # Compute the double sum
    loss =torch .sum (product )

    # Multiply by lambda
    loss =lambda_val *loss

    return loss

@torch .no_grad ()
def distributed_sinkhorn (out ,args ,proportion ):
    Q =out
    K =Q .shape [0 ]
    B =Q .shape [1 ]

    for it in range (args .sinkhorn_iterations ):
        sum_of_rows =torch .sum (Q ,dim =1 ,keepdim =True )
        sum_of_rows [sum_of_rows ==0 ]=1
        Q /=sum_of_rows

        sum_of_cols =torch .sum (Q ,dim =0 ,keepdim =True )
        sum_of_cols [sum_of_cols ==0 ]=1
        Q /=sum_of_cols
        Q *=proportion

    sum_of_cols =torch .sum (Q ,dim =0 ,keepdim =True )
    sum_of_cols [sum_of_cols ==0 ]=1
    Q /=sum_of_cols
    Q *=proportion

    return Q
def thre_ema (thre ,sum_values ,ema ):
    return thre *ema +(1 -ema )*sum_values


def weight_decay_with_mask (mask ,initial_weight ,max_mask_count ):
    mask_count =mask .sum ().item ()
    weight_decay =max (0 ,1 -mask_count /max_mask_count )
    return initial_weight *weight_decay


def train_one_epoch (epoch ,
bagsize ,
n_classes ,
model ,
ema_model ,
prob_list ,
criteria_x ,
criteria_u ,
optim ,
lr_schdlr ,
dltrain_u ,
args ,
n_iters ,
logger ,
samp_ran
):
    model .train ()
    loss_u_meter =AverageMeter ()
    loss_prop_meter =AverageMeter ()
    thre_meter =AverageMeter ()
    kl_meter =AverageMeter ()
    kl_hard_meter =AverageMeter ()
    loss_contrast_meter =AverageMeter ()
    # the number of correct pseudo-labels
    n_correct_u_lbs_meter =AverageMeter ()
    # the number of confident unlabeled data
    n_strong_aug_meter =AverageMeter ()
    mask_meter =AverageMeter ()
    entropy_meter =AverageMeter ()

    # the number of edges in the pseudo-label graph
    pos_meter =AverageMeter ()
    samp_lb_meter ,samp_p_meter =[],[]
    for i in range (0 ,bagsize ):
        x =CurrentValueMeter ()
        y =CurrentValueMeter ()
        samp_lb_meter .append (x )
        samp_p_meter .append (y )
    epoch_start =time .time ()# start time
    dl_u = iter(dltrain_u)
    n_iter = len(dltrain_u)

    num_samples = len(dltrain_u.dataset)
    for it in range(len(dltrain_u)):
        (var1, var2, var3, var4, var5) = next(dl_u)
        var1 =var1 [0 ]
        # var2 = torch.stack(var2)
        # print(var2)
        # print(f'var1:{var1.shape};\n var2: {var2.shape};\n var3: {var3.shape};\n var4: {var4.shape}')
        length =len (var2 [0 ])

        """
        pseudo_counter = Counter(selected_label.tolist())
        for i in range(args.n_classes):
            classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

        """
        ims_u_weak1 =var1

        imsw ,labels_real ,labels_idx =[],[],[]# $$$$$$$$$$$$$

        for i in range (length ):
            imsw .append (ims_u_weak1 [i ])
            labels_real .append (var3 [i ])
            labels_idx .append (var4 [i ])
        ims_u_weak =torch .cat (imsw ,dim =0 )
        lbs_u_real =torch .cat (labels_real ,dim =0 )
        label_proportions =[[]for _ in range (length )]
        lbs_u_real =lbs_u_real .cuda ()
        lbs_idx =torch .cat (labels_idx ,dim =0 )
        lbs_idx =lbs_idx .cuda ()

        positions =torch .nonzero (lbs_idx ==37821 ).squeeze ()

        if positions .numel ()!=0 :
            head =positions -positions %bagsize
            rear =head +bagsize -1

        for i in range (length ):
            labels =[]
            for j in range (n_classes ):
                labels .append (var2 [j ][i ])
            label_proportions [i ].append (labels )

            # --------------------------------------
        btu =ims_u_weak .size (0 )
        if args .dataset in ["MNIST","FashionMNIST","KMNIST"]:
            ims_u_weak =ims_u_weak .permute (0 ,2 ,1 ,3 )

        bt =0
        imgs =torch .cat ([ims_u_weak ],dim =0 ).cuda ()
        logits =model (imgs )

        # logits_x = logits[:bt]
        logits_u_w =torch .split (logits [0 :],btu )
        logits_u_w =logits_u_w [0 ]
        probs =torch .softmax (logits_u_w ,dim =-1 )
        # logits_x = logits[:bt]
        logits_u_w = torch.split(logits[0:], btu)
        logits_u_w = logits_u_w[0]

        # loss_x = criteria_x(logits_x, lbs_x)

        B = length
        device = logits_u_w.device

        chunk_size = len(logits_u_w) // length

        # 分成 length 节
        chunks = [logits_u_w[i * chunk_size:(i + 1) * chunk_size] for i in range(length)]

        proportion =torch .empty ((0 ,n_classes ),dtype =torch .float64 ).cuda ()
        batch_size =length

        for i in range (length ):
            pr =label_proportions [i ][0 ]
            pr =torch .stack (pr ).cuda ()
            proportion =torch .cat ((proportion ,pr .unsqueeze (0 )))
        proportion =proportion .view (length ,n_classes ,1 )
        proportion =proportion .squeeze (-1 )
        proportion =proportion .double ()
        loss_prop =torch .Tensor ([]).cuda ()
        loss_prop =loss_prop .double ()
        kl_divergence =torch .Tensor ([]).cuda ()
        kl_divergence =kl_divergence .double ()
        kl_divergence_hard =torch .Tensor ([]).cuda ()
        kl_divergence_hard =kl_divergence_hard .double ()
        for i ,chunk in enumerate (chunks ):
            labels_p =torch .softmax (chunk ,dim =1 )
            scores ,lbs_u_guess =torch .max (labels_p ,dim =1 )
            #opt_onehot = solve_optimal_onehot_with_proportions_torch(labels_p, proportion[i], bagsize, n_classes).float()
            #opt_onehot=opt_onehot.cuda()
            labels_p =torch .mean (labels_p ,dim =0 )

            loss_p =llp_loss (proportion [i ],labels_p )

            #loss_p = compute_CC_loss_logDP(labels_p,proportion[i])
            #loss_p= compute_CC_loss_simplified_gpu(labels_p,proportion[i])

            label_prop =torch .tensor (label_proportions [i ],dtype =torch .float64 ).cuda ()
            loss_prop =torch .cat ((loss_prop ,loss_p .view (1 )))

            label_prop +=1e-9
            labels_p +=1e-9
            log_labels_p =torch .log (labels_p )
            one_hot_matrix =F .one_hot (lbs_u_guess ,num_classes =n_classes )
            one_hot_matrix =one_hot_matrix .float ()
            one_hot_matrix =torch .mean (one_hot_matrix ,dim =0 )

            one_hot_matrix +=1e-9
            log_one_hot_matrix =torch .log (one_hot_matrix )

            kl_soft =F .kl_div (log_labels_p ,label_prop ,reduction ='batchmean')

            kl_hard =F .kl_div (log_one_hot_matrix ,label_prop ,reduction ='batchmean')
            kl_divergence =torch .cat ((kl_divergence ,kl_soft .view (1 )))
            kl_divergence_hard =torch .cat ((kl_divergence_hard ,kl_hard .view (1 )))
        kl_divergence =kl_divergence .mean ()
        kl_divergence_hard =kl_divergence_hard .mean ()
        loss_prop =loss_prop .mean ()
        probs =torch .softmax (logits_u_w ,dim =1 )
        probs =probs .mean (dim =0 )
        prior =torch .full_like (probs ,0.1 ).detach ()
        prior =proportion .mean (dim =0 ).detach ()

        loss_debais =llp_loss (prior ,probs )
        x =loss_prop *bagsize
        loss =loss_prop
        with torch.no_grad():
            logits_u_w = logits_u_w.detach()
            probs = torch.softmax(logits_u_w, dim=1)
            eps = 1e-12  # 防止 log(0)
            entropy_per_sample = -(probs * (probs + eps).log()).sum(dim=1)  # [B]
            entropy_new = entropy_per_sample.mean()
            fl = 0
            new_probs = []  # 存储修改后的 probs 块

            for i in range(0, probs.size(0), args.bagsize):
                # 获取当前块
                prob_chunk = probs[i:i + args.bagsize]

                # 获取当前块的列约束比例并调整
                lap = torch.tensor(label_proportions[fl], dtype=torch.float64).cuda()
                lap = lap * args.bagsize

                # 使用 distributed_sinkhorn 函数处理每个块
                adjusted_chunk = distributed_sinkhorn(prob_chunk, args, lap)
                new_probs.append(adjusted_chunk)

                fl += 1

            # 将所有处理过的块重新组合成一个新的 probs 张量
            new_probs = torch.cat(new_probs, dim=0)
            scores, lbs_u_guess = torch.max(new_probs, dim=1)
            mask = scores.ge(args.thr).float()
            scores, lbs_u_guess = torch.max(probs, dim=1)
            # probs.shape=(1024, 10)

            mask = scores.ge(args.thr).float()
            lambda_b = torch.Tensor([]).cuda()

            entropy_i = calc_instance_entropy(probs)
            lambda_i = normal(entropy_i, h_tilde=0, beta=1)

            # print(label_proportions)
            # print(label_proportions.shape)

            entropy_b = calc_bag_entropy(probs.reshape(length, bagsize, args.n_classes))
            # print(entropy_b.shape)

            _, indices = torch.max(probs, dim=1)  # 获取概率最大值对应的索引
            one_hot = torch.zeros_like(probs).scatter_(1, indices.unsqueeze(1), 1)

            chunks_prob = [probs[i * chunk_size:(i + 1) * chunk_size] for i in range(length)]
            chunks = [one_hot[i * chunk_size:(i + 1) * chunk_size] for i in range(length)]
            for i, chunk in enumerate(chunks):
                _, indices = torch.max(chunk, dim=1)
                chunk_mean = torch.mean(chunk, dim=0)
                label_proportion = torch.tensor(label_proportions[i], dtype=torch.float64).cuda()
                difference = chunk_mean - label_proportion
                abs_difference = torch.abs(difference)
                errors = 1 - abs_difference.pow(1 / 2)

                opt_entropy_b = calc_opt_entropy(label_proportion * bagsize)
                # print(opt_entropy_b)
                entropy_b_normal = normal(entropy_b[i], h_tilde=opt_entropy_b, beta=1)
                # print(entropy_b_normal)

                selected_errors = errors[0][indices]
                selected_entropy_b_normal = entropy_b_normal[0][indices]
                # print(selected_entropy_b_normal.shape)
                # print((selected_errors * selected_entropy_b_normal).shape)

            lambda_total = 0

        loss_u =(criteria_u (logits_u_w ,lbs_u_guess )).mean ()
        loss =(2*loss_prop + loss_u)
        optim .zero_grad ()
        loss .backward ()
        optim .step ()
        lr_schdlr .step ()

        if args .eval_ema :
            with torch .no_grad ():
                ema_model_update (model ,ema_model ,args .ema_m )
        loss_prop_meter .update (loss .item ())
        mask_meter .update (mask .mean ().item ())
        kl_meter .update (kl_divergence .mean ().item ())
        kl_hard_meter .update (kl_divergence_hard .mean ().item ())
        corr_u_lb =(lbs_u_guess ==lbs_u_real ).float ()*mask
        n_correct_u_lbs_meter .update (corr_u_lb .sum ().item ())
        n_strong_aug_meter .update (mask .sum ().item ())
        entropy_meter .update (entropy_new )

        if (it +1 )%n_iter ==0 :
            t =time .time ()-epoch_start

            lr_log =[pg ['lr']for pg in optim .param_groups ]
            lr_log =sum (lr_log )/len (lr_log )
            logger .info ("{}-x{}-s{}, {} | epoch:{}, iter: {}.  loss: {:.3f}. kl: {:.3f}. kl_hard:{:.3f}."
            "LR: {:.3f}. Entropy: {:.2f}. Time: {:.2f}".format (
            args .dataset ,args .n_labeled ,args .seed ,args .exp_dir ,epoch ,it +1 ,loss_prop_meter .avg ,kl_meter .avg ,
            kl_hard_meter .avg ,lr_log ,entropy_meter .avg,t ))

            epoch_start =time .time ()

            bagsize =getattr (args ,"bagsize",getattr (args ,"bag_size",None ))
            if bagsize is None :
                raise ValueError ("s")

            exp_dir_name =os .path .basename (os .path .normpath (args .exp_dir ))

            os .makedirs (args .exp_dir ,exist_ok =True )
            csv_name =f"{args.dataset}_{exp_dir_name}_{bagsize}.csv"
            csv_path =os .path .join (args .exp_dir ,csv_name )


            is_new =not os .path .exists (csv_path )
            with open (csv_path ,"a",newline ="")as f :
                writer =csv .writer (f )
                if is_new :
                    writer .writerow (["epoch","time_sec"])
                writer .writerow ([epoch ,round (t ,2 )])

            logger .info (f"CSV path -> {os.path.abspath(csv_path)}")
    return loss_prop_meter .avg ,n_correct_u_lbs_meter .avg ,n_strong_aug_meter .avg ,mask_meter .avg ,kl_meter .avg ,kl_hard_meter .avg


def evaluate(model, ema_model, dataloader, dataset):
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

    num_classes = None

    entropy_sum = 0.0
    entropy_count = 0

    def _prf(y_true, y_pred, n_cls, average="macro", eps=1e-12):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        cm = np.zeros((n_cls, n_cls), dtype=np.int64)
        np.add.at(cm, (y_true, y_pred), 1)

        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)

        if average == "micro":
            TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
            p = TP / (TP + FP + eps)
            r = TP / (TP + FN + eps)
            f = 2 * p * r / (p + r + eps)
            return float(p), float(r), float(f)
        else:  # macro
            return float(prec.mean()), float(rec.mean()), float(f1.mean())

    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            if dataset in ["MNIST", "FashionMNIST", "KMNIST"]:
                ims = ims.permute(0, 2, 1, 3)

            logits = model(ims)
            if num_classes is None:
                num_classes = logits.size(1)

            loss = torch.nn.CrossEntropyLoss()(logits, lbs)
            loss_meter.update(loss.item())

            scores = torch.softmax(logits, dim=1)

            # ===== 新增：普通模型 entropy（按 batch 加权平均）=====
            eps = 1e-12
            ent = -(scores * (scores + eps).log()).sum(dim=1)   # [B]
            entropy_sum += ent.sum().item()
            entropy_count += ent.numel()
            # ===============================================

            preds = scores.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbs.cpu().numpy())

            top1, top5 = accuracy(scores, lbs, (1, 2))
            top1_meter.update(top1.item())
            top5_meter.update(top5.item())

            if ema_model is not None:
                ema_logits = ema_model(ims)
                ema_scores = torch.softmax(ema_logits, dim=1)
                ema_preds = ema_scores.argmax(dim=1)
                ema_all_preds.extend(ema_preds.cpu().numpy())
                ema_all_labels.extend(lbs.cpu().numpy())

                ema_top1, ema_top5 = accuracy(ema_scores, lbs, (1, 2))
                ema_top1_meter.update(ema_top1.item())

    prec, rec, f1 = _prf(all_labels, all_preds, num_classes, average="macro")
    entropy = entropy_sum / max(1, entropy_count)

    return (top1_meter.avg, ema_top1_meter.avg,
            top5_meter.avg, ema_top5_meter.avg,
            loss_meter.avg, prec, rec, f1, entropy)


def main ():
    parser =argparse .ArgumentParser (description ='DLLP Cifar Training')
    parser .add_argument ('--root',default ='./data',type =str ,help ='dataset directory')
    parser .add_argument ('--wresnet-k',default =4 ,type =int ,
    help ='width factor of wide resnet')
    parser .add_argument ('--wresnet-n',default =16 ,type =int ,
    help ='depth of wide resnet')
    parser .add_argument ('--dataset',type =str ,default ="KMNIST",
    help ='number of classes in dataset')
    parser .add_argument ('--n-classes',type =int ,default =10 ,
    help ='number of classes in dataset')
    parser .add_argument ('--n-labeled',type =int ,default =10 ,
    help ='number of labeled samples for training')
    parser .add_argument ('--n-epoches',type =int ,default =500 ,
    help ='number of training epoches')
    parser .add_argument ('--batchsize',type =int ,default =8 ,
    help ='train batch size of bag samples')
    parser .add_argument ('--bagsize',type =int ,default =128 ,
    help ='train bag size of samples')
    parser .add_argument ('--n-imgs-per-epoch',type =int ,default =1024 ,
    help ='number of training images for each epoch')

    parser .add_argument ('--eval-ema',default =False ,help ='whether to use ema model for evaluation')
    parser .add_argument ('--ema-m',type =float ,default =0.999 )

    parser .add_argument ('--lam-u',type =float ,default =1. ,
    help ='c oefficient of unlabeled loss')
    parser .add_argument ('--lr',type =float ,default =0.01 ,
    help ='learning rate for training')
    parser .add_argument ('--weight-decay',type =float ,default =5e-4 ,
    help ='weight decay')
    parser .add_argument ('--momentum',type =float ,default =0.9 ,
    help ='momentum for optimizer')
    parser .add_argument ('--seed',type =int ,default =1 ,
    help ='seed for random behaviors, no seed if negtive')
    parser.add_argument(
        '--pl_method',
        type=str,
        default='cluster',
        choices=['random', 'cluster', 'alphafirst'],
        help='pseudo label selection method: random | cluster | alphafirst'
    )
    parser .add_argument ('--temperature',default =0.2 ,type =float ,help ='softmax temperature')
    parser .add_argument ('--low-dim',type =int ,default =64 )
    parser .add_argument ('--lam-c',type =float ,default =1 ,
    help ='coefficient of contrastive loss')
    parser .add_argument ('--lam-p',type =float ,default =2 ,
    help ='coefficient of proportion loss')
    parser .add_argument ('--contrast-th',default =0.8 ,type =float ,
    help ='pseudo label graph threshold')
    parser .add_argument ('--thr',type =float ,default =0.95 ,
    help ='pseudo label threshold')
    parser .add_argument ('--alpha',type =float ,default =0.9 )
    parser .add_argument ('--queue_batch',type =float ,default =5 ,
    help ='number of batches stored in memory bank')
    parser .add_argument ('--sinkhorn-iterations',type =float ,default =3 )

    parser .add_argument ('--exp-dir',default ='ROT',type =str ,help ='experiment id')
    parser .add_argument ('--checkpoint',default ='',type =str ,help ='use pretrained model')
    parser .add_argument ('--folds',default ='2',type =str ,help ='number of dataset')
    args =parser .parse_args ()

    logger ,output_dir =setup_default_logging (args )
    logger .info (dict (args ._get_kwargs ()))

    tb_logger =tensorboard_logger .Logger (logdir =output_dir ,flush_secs =2 )
    samp_ran =37821
    if args .seed >0 :
        torch .manual_seed (args .seed )
        random .seed (args .seed )
        np .random .seed (args .seed )

    n_iters_per_epoch =args .n_imgs_per_epoch # 1024

    logger .info ("***** Running training *****")
    logger .info (f"  Task = {args.dataset}@{args.n_labeled}")

    model ,criteria_x ,criteria_u ,ema_model =set_model (args )
    logger .info ("Total params: {:.2f}M".format (
    sum (p .numel ()for p in model .parameters ())/1e6 ))
    if args.pl_method == 'random':
        from datasets.cifar import get_train_loader
    elif args.pl_method == 'alphafirst':
        from datasets.cifar_dir import get_train_loader
    elif args.pl_method == 'cluster':
        from datasets.cifar_cluster2 import get_train_loader
    else:
        raise ValueError(f"Unknown pl_method: {args.pl_method}")
    dltrain_u ,dataset_length ,_ =get_train_loader (args .n_classes ,
    args .dataset ,args .batchsize ,args .bagsize ,root =args .root ,
    method ='DLLP',
    supervised =False )
    dlval =get_val_loader (dataset =args .dataset ,batch_size =64 ,num_workers =2 ,root =args .root )
    n_iters_all =len (dltrain_u )*args .n_epoches
    wd_params ,non_wd_params =[],[]
    for name ,param in model .named_parameters ():
        if 'bn'in name :
            non_wd_params .append (param )
        else :
            wd_params .append (param )
    param_list =[
    {'params':wd_params },{'params':non_wd_params ,'weight_decay':0 }]
    optim =torch .optim .SGD (param_list ,lr =args .lr ,weight_decay =args .weight_decay ,
    momentum =args .momentum ,nesterov =True )

    lr_schdlr =WarmupCosineLrScheduler (optim ,n_iters_all ,warmup_iter =0 )

    # memory bank
    args .queue_size =5120
    queue_feats =torch .zeros (args .queue_size ,args .low_dim ).cuda ()
    queue_probs =torch .zeros (args .queue_size ,args .n_classes ).cuda ()
    queue_ptr =0

    # for distribution alignment
    prob_list =[]

    train_args =dict (
    model =model ,
    ema_model =ema_model ,
    prob_list =prob_list ,
    criteria_x =criteria_x ,
    criteria_u =criteria_u ,
    optim =optim ,
    lr_schdlr =lr_schdlr ,
    dltrain_u =dltrain_u ,
    args =args ,
    n_iters =n_iters_per_epoch ,
    logger =logger
    )

    best_acc =-1
    best_acc_5 =-1
    best_epoch_5 =0

    best_epoch =0

    logger .info ('-----------start training--------------')
    for epoch in range (args .n_epoches ):
        loss_prob ,n_correct_u_lbs ,n_strong_aug ,mask_mean ,num_pos ,samp_lb =train_one_epoch (epoch ,bagsize =args .bagsize ,n_classes =args .n_classes ,**train_args ,samp_ran =samp_ran ,
        )

        top1 ,ema_top1 ,top5 ,ema_top5 ,loss_test,prec, rec, f1,entropy =evaluate (model ,ema_model ,dlval ,args .dataset )
        tb_logger .log_value ('loss_prob',loss_prob ,epoch )
        if (n_strong_aug ==0 ):
            tb_logger .log_value ('guess_label_acc',0 ,epoch )
        else :
            tb_logger .log_value ('guess_label_acc',n_correct_u_lbs /n_strong_aug ,epoch )
        tb_logger .log_value ('test_acc',top1 ,epoch )
        tb_logger .log_value ('mask',mask_mean ,epoch )

        tb_logger.log_value('test_precision', prec, epoch)
        tb_logger.log_value('test_recall', rec, epoch)
        tb_logger.log_value('test_f1', f1, epoch)
        tb_logger.log_value('test_entropy', entropy, epoch)
        tb_logger .log_value ('loss_test',loss_test ,epoch )
        if best_acc <top1 :
            best_acc =top1
            best_epoch =epoch
        if best_acc_5 <top5 :
            best_acc_5 =top5
            best_epoch_5 =epoch
        logger.info(
            "Epoch {}.loss_test: {:.4f}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f} in epoch{},"
            "Acc_5: {:.4f}.  best_acc_5: {:.4f} in epoch{},"
            " Precision: {:.4f}. Recall: {:.4f}. F1: {:.4f}. Entropy: {:.4f}.".
            format(epoch, loss_test, top1, ema_top1, best_acc, best_epoch,
                   top5, best_acc_5, best_epoch_5,
                   prec, rec, f1, entropy)
        )
        if epoch %1000 ==0 :
            save_obj ={
            'model':model .state_dict (),
            'optimizer':optim .state_dict (),
            'lr_scheduler':lr_schdlr .state_dict (),
            'prob_list':prob_list ,
            'queue':{'queue_feats':queue_feats ,'queue_probs':queue_probs ,'queue_ptr':queue_ptr },
            'epoch':epoch ,
            }
            torch .save (save_obj ,os .path .join (output_dir ,'checkpoint_%02d.pth'%epoch ))


if __name__ =='__main__':
    main ()