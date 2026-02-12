from __future__ import print_function
import random

import time
import argparse
import os
import sys
import os ,csv

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
from functools import lru_cache

import torch
import math
from typing import Optional, Sequence, Union,Dict





@torch.no_grad()
def llp_high_order_mse_metrics(
    labels_proportion: torch.Tensor,   # [B, C] 每个 bag 的真实比例
    y: torch.Tensor,                   # [B, C] 每个 bag 的预测比例（已经聚合好的）
    max_order: int = 3,                # 最高阶，比如 3 就是 1/2/3 阶
    bag_sizes: Union[torch.Tensor, Sequence[int], None] = None,  # [B] 每个 bag 的长度，可选
) -> Dict[int, torch.Tensor]:
    """
    返回一个 dict: {1: mse_1阶, 2: mse_2阶, 3: mse_3阶, ...}
    这里的第 k 阶就是先对比例做 x**k / y**k，然后算 MSE。
    """

    x = labels_proportion.to(dtype=torch.float64, device=y.device)  # [B, C]
    y = y.to(dtype=torch.float64)                                   # [B, C]

    if bag_sizes is not None:
        if not isinstance(bag_sizes, torch.Tensor):
            bag_sizes = torch.tensor(bag_sizes, device=y.device, dtype=torch.float64)
        else:
            bag_sizes = bag_sizes.to(device=y.device, dtype=torch.float64)
        assert bag_sizes.shape[0] == x.shape[0], \
            f"bag_sizes.shape={bag_sizes.shape} vs x.shape[0]={x.shape[0]}"

    metrics: Dict[int, torch.Tensor] = {}

    for k in range(1, max_order + 1):
        # 第 k 阶：直接做 x**k, y**k，然后算 MSE
        xk = x ** k                     # [B, C]
        yk = y ** k                     # [B, C]

        mse_per_bag = ((xk - yk) ** 2).mean(dim=1)   # [B]

        if bag_sizes is None:
            mse_k = mse_per_bag.mean()
        else:
            mse_k = (mse_per_bag * bag_sizes).sum() / bag_sizes.sum()

        metrics[k] = mse_k.to(dtype=labels_proportion.dtype)

    return metrics

def cross_entropy_loss_torch (softmax_matrix ,onehot_labels ):

    log_softmax =torch .log (softmax_matrix +1e-12 )

    cross_entropy =-torch .sum (onehot_labels *log_softmax ,dim =1 )

    mean_loss =torch .mean (cross_entropy )
    return mean_loss
class LinearClassifier (nn .Module ):
    def __init__ (self ,input_dim ,output_dim ):
        super ().__init__ ()
        self .fc =nn .Linear (input_dim ,output_dim )
    def forward (self ,x ):
        return self .fc (x )



def set_model (args ,input_dim ):
    if args .dataset in ['CIFAR10','SVHN','CIFAR100','miniImageNet']:
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.n_classes)
    elif args .dataset in [
    'Corel16k','Corel5k','Delicious','Bookmarks',
    'Eurlex_DC','Eurlex_SM','Scene','Yeast'
    ]:
        model =LinearClassifier (
        input_dim =input_dim ,
        output_dim =args .n_classes
        )

    else :
        model =LeNet5 ()


    if args .checkpoint :
        ckpt =torch .load (args .checkpoint )
        msg =model .load_state_dict (ckpt ,strict =False )
        assert set (msg .missing_keys )=={"classifier.weight","classifier.bias"}
        print ('loaded from checkpoint:',args .checkpoint )

    model .train ().cuda ()

    if args .eval_ema :
        ema_model =copy .deepcopy (model ).cuda ().eval ()
        for p in ema_model .parameters ():
            p .requires_grad =False
    else :
        ema_model =None

    criteria_x =nn .CrossEntropyLoss ().cuda ()
    criteria_u =nn .CrossEntropyLoss (reduction ='none').cuda ()

    return model ,criteria_x ,criteria_u ,ema_model


@torch .no_grad ()
def ema_model_update (model ,ema_model ,ema_m ):

    for param_train ,param_eval in zip (model .parameters (),ema_model .parameters ()):
        param_eval .copy_ (param_eval *ema_m +param_train .detach ()*(1 -ema_m ))

    for buffer_train ,buffer_eval in zip (model .buffers (),ema_model .buffers ()):
        buffer_eval .copy_ (buffer_train )


def llp_loss (labels_proportion ,y ):
    x =torch .tensor (labels_proportion ,dtype =torch .float64 ).cuda ()
    x =x .squeeze (0 )



    y =y .double ()
    cross_entropy =torch .sum (-x *(torch .log (y )+1e-7 ))
    mse_loss =torch .mean ((x -y )**2 )

    return cross_entropy


def custom_loss (probs ,lambda_val =1.0 ):

    log_probs =torch .log (probs )

    product =-probs *log_probs

    loss =torch .sum (product )

    loss =lambda_val *loss

    return loss



def thre_ema (thre ,sum_values ,ema ):
    return thre *ema +(1 -ema )*sum_values


def weight_decay_with_mask (mask ,initial_weight ,max_mask_count ):
    mask_count =mask .sum ().item ()
    weight_decay =max (0 ,1 -mask_count /max_mask_count )
    return initial_weight *weight_decay

from typing import Union, Sequence

def llp_loss_batch(
    labels_proportion: torch.Tensor,   # [B, C]  每个 bag 的真实比例
    y: torch.Tensor,                   # [B, C]  每个 bag 的预测比例（已经聚合好的）
    bag_sizes: Union[torch.Tensor, Sequence[int], None] = None,  # [B] 每个 bag 的长度，可选
    reduce: str = "mean",
    eps: float = 1e-7,
) -> torch.Tensor:

    # 转 double 做数值更稳
    x = labels_proportion.to(dtype=torch.float64, device=y.device)  # [B, C]
    y = y.to(dtype=torch.float64)                                   # [B, C]

    # 逐 bag 的 CE / MSE
    cross_entropy = -(x * torch.log(y.clamp_min(eps))).sum(dim=1)   # [B]
    mse_loss      = ((x - y) ** 2).mean(dim=1)                      # [B]

    loss = cross_entropy   # 你原来只用 CE

    if reduce is None:
        return loss.to(dtype=labels_proportion.dtype)

    # 如果给了 bag_sizes，用它做加权平均
    if bag_sizes is not None:
        if not isinstance(bag_sizes, torch.Tensor):
            bag_sizes = torch.tensor(bag_sizes, device=loss.device, dtype=loss.dtype)
        else:
            bag_sizes = bag_sizes.to(device=loss.device, dtype=loss.dtype)

        assert bag_sizes.shape[0] == loss.shape[0], \
            f"bag_sizes.shape={bag_sizes.shape} vs loss.shape={loss.shape}"

    if reduce == "mean":
        if bag_sizes is None:
            out = loss.mean()
        else:
            # 按 bagsize 加权平均
            out = (loss * bag_sizes).sum() / bag_sizes.sum()
        return out.to(dtype=labels_proportion.dtype)

    elif reduce == "sum":
        return loss.sum().to(dtype=labels_proportion.dtype)

    else:
        raise ValueError("reduce must be None|'mean'|'sum'")

import math
from functools import lru_cache
from typing import Optional, Sequence, Union

import torch


# -----------------------------
# helpers: compositions (stars and bars)
# -----------------------------
import math
from functools import lru_cache
from typing import Optional, Sequence, Union

import torch


@lru_cache(maxsize=None)
def _compositions_cpu(C: int, r: int) -> torch.Tensor:
    """
    All kappa in N^C s.t. sum kappa = r.
    Returns K_cpu: LongTensor [P, C] on CPU, where P = comb(C+r-1, r).
    """
    comps = []

    def rec(pos, remaining, cur):
        if pos == C - 1:
            comps.append(cur + [remaining])
            return
        for v in range(remaining + 1):
            rec(pos + 1, remaining - v, cur + [v])

    rec(0, r, [])
    return torch.tensor(comps, dtype=torch.long, device="cpu")


def _ensure_bag_sizes(bag_sizes, B: int, device, dtype) -> Optional[torch.Tensor]:
    """
    Normalize bag_sizes into shape [B] float tensor on device.
    Accept int/float, 0-d tensor, list/tuple, [B], [B,1].
    """
    if bag_sizes is None:
        return None

    if isinstance(bag_sizes, (int, float)):
        return torch.full((B,), float(bag_sizes), device=device, dtype=dtype)

    if not isinstance(bag_sizes, torch.Tensor):
        bs = torch.tensor(bag_sizes, device=device, dtype=dtype)
    else:
        bs = bag_sizes.to(device=device, dtype=dtype)

    if bs.dim() == 0:
        bs = bs.expand(B).clone()
    else:
        bs = bs.reshape(-1)

    if bs.numel() != B:
        raise ValueError(f"bag_sizes needs length B={B}, got numel={bs.numel()}, shape={tuple(bs.shape)}")

    return bs


def _log_multinomial_multiplicity(K: torch.Tensor, r: int) -> torch.Tensor:
    """
    log mult(kappa) = log(r!) - sum_c log(kappa_c!)
    K: [P, C] long
    return log_mult: [P] float64
    """
    Kf = K.to(dtype=torch.float64)
    log_rfact = torch.lgamma(torch.tensor(float(r + 1), device=K.device, dtype=torch.float64))
    return log_rfact - torch.lgamma(Kf + 1.0).sum(dim=1)


def _log_falling_step_table(alpha: torch.Tensor, m: torch.Tensor, r: int, eps: float):
    """
    L[:,:,k] = log prod_{j=0}^{k-1} clamp(alpha - j/m, eps)
    L[:,:,0] = 0
    """
    B, C = alpha.shape
    if r == 0:
        return torch.zeros((B, C, 1), device=alpha.device, dtype=alpha.dtype)

    j = torch.arange(r, device=alpha.device, dtype=alpha.dtype).view(1, 1, r)  # [1,1,r]
    term = alpha.unsqueeze(-1) - j / m.unsqueeze(-1)                            # [B,C,r]
    term = term.clamp_min(eps)                                                  # 关键：永远 >0
    log_term = torch.log(term)                                                  # [B,C,r]
    csum = log_term.cumsum(dim=-1)                                              # [B,C,r]
    zeros = torch.zeros((B, C, 1), device=alpha.device, dtype=alpha.dtype)
    return torch.cat([zeros, csum], dim=-1)                                     # [B,C,r+1]

def llp_high_order_loss_batch_taylor(
    labels_proportion: torch.Tensor,   # [B, C]
    y: torch.Tensor,                   # [B, C] bag-level predicted proportion (already aggregated)
    max_order: Union[int, float, torch.Tensor] = 1,
    loss_type: str = "ce",             # "ce" or "mse"
    order_weights: Optional[Sequence[float]] = None,
    bag_sizes: Union[torch.Tensor, Sequence[int], int, float, None] = None,
    reduce: str = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Numerically-stable Taylor/factorial-moment high-order LLP loss (log-domain).
    - CE: computed fully in log space (stable).
    - MSE: computed on normalized distributions (stable). (Recommended)
    """

    # ----- max_order -> int -----
    if isinstance(max_order, torch.Tensor):
        if max_order.numel() != 1:
            raise ValueError(f"max_order tensor must be scalar, got {tuple(max_order.shape)}")
        max_order = int(max_order.item())
    else:
        max_order = int(max_order)
    if max_order < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")

    if labels_proportion.dim() != 2 or y.dim() != 2:
        raise ValueError(f"labels_proportion={tuple(labels_proportion.shape)}, y={tuple(y.shape)} must be [B,C]")
    B, C = labels_proportion.shape
    if y.shape != (B, C):
        raise ValueError(f"y.shape={tuple(y.shape)} must match labels_proportion.shape={(B,C)}")

    device = y.device
    dtype_work = torch.float64  # highest practical precision on GPU

    x = labels_proportion.to(device=device, dtype=dtype_work)
    y = y.to(device=device, dtype=dtype_work)

    # ----- bag_sizes -> m -----
    bs = _ensure_bag_sizes(bag_sizes, B=B, device=device, dtype=dtype_work)
    if max_order > 1 and bs is None:
        raise ValueError("bag_sizes (m per bag) is required when max_order > 1 for Taylor/factorial terms.")
    if bs is None:
        m = torch.ones((B, 1), device=device, dtype=dtype_work)
    else:
        m = bs.view(B, 1).clamp_min(1.0)

    # ----- normalize alpha / alpha_hat -----
    p_true_1 = x.clamp_min(0.0)
    sum_true = p_true_1.sum(dim=1, keepdim=True)
    p_true_1 = torch.where(sum_true > 0, p_true_1 / sum_true.clamp_min(eps), p_true_1)

    p_pred_1 = y.clamp_min(eps)
    sum_pred = p_pred_1.sum(dim=1, keepdim=True)
    p_pred_1 = torch.where(sum_pred > 0, p_pred_1 / sum_pred, p_pred_1)

    # ----- weights -----
    if order_weights is None:
        order_weights = [1.0 / max_order] * max_order
    else:
        if len(order_weights) != max_order:
            raise ValueError(f"order_weights length must be {max_order}, got {len(order_weights)}")
        s = float(sum(order_weights))
        order_weights = [float(w) / s for w in order_weights]

    loss_per_bag = torch.zeros((B,), device=device, dtype=dtype_work)

    for r in range(1, max_order + 1):
        # compositions K: [P,C]
        K = _compositions_cpu(C, r).to(device=device)  # long
        Pn = K.shape[0]

        # log multiplicity: [P]
        log_mult = _log_multinomial_multiplicity(K, r)  # float64

        # log falling-step tables: [B,C,r+1]
        L_true = _log_falling_step_table(p_true_1, m, r, eps=eps)
        L_pred = _log_falling_step_table(p_pred_1, m, r, eps=eps)

        # log_val(b,kappa) = sum_c L[:,c,kappa_c]
        log_val_true = torch.zeros((B, Pn), device=device, dtype=dtype_work)
        log_val_pred = torch.zeros((B, Pn), device=device, dtype=dtype_work)

        for c in range(C):
            idx = K[:, c].view(1, -1).expand(B, -1)          # [B,P]
            log_val_true = log_val_true + L_true[:, c, :].gather(1, idx)
            log_val_pred = log_val_pred + L_pred[:, c, :].gather(1, idx)

        # log_mass = log_val + log_mult
        log_mass_true = log_val_true + log_mult.view(1, -1)
        log_mass_pred = log_val_pred + log_mult.view(1, -1)

        # normalize => log probabilities
        logZ_true = torch.logsumexp(log_mass_true, dim=1, keepdim=True)
        logZ_pred = torch.logsumexp(log_mass_pred, dim=1, keepdim=True)
        log_pt = log_mass_true - logZ_true
        log_pp = log_mass_pred - logZ_pred

        if loss_type == "ce":
            # CE = - sum pt * log pp, with pt = exp(log_pt)
            pt = torch.exp(log_pt)  # stable because log_pt <= 0 typically
            loss_r = -torch.sum(pt * log_pp, dim=1)

        elif loss_type == "mse":
            # stable MSE on normalized distributions
            pt = torch.exp(log_pt)
            pp = torch.exp(log_pp)
            loss_r = torch.mean((pt - pp) ** 2, dim=1)

        else:
            raise ValueError("loss_type must be 'ce' or 'mse'")

        loss_per_bag = loss_per_bag + float(order_weights[r - 1]) * loss_r

    # ----- reduce -----
    if reduce is None:
        return loss_per_bag.to(dtype=labels_proportion.dtype)

    if reduce == "mean":
        if bs is None:
            out = loss_per_bag.mean()
        else:
            out = (loss_per_bag * bs).sum() / bs.sum().clamp_min(eps)
        return out.to(dtype=labels_proportion.dtype)

    if reduce == "sum":
        return loss_per_bag.sum().to(dtype=labels_proportion.dtype)

    raise ValueError("reduce must be None|'mean'|'sum'")


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
    order1 =AverageMeter ()
    order2 =AverageMeter ()
    order3 =AverageMeter ()

    # the number of edges in the pseudo-label graph
    entropy_meter =AverageMeter ()
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

        positions = torch.nonzero(lbs_idx == 37821).squeeze()

        if positions.numel() != 0:
            head = positions - positions % bagsize
            rear = head + bagsize - 1

        for i in range(length):
            labels = []
            for j in range(n_classes):
                labels.append(var2[j][i])
            label_proportions[i].append(labels)

        # --------------------------------------
        btu = ims_u_weak.size(0)

        if args.dataset in ["MNIST", "FashionMNIST", "KMNIST"]:
            ims_u_weak = ims_u_weak.permute(0, 2, 1, 3)
        bt = 0
        imgs = torch.cat([ims_u_weak], dim=0).cuda()
        logits = model(imgs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]  # 取真正的分类输出那个 tensor

        # logits_x = logits[:bt]
        logits_u_w = torch.split(logits[0:], btu)
        logits_u_w = logits_u_w[0]
        proportion = torch.stack([torch.stack(lp[0]) for lp in label_proportions]).cuda()
        proportion = proportion.view(length, n_classes, 1)
        proportion = proportion.squeeze(-1)
        proportion = proportion.double()
        #print(proportion[1])
        probs = torch.softmax(logits_u_w, dim=-1)
        N, C = probs.shape
        s = bagsize
        assert N % s == 0, f"N={N} 不能被 bagsize={s} 整除"
        B = N // s

        labels_p_batch = probs.contiguous().view(B, s, C)
        bag_preds = labels_p_batch.mean(dim=1)  # [B, C]

        # loss_x = criteria_x(logits_x, lbs_x)

        chunk_size = len(logits_u_w) // length

        loss_prop = torch.Tensor([]).cuda()
        loss_prop = loss_prop.double()
        kl_divergence = torch.Tensor([]).cuda()
        kl_divergence = kl_divergence.double()
        kl_divergence_hard = torch.Tensor([]).cuda()
        kl_divergence_hard = kl_divergence_hard.double()

        loss = llp_loss_batch(proportion, bag_preds, reduce="mean")

        order_mse_dict = llp_high_order_mse_metrics(
            labels_proportion=proportion,  # [B, C]
            y=bag_preds,  # [B, C]
            max_order=3,  # 一般是 3
            bag_sizes=None,  # 如果你用 bagsize 加权，就传进去
        )
        mse_1 = order_mse_dict[1].item()*bagsize*bagsize
        mse_2 = order_mse_dict[2].item()*bagsize*bagsize*bagsize*bagsize
        mse_3 = order_mse_dict[3].item()*bagsize*bagsize*bagsize*bagsize*bagsize*bagsize
        order1.update(mse_1)
        order2.update(mse_2)
        order3.update(mse_3)


        x =1.2
        kl_divergence =x
        kl_divergence_hard =x
        loss_prop =loss_prop .mean ()
        probs =torch .softmax (logits_u_w ,dim =1 )
        #probs =probs .mean (dim =0 )
        prior =torch .full_like (probs ,0.1 ).detach ()
        prior =proportion .mean (dim =0 ).detach ()
        entropy =-(probs *probs .log ()).sum (dim =-1 )# shape: [batch_size]

        # (Optional) average entropy over the batch
        mean_entropy =entropy .mean ()
        with torch .no_grad ():

            probs =torch .softmax (logits_u_w ,dim =1 )

            scores ,lbs_u_guess =torch .max (probs ,dim =1 )
            mask =scores .ge (args .thr ).float ()

        optim .zero_grad ()
        loss .backward ()
        optim .step ()
        lr_schdlr .step ()

        if args .eval_ema :
            with torch .no_grad ():
                ema_model_update (model ,ema_model ,args .ema_m )
        loss_prop_meter .update (loss .item ())
        mask_meter .update (mask .mean ().item ())
        kl_meter .update (kl_divergence )
        kl_hard_meter .update (kl_divergence_hard )
        corr_u_lb =(lbs_u_guess ==lbs_u_real ).float ()*mask
        n_correct_u_lbs_meter .update (corr_u_lb .sum ().item ())
        n_strong_aug_meter .update (mask .sum ().item ())
        entropy_meter .update (mean_entropy )
        if (it +1 )%n_iter ==0 :
            t =time .time ()-epoch_start

            lr_log =[pg ['lr']for pg in optim .param_groups ]
            lr_log =sum (lr_log )/len (lr_log )
            logger .info ("{}-x{}-s{}, {} | epoch:{}, iter: {}.  loss: {:.3f}. kl: {:.3f}. kl_hard:{:.3f}."
            "LR: {:.3f}. Time: {:.2f}. Entropy: {:.2f}. order1: {:.6f}. order2: {:.6f}. order3: {:.6f}.".format (
            args .dataset ,args .n_labeled ,args .seed ,args .exp_dir ,epoch ,it +1 ,loss_prop_meter .avg ,kl_meter .avg ,
            kl_hard_meter .avg ,lr_log ,t ,entropy_meter .avg,order1.avg,order2.avg, order3.avg ))

            epoch_start =time .time ()
            bagsize =getattr (args ,"bagsize",getattr (args ,"bag_size",None ))
            if bagsize is None :
                raise ValueError (" args  bagsize  bag_size")

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
    parser .add_argument ('--dataset',type =str ,default ="CIFAR10",
    help ='number of classes in dataset')
    parser .add_argument ('--n-classes',type =int ,default =10 ,
    help ='number of classes in dataset')
    parser .add_argument ('--n-labeled',type =int ,default =10 ,
    help ='number of labeled samples for training')
    parser .add_argument ('--n-epoches',type =int ,default =500,
    help ='number of training epoches')
    parser .add_argument ('--batchsize',type =int ,default =8 ,
    help ='train batch size of bag samples')
    parser .add_argument ('--bagsize',type =int ,default = 128 ,
    help ='train bag size of samples')
    parser .add_argument ('--n-imgs-per-epoch',type =int ,default =1024 ,
    help ='number of training images for each epoch')

    parser .add_argument ('--eval-ema',default =False ,help ='whether to use ema model for evaluation')
    parser .add_argument ('--ema-m',type =float ,default =0.999 )

    parser .add_argument ('--lam-u',type =float ,default =1. ,
    help ='c oefficient of unlabeled loss')
    parser .add_argument ('--lr',type =float ,default =0.05 ,
    help ='learning rate for training')
    parser .add_argument ('--weight-decay',type =float ,default =5e-4 ,
    help ='weight decay')
    parser .add_argument ('--momentum',type =float ,default =0.9 ,
    help ='momentum for optimizer')
    parser .add_argument ('--seed',type =int ,default =100 ,
    help ='seed for random behaviors, no seed if negtive')

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

    parser.add_argument(
        '--pl_method',
        type=str,
        default='random',
        choices=['random', 'cluster', 'alphafirst'],
        help='pseudo label selection method: random | cluster | alphafirst'
    )
    parser .add_argument ('--exp-dir',default ='PM',type =str ,help ='experiment id')
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
    if args.pl_method == 'random':
        from datasets.cifar import get_train_loader
    elif args.pl_method == 'alphafirst':
        from datasets.cifar_dir import get_train_loader
    elif args.pl_method == 'cluster':
        from datasets.cifar_cluster2 import get_train_loader
    else:
        raise ValueError(f"Unknown pl_method: {args.pl_method}")
    dltrain_u ,dataset_length ,input_dim =get_train_loader (args .n_classes ,
    args .dataset ,args .batchsize ,args .bagsize ,root =args .root ,
    method ='DLLP',
    supervised =False )
    dlval =get_val_loader (dataset =args .dataset ,batch_size =64 ,num_workers =2 ,root =args .root )
    model ,criteria_x ,criteria_u ,ema_model =set_model (args ,input_dim )
    logger .info ("Total params: {:.2f}M".format (
    sum (p .numel ()for p in model .parameters ())/1e6 ))



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
                   prec, rec, f1,entropy)
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