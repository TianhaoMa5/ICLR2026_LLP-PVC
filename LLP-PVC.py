# -*- coding: utf-8 -*-
from __future__ import print_function

import os ,csv

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
from LeNet import LeNet5 ,MLPDropIn,LeNet5Plus
from torchvision import models

import math
def init_fc_bias_sigmoid_to_1_over_k(model: nn.Module, num_classes: int):
    K = int(num_classes)
    if K <= 1:
        return

    b0 = -math.log(K - 1)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear) and model.fc.bias is not None:
        with torch.no_grad():
            model.fc.bias.fill_(b0)

def cross_entropy_loss_torch (softmax_matrix ,onehot_labels ):

    log_softmax =torch .log (softmax_matrix +1e-12 )

    cross_entropy =-torch .sum (onehot_labels *log_softmax ,dim =1 )

    mean_loss =torch .mean (cross_entropy )
    return mean_loss
def set_model (args ):
    if args .dataset in ['CIFAR10','CIFAR100','miniImageNet','SVHN']:



        model = WideResnet(
            n_classes=args.n_classes,
            k=args.wresnet_k,
            n=args.wresnet_n,
            proj=False
        )
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.n_classes)
        init_fc_bias_sigmoid_to_1_over_k(model, args.n_classes)

    else :
        model = WideResnet(
            n_classes=args.n_classes,
            k=args.wresnet_k,
            n=args.wresnet_n,
            proj=False
        )
        model =LeNet5 (num_classes=args.n_classes)
        init_fc_bias_sigmoid_to_1_over_k(model, args.n_classes)

        #model = ResNet18CIFAR10(num_classes=args.n_classes)

    if args .checkpoint :
        checkpoint =torch .load (args .checkpoint )

        msg =model .load_state_dict (checkpoint ,strict =False )


        assert set (msg .missing_keys )=={"classifier.weight","classifier.bias"}
        print ('loaded from checkpoint: %s'%args .checkpoint )
    model .train ()
    model .cuda ()

    if args .eval_ema :
        if args .dataset in ['CIFAR10','CIFAR100','SVHN']:
            ema_model =WideResnet (
            n_classes =args .n_classes ,
            k =args .wresnet_k ,
            n =args .wresnet_n ,
            proj =False
            )
            model =models .resnet18 (pretrained =False )
            model .fc =nn .Linear (model .fc .in_features ,10 )
            #ema_model = resnet34(num_classes=10)

        else :
            ema_model =MLPDropIn ()
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
    x =x .squeeze (0 )#  x.squeeze()

    # Ensure y is also double

    y =y .double ()
    cross_entropy =torch .sum (-x *(torch .log (y )+1e-7 ))
    mse_loss =torch .mean ((x -y )**2 )

    return mse_loss
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

LN2 =math .log (2.0 )




def compute_CC_loss_dp_precise_batched(
    softmax_p_batch: torch.Tensor,      # [B, s, C]
    proportions_batch: torch.Tensor,    # [B, C]
    reduce: str = "sum",
    eps: float = 1e-12
) -> torch.Tensor:
    assert softmax_p_batch.dim() == 3, f"softmax_p_batch should be [B,s,C], got {softmax_p_batch.shape}"
    assert proportions_batch.dim() == 2, f"proportions_batch should be [B,C], got {proportions_batch.shape}"
    B, s, C = softmax_p_batch.shape
    assert proportions_batch.shape == (B, C), f"proportions shape {proportions_batch.shape} vs (B,C)=({B},{C})"

    dev = softmax_p_batch.device

    try:
        from torch.func import vmap
        def _one(softmax_p, proportions):
            return compute_CC_loss_fft_precise(softmax_p, proportions, eps=eps)
        loss_b = vmap(_one)(softmax_p_batch, proportions_batch)  # [B]
    except Exception:
        loss_list = []
        for b in range(B):
            loss_list.append(
                compute_CC_loss_fft_precise(
                    softmax_p_batch[b],       # [s,C]
                    proportions_batch[b],     # [C]
                    eps=eps
                )
            )
        loss_b = torch.stack(loss_list, dim=0).to(dev)  # [B]

    if reduce is None:
        return loss_b.to(softmax_p_batch.dtype)
    elif reduce == "mean":
        return loss_b.mean().to(softmax_p_batch.dtype)
    elif reduce == "sum":
        return loss_b.sum().to(softmax_p_batch.dtype)
    else:
        raise ValueError("reduce must be None|'mean'|'sum'")


def compute_CC_loss_dp_precise(
    softmax_p: torch.Tensor,     # [s, C]
    proportions: torch.Tensor,   # [C]
    eps: float = 1e-12
) -> torch.Tensor:
    """
    DP version (drop-in replacement): Compute CC loss via Poisson-binomial DP.

    softmax_p: [s, C]
    proportions: [C]
    Returns scalar loss.

    Notes:
      - Uses float64 internally.
      - Per-step global normalisation (max) per class C, accumulate log_scale.
      - Final logsumexp normalisation to ensure coefficients sum to 1 (in log domain).
    """
    assert softmax_p.dim() == 2, f"softmax_p should be [s,C], got {softmax_p.shape}"
    assert proportions.dim() == 1, f"proportions should be [C], got {proportions.shape}"

    s, C = softmax_p.shape
    dev = softmax_p.device
    dt = torch.float64

    # same tiny clamp spirit as your FFT version
    epsilon = 1e-300

    # [C, s]
    p = softmax_p.to(dt).t().clamp(min=epsilon, max=1.0 - epsilon)

    # target counts per class
    k_c = (proportions.to(dt) * s).round().clamp(0, s).long()  # [C]

    # DP coefficients for each class c: dp[c, k] = P(K=k) (scaled)
    # start with polynomial "1"
    dp = torch.zeros((C, s + 1), device=dev, dtype=dt)
    dp[:, 0] = 1.0

    log_scale_tot = torch.zeros((C,), device=dev, dtype=dt)
    zero_col = torch.zeros((C, 1), device=dev, dtype=dt)

    # Poisson-binomial DP:
    # dp_new = dp*(1-p) + shift(dp)*p
    for t in range(s):
        pt = p[:, t].unsqueeze(1)                 # [C,1]
        dp_shift = torch.cat((zero_col, dp[:, :-1]), dim=1)  # [C, s+1]
        dp = dp * (1.0 - pt) + dp_shift * pt

        # one-pass global normalisation per class (like your FFT one-pass)
        max_val = dp.abs().amax(dim=1, keepdim=True).clamp_min(epsilon)  # [C,1]
        dp = dp / max_val
        log_scale_tot = log_scale_tot + max_val.squeeze(1).log()

    # log-domain final normalisation (same pattern as your FFT version)
    dp_pos = dp.clamp_min(0.0) + eps                         # [C, s+1]
    log_coeffs = torch.log(dp_pos) + log_scale_tot.unsqueeze(1)  # [C, s+1]
    logZ = torch.logsumexp(log_coeffs, dim=1, keepdim=True)      # [C,1]
    log_coeffs_norm = log_coeffs - logZ

    idx = torch.arange(C, device=dev)
    log_a_k = log_coeffs_norm[idx, k_c]   # [C]
    loss = -log_a_k.sum()

    return loss.to(softmax_p.dtype)

import torch
from typing import Optional

# ============================================================
# 0) proportions -> integer counts (do NOT renormalize)
# ============================================================
def proportions_to_counts_exact(
    proportions: torch.Tensor,  # [C], assumed correct
    s: int,
) -> torch.Tensor:
    """
    Convert proportions to integer counts k with minimal disturbance.
    Does NOT renormalize proportions.
    Fixes tiny float rounding so sum(k) == s.
    """
    # proportions: keep on same device
    k = torch.round(proportions * s).to(torch.long)  # [C]
    diff = int(s - int(k.sum().item()))
    if diff != 0:
        # put mismatch to the largest-proportion class (minimal disturbance)
        j = int(torch.argmax(proportions).item())
        k[j] += diff
    k = torch.clamp(k, 0, s)
    return k


# ============================================================
# 1) FFT convolution (IMPORTANT: detach the stabilization scale)
# ============================================================
def batch_fft_convolve_one_pass(
    A: torch.Tensor,  # [C, n_pair, L]
    B: torch.Tensor,  # [C, n_pair, L]
    eps: float = 1e-300,
):
    """
    Return:
      conv_scaled : [C, n_pair, L_a+L_b-1]
      log_scale   : [C]  (log of stabilization scale per C)
    """
    next_len = A.size(-1) + B.size(-1) - 1
    N = 1 << ((next_len - 1).bit_length())

    fftA = torch.fft.rfft(A, n=N, dim=-1)
    fftB = torch.fft.rfft(B, n=N, dim=-1)
    conv_full = torch.fft.irfft(fftA * fftB, n=N, dim=-1)
    conv = conv_full[..., :next_len]  # [C, n_pair, next_len]

    # numerical stabilization ONLY: detach so it won't kill gradients via amax
    max_val = conv.detach().abs().amax(dim=(1, 2), keepdim=True).clamp_min(eps)  # [C,1,1]
    conv_scaled = conv / max_val
    log_scale = max_val.squeeze(-1).squeeze(-1).log()  # [C]

    return conv_scaled.contiguous(), log_scale


# ============================================================
# 2) Multiply (1-p + p z) polynomials per class via FFT merges
# ============================================================
def coeff_product_fast_norm(p_c_batch: torch.Tensor):
    """
    p_c_batch : [C, s]  (float64 recommended)
    Return:
      coeffs_scaled : [C, s+1]
      log_scale_tot : [C]
    """
    C, s = p_c_batch.shape
    polys = torch.stack((1.0 - p_c_batch, p_c_batch), dim=2)  # [C, s, 2]
    cur_len = 2
    log_scale_tot = torch.zeros(C, 1, device=p_c_batch.device, dtype=p_c_batch.dtype)  # [C,1]

    while polys.size(1) > 1:
        if polys.size(1) & 1:
            pad = polys.new_zeros(C, 1, cur_len)
            pad[..., 0] = 1.0
            polys = torch.cat((polys, pad), dim=1)

        polys = polys.view(C, -1, 2, cur_len)  # [C, n_pair, 2, cur_len]
        A = polys[:, :, 0]  # [C, n_pair, cur_len]
        B = polys[:, :, 1]

        conv, log_s = batch_fft_convolve_one_pass(A, B)  # conv: [C, n_pair, next_len]
        log_scale_tot = log_scale_tot + log_s.unsqueeze(1)

        cur_len = conv.size(-1)
        polys = conv.view(C, -1, cur_len)  # [C, n_poly, cur_len]

    coeffs = polys.squeeze(1)             # [C, s+1]
    log_scale_tot = log_scale_tot.squeeze(1)  # [C]
    return coeffs, log_scale_tot


# ============================================================
# 3) Single-bag CC loss (IMPORTANT: no "+1e-12" smoothing)
# ============================================================
def compute_CC_loss_fft_precise(
        args,
    softmax_p: torch.Tensor,      # [s, C] probabilities
    proportions: torch.Tensor,    # [C] assumed correct
) -> torch.Tensor:
    """
    Returns scalar:
      loss = - sum_c log P(K_c = k_c)
    where K_c is Poisson-binomial count from Bernoulli probs softmax_p[:,c].
    """
    assert softmax_p.dim() == 2, f"softmax_p expected [s,C], got {softmax_p.shape}"
    assert proportions.dim() == 1, f"proportions expected [C], got {proportions.shape}"

    s, C = softmax_p.shape
    dev = softmax_p.device
    dt = torch.float64
    tiny = torch.finfo(torch.float64).tiny  # ~2e-308
    # [C, s]

    p = softmax_p.to(dt).t().clamp(min=tiny, max=1.0 - tiny)

    # integer targets (do not renorm proportions)
    k_c = proportions_to_counts_exact(proportions.to(dev), s)  # [C]

    # coefficients
    coeffs, log_sc = coeff_product_fast_norm(p)  # [C, s+1], [C]

    # IMPORTANT: do NOT add 1e-12 here (kills gradients for s=128)
    coeffs_pos = coeffs.clamp_min(args.eps)  # only fix tiny negatives from FFT noise

    log_coeffs = torch.log(coeffs_pos) + log_sc.unsqueeze(1)  # [C, s+1]

    # normalize per class in log-spac
    logZ = torch.logsumexp(log_coeffs, dim=1, keepdim=True)    # [C,1]
    log_coeffs_norm = log_coeffs - logZ

    idx = torch.arange(C, device=dev)
    log_a_k = log_coeffs_norm[idx, k_c]  # prob >= ~1e-35

    loss = -log_a_k.sum()

    return loss.to(softmax_p.dtype)


# ============================================================
# 4) Batched wrapper
# ============================================================
def compute_CC_loss_fft_precise_batched(
    args,
    softmax_p_batch: torch.Tensor,     # [B, s, C]
    proportions_batch: torch.Tensor,   # [B, C]
    reduce: Optional[str] = "mean",    # "mean" recommended
) -> torch.Tensor:
    assert softmax_p_batch.dim() == 3, f"softmax_p_batch expected [B,s,C], got {softmax_p_batch.shape}"
    assert proportions_batch.dim() == 2, f"proportions_batch expected [B,C], got {proportions_batch.shape}"
    B, s, C = softmax_p_batch.shape
    assert proportions_batch.shape == (B, C)

    dev = softmax_p_batch.device

    try:
        from torch.func import vmap
        def _one(softmax_p, proportions):
            return compute_CC_loss_fft_precise(softmax_p, proportions)
        loss_b = vmap(_one)(softmax_p_batch, proportions_batch)  # [B]
    except Exception:
        loss_list = []
        for b in range(B):
            loss_list.append(compute_CC_loss_fft_precise(
                softmax_p_batch[b],
                proportions_batch[b],
            ))
        loss_b = torch.stack(loss_list, dim=0).to(dev)

    if reduce is None:
        return loss_b.to(softmax_p_batch.dtype)
    if reduce == "mean":
        return loss_b.mean().to(softmax_p_batch.dtype)
    if reduce == "sum":
        return loss_b.sum().to(softmax_p_batch.dtype)
    raise ValueError("reduce must be None | 'mean' | 'sum'")

from typing import Union, Sequence

def compute_CC_loss_fft_precise_batched_varlen(
    softmax_p_padded: torch.Tensor,      # [B, s_max, C]，按最长 bag 做了 padding
    proportions_batch: torch.Tensor,     # [B, C]
    bag_sizes: Union[torch.Tensor, Sequence[int]],  # [B]，每个 bag 的真实长度 s_b
    reduce: str = "sum",                 # None | "mean" | "sum"
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    支持变长 bagsize 的 batched 版本：
      - 每个样本的 softmax_p 用自己的 s_b 截取，
      - reduce="mean" 时按 bagsize 加权平均。
    """
    assert softmax_p_padded.dim() == 3, \
        f"softmax_p_padded should be [B, s_max, C], got {softmax_p_padded.shape}"
    assert proportions_batch.dim() == 2, \
        f"proportions_batch should be [B, C], got {proportions_batch.shape}"

    B, s_max, C = softmax_p_padded.shape
    assert proportions_batch.shape == (B, C), \
        f"proportions shape {proportions_batch.shape} vs (B,C)=({B},{C})"

    dev = softmax_p_padded.device
    dtype = softmax_p_padded.dtype

    # 让 bag_sizes 变成 tensor，放到同一个 device 上
    if not isinstance(bag_sizes, torch.Tensor):
        bag_sizes = torch.tensor(bag_sizes, device=dev, dtype=torch.long)
    else:
        bag_sizes = bag_sizes.to(device=dev, dtype=torch.long)

    assert bag_sizes.shape == (B,), f"bag_sizes shape {bag_sizes.shape} vs (B,)={B}"

    loss_list = []
    weight_list = []

    for b in range(B):
        s_b = int(bag_sizes[b].item())
        assert 1 <= s_b <= s_max, f"bag_sizes[{b}]={s_b} out of range (1..{s_max})"

        # 只取真实长度部分 [s_b, C]
        softmax_p_b = softmax_p_padded[b, :s_b, :]   # [s_b, C]
        proportions_b = proportions_batch[b]         # [C]

        loss_b = compute_CC_loss_fft_precise(
            softmax_p_b,
            proportions_b,
            eps=eps,
        )  # scalar
        loss_list.append(loss_b)
        weight_list.append(s_b)      # 用 bag 的长度当权重

    loss_b = torch.stack(loss_list, dim=0)          # [B]
    weights = torch.as_tensor(weight_list, device=dev, dtype=loss_b.dtype)  # [B]

    if reduce is None:
        return loss_b.to(dtype)

    elif reduce == "mean":
        # 按 bagsize 加权平均：相当于总 loss / 总实例数
        weighted_mean = (loss_b * weights).sum() / weights.sum()
        return weighted_mean.to(dtype)

    elif reduce == "sum":
        # 这里通常就直接 sum 各 bag 的 loss
        return loss_b.sum().to(dtype)

    else:
        raise ValueError("reduce must be None|'mean'|'sum'")


def thre_ema (thre ,sum_values ,ema ):
    return thre *ema +(1 -ema )*sum_values


def weight_decay_with_mask (mask ,initial_weight ,max_mask_count ):
    mask_count =mask .sum ().item ()
    weight_decay =max (0 ,1 -mask_count /max_mask_count )
    return initial_weight *weight_decay

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
    n_correct_u_lbs_meter =AverageMeter ()
    n_strong_aug_meter =AverageMeter ()
    mask_meter =AverageMeter ()
    pos_meter =AverageMeter ()
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
        if args .dataset in ["MNIST","FashionMNIST","KMNIST","EMNISTBalanced"]:
            ims_u_weak =ims_u_weak .permute (0 ,2 ,1 ,3 )

        bt =0
        imgs =torch .cat ([ims_u_weak ],dim =0 ).cuda ()
        logits =model (imgs )

        # logits_x = logits[:bt]
        logits_u_w =torch .split (logits [0 :],btu )
        logits_u_w =logits_u_w [0 ]

        probs = torch.sigmoid(logits_u_w)
        #probs= torch.softmax(logits_u_w,dim=1)
        #logits_clip = logits.clamp(-15, 15)
        #probs = torch.sigmoid(logits_clip)

        # loss_x = criteria_x(logits_x, lbs_x)
        N ,C =probs .shape
        s =bagsize
        assert N %s ==0 ,f"N={N}  bagsize={s} "
        B =N //s

        labels_p_batch =probs .contiguous ().view (B ,s ,C )

        bag_preds = labels_p_batch.mean(dim=1)  # [B, C]

        loss_prop =torch .Tensor ([]).cuda ()
        loss_prop =loss_prop .double ()
        kl_divergence =torch .Tensor ([]).cuda ()
        kl_divergence =kl_divergence .double ()
        kl_divergence_hard =torch .Tensor ([]).cuda ()
        kl_divergence_hard =kl_divergence_hard .double ()

        proportion =torch .stack ([torch .stack (lp [0 ])for lp in label_proportions ]).cuda ()
        proportion =proportion .view (length ,n_classes ,1 )
        proportion =proportion .squeeze (-1 )
        proportion =proportion .double ()
        loss =compute_CC_loss_fft_precise_batched (args,labels_p_batch ,proportion ,reduce ="mean")

        #loss =compute_CC_loss_fft_precise_batched_varlen (labels_p_batch ,proportion ,reduce ="mean")
        #loss =compute_CC_loss_dp_precise_batched (labels_p_batch ,proportion ,reduce ="mean")

        x =1.2

        kl_divergence =x
        kl_divergence_hard =x
        with torch .no_grad ():

            probs =torch .softmax (logits_u_w ,dim =1 )


            scores ,lbs_u_guess =torch .max (probs ,dim =1 )
            mask =scores .ge (args .thr ).float ()


            probs1 =torch .softmax (logits_u_w ,dim =1 ).detach()
            entropy = -(probs1 * probs1.log()).sum(dim=-1)  # shape: [batch_size]
        # (Optional) average entropy over the batch
        x =1
        mean_entropy =x

        mean_entropy = x
        # (Optional) average entropy over the batch
        mean_entropy = entropy.mean()


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
            "LR: {:.3f}. Time: {:.2f}. Entropy: {:.2f}".format (
            args .dataset ,args .n_labeled ,args .seed ,args .exp_dir ,epoch ,it +1 ,loss_prop_meter .avg ,kl_meter .avg ,
            kl_hard_meter .avg ,lr_log ,t ,entropy_meter .avg ))

            epoch_start =time .time ()
            bagsize =getattr (args ,"bagsize",getattr (args ,"bag_size",None ))
            if bagsize is None :
                raise ValueError (" args  bagsize  bag_size")

            csv_name =f"{args.dataset}_{args.exp_dir}_{bagsize}.csv"

            os .makedirs (args .exp_dir ,exist_ok =True )
            csv_path =os .path .join (args .exp_dir ,csv_name )

            if not os .path .exists (csv_path ):
                with open (csv_path ,"w",newline ="")as f :
                    writer =csv .writer (f )
                    writer .writerow (["epoch","time_sec"])

            with open (csv_path ,"a",newline ="")as f :
                csv .writer (f ).writerow ([epoch ,round (t ,2 )])
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
            if dataset in ["MNIST", "FashionMNIST", "KMNIST","EMNISTBalanced"]:
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
    parser .add_argument ('--n-epoches',type =int ,default =1024 ,
    help ='number of training epoches')
    parser .add_argument ('--batchsize',type =int ,default =4 ,
    help ='train batch size of bag samples')
    parser .add_argument ('--bagsize',type =int ,default = 256 ,
    help ='train bag size of samples')
    parser .add_argument ('--n-imgs-per-epoch',type =int ,default =1024 ,
    help ='number of training images for each epoch')

    parser .add_argument ('--eval-ema',default =False ,help ='whether to use ema model for evaluation')
    parser .add_argument ('--ema-m',type =float ,default =0.999 )

    parser .add_argument ('--lam-u',type =float ,default =1. ,
    help ='c oefficient of unlabeled loss')
    parser .add_argument ('--lr',type =float ,default= 2.5e-3 ,
    help ='learning rate for training')
    parser.add_argument('--eps', type=float, default=1e-30,
                        help='numerical stability epsilon')
    parser .add_argument ('--momentum',type =float ,default =0.9 ,
    help ='momentum for optimizer')
    parser .add_argument ('--seed',type =int ,default =13 ,
    help ='seed for random behaviors, no seed if negtive')
    parser.add_argument(
        '--pl_method',
        type=str,
        default='random',
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
    parser .add_argument ('--exp-dir',default ='CC',type =str ,help ='experiment id')
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
    logger .info ("  Task = {}@{}".format (args .dataset ,args .n_labeled ))

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
    dltrain_u, dataset_length, input_dim = get_train_loader(args.n_classes,
                                                            args.dataset, args.batchsize, args.bagsize, root=args.root,
                                                            method='DLLP',
                                                            supervised=False)
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
    optim = torch.optim.SGD(
        param_list,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=True
    )

    # 4% iter warmup: 2e-4 -> 1e-3 (args.lr)
    warmup_iter = int(0.08 * n_iters_all)
    warmup_ratio = (5e-5 / args.lr)

    lr_schdlr = WarmupCosineLrScheduler(
        optim,
        n_iters_all,
        warmup_iter=warmup_iter,
        warmup_ratio=warmup_ratio,
        warmup="linear",
    )
    #lr_schdlr =WarmupCosineLrScheduler (optim ,n_iters_all ,warmup_iter =0 )

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

        top1, ema_top1, top5, ema_top5, loss_test, prec, rec, f1   ,entropy = evaluate(model, ema_model, dlval, args.dataset)
        tb_logger.log_value('loss_prob', loss_prob, epoch)
        if (n_strong_aug == 0):
            tb_logger.log_value('guess_label_acc', 0, epoch)
        else:
            tb_logger.log_value('guess_label_acc', n_correct_u_lbs / n_strong_aug, epoch)
        tb_logger.log_value('test_acc', top1, epoch)
        tb_logger.log_value('mask', mask_mean, epoch)
        tb_logger.log_value('test_precision', prec, epoch)
        tb_logger.log_value('test_recall', rec, epoch)
        tb_logger.log_value('test_f1', f1, epoch)
        tb_logger.log_value('test_entropy', entropy, epoch)

        tb_logger.log_value('loss_test', loss_test, epoch)
        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch
        if best_acc_5 < top5:
            best_acc_5 = top5
            best_epoch_5 = epoch
        logger.info(
            "Epoch {}.loss_test: {:.4f}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f} in epoch{},"
            "Acc_5: {:.4f}.  best_acc_5: {:.4f} in epoch{},"
            " Precision: {:.4f}. Recall: {:.4f}.  F1: {:.4f}. Entropy: {:.4f}.".
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