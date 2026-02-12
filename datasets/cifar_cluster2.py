import os.path as osp
import pickle
import numpy as np
import scipy.io as sio

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter, OrderedDict
import h5py
from MLclf import MLclf
from collections import defaultdict

from datasets import tran as T
from datasets.rand import RandomAugment
from datasets.sampler import RandomSampler, BatchSampler

from datasets import transform as T1
from datasets.randaugment_grey import RandomAugment as RandomAugment1

import pickle
import os
from PIL import Image

label_map = {}
class_mapping={}


def extract_labels_from_class_dict(class_dict):
    for class_idx, image_indices in enumerate(class_dict.values()):
        for image_index in image_indices:
            label_map[image_index] = class_idx
    sorted_label_map = dict(sorted(label_map.items()))
    labels = list(sorted_label_map.values())
    return labels


def load_mini_imagenet_data(dspth, split='train'):

    if split == 'train':
        pkl_file = osp.join(dspth, 'miniimagenet/mini-imagenet-cache-train.pkl')
    elif split == 'val':
        pkl_file = osp.join(dspth, 'miniimagenet/mini-imagenet-cache-val.pkl')
    elif split == 'test':
        pkl_file = osp.join(dspth, 'miniimagenet/mini-imagenet-cache-test.pkl')
    else:
        raise ValueError("无效的 split 参数，应为 'train', 'val' 或 'test'")

    with open(pkl_file, 'rb') as f:
        data_dict = pickle.load(f)

    data = data_dict['image_data']
    labels = data_dict['class_dict']
    return data, labels

import numpy as np

def merge_train_val_test(dspth):
    """
    把 Mini-ImageNet 的 train/val/test 合并；
    支持 labels 为：
      1) dict: {class_name: [sample_indices_in_split, ...]}
      2) list/array: [class_name_or_id_per_sample, ...]
    返回：
      merged_data: 按 train→val→test 拼接后的数据
      merged_labels: 与 merged_data 一一对齐的稳定数值标签（相同输入→相同输出）
    """
    train_data, train_labels = load_mini_imagenet_data(dspth, split='train')
    val_data,   val_labels   = load_mini_imagenet_data(dspth, split='val')
    test_data,  test_labels  = load_mini_imagenet_data(dspth, split='test')

    # ---------- 1) 稳定的类别到ID映射 ----------
    def collect_classes(lbl):
        if isinstance(lbl, dict):
            return {str(k) for k in lbl.keys()}
        else:
            return {str(x) for x in (lbl.tolist() if hasattr(lbl, "tolist") else list(lbl))}
    all_classes = sorted(collect_classes(train_labels) |
                         collect_classes(val_labels)   |
                         collect_classes(test_labels))     # 排序=稳定
    cls2id = {c: i for i, c in enumerate(all_classes)}     # 稳定映射

    # ---------- 2) 把每个 split 还原成“样本级标签数组” ----------
    def to_per_sample_labels(data_split, labels_split):
        n = len(data_split)
        if isinstance(labels_split, dict):
            y = np.empty(n, dtype=np.int64)
            # 安全性：索引必须在 [0, n)
            for c, idxs in labels_split.items():
                idxs = np.asarray(idxs, dtype=np.int64)
                assert idxs.ndim == 1, f"indices dim error for class {c}"
                assert idxs.size > 0, f"empty indices for class {c}"
                assert idxs.min() >= 0 and idxs.max() < n, \
                    f"indices out of range for split: max={idxs.max()} n={n}"
                y[idxs] = cls2id[str(c)]
            return y
        else:
            # 已是一一对齐的标签列表/数组
            arr = labels_split.tolist() if hasattr(labels_split, "tolist") else list(labels_split)
            assert len(arr) == n, f"labels length {len(arr)} != data length {n}"
            return np.fromiter((cls2id[str(c)] for c in arr), dtype=np.int64, count=n)

    y_train = to_per_sample_labels(train_data, train_labels)
    y_val   = to_per_sample_labels(val_data,   val_labels)
    y_test  = to_per_sample_labels(test_data,  test_labels)

    # ---------- 3) 按相同顺序拼接数据与标签 ----------
    merged_data   = np.concatenate([train_data, val_data, test_data], axis=0)
    merged_labels = np.concatenate([y_train,    y_val,   y_test],     axis=0)

    return merged_data, merged_labels

def load_tiny_imagenet_val(root, image_size=(64, 64)):
    datalist = []
    labels = []
    n_class = 0

    # 读取标签文件
    with open(os.path.join(root, 'tiny-imagenet-200/val', 'val_annotations.txt'), 'r') as f:
        for line in f:
            parts = line.split('\t')
            image_name = parts[0]
            class_name = parts[1]
            bbox = list(map(int, parts[2:]))

            if class_name not in label_map:
                label_map[class_name] = n_class
                n_class += 1

            # 读取图像
            image_path = os.path.join(root, 'tiny-imagenet-200/val', 'images', image_name)
            image = Image.open(image_path)
            image = image.resize(image_size)
            image = np.array(image)

            # 确保图像具有3个通道
            if len(image.shape) != 3 or image.shape[2] != 3:
                continue

            # 添加到数据列表和标签列表
            datalist.append(image)
            labels.append(label_map[class_name])

    return np.array(datalist), np.array(labels), n_class
def load_tiny_imagenet_data(root, image_size=(64, 64)):
    datalist = []
    labels = []
    n_class = 0

    # Loop through each class folder
    for class_folder in os.listdir(os.path.join(root, 'tiny-imagenet-200/train')):
        class_folder_path = os.path.join(root, 'tiny-imagenet-200/train', class_folder)
        if os.path.isdir(class_folder_path):
            label_map[class_folder] = n_class
            n_class += 1
            for image_file in os.listdir(os.path.join(class_folder_path, 'images')):
                image_path = os.path.join(class_folder_path, 'images', image_file)
                # Load and resize image
                image = Image.open(image_path)
                image = image.resize(image_size)
                # Convert to numpy array
                image = np.array(image)
                # Ensure image has 3 channels
                if len(image.shape) != 3 or image.shape[2] != 3:
                    continue
                # Append to data list and label list
                datalist.append(image)
                labels.append(label_map[class_folder])
    labels = np.array(labels)

    return np.array(datalist), labels, n_class
def load_test_data(test_data, test_labels, class_mapping):
    # 创建一个新的标签列表，用于存储测试数据的标签
    final_test_labels = [None] * sum(len(v) for v in test_labels.values())  # 初始化labels列表
    test_data_list = []  # 用于存储测试数据

    # 使用和训练集相同的 class_label 映射
    for key, indices in test_labels.items():
        # 获取训练集中相同类的标签索引
        if key in class_mapping:
            class_label = class_mapping[key]  # 获取对应的类别标签
        else:
            raise ValueError(f"测试数据集中找不到训练数据中的类: {key}")

        # 将测试数据中的索引与类别标签进行映射
        for index in indices:
            final_test_labels[index] = class_label  # 为每个索引赋值类别标签
            test_data_list.append(test_data[index])  # 加载测试数据

    # 将标签转换为 numpy 数组
    final_test_labels = np.array(final_test_labels)
    return np.array(test_data_list), final_test_labels

class OneCropsTransform:

    def __init__(self,trans_weak):
        self.trans_weak = trans_weak

    def __call__(self,x):
        x1=self.trans_weak(x)
        return [x1]

class TwoCropsTransform:
    """Take 2 random augmentations of one image."""

    def __init__(self, trans_weak, trans_strong):
        self.trans_weak = trans_weak
        self.trans_strong = trans_strong

    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong(x)
        return [x1, x2]


class ThreeCropsTransform:
    """Take 3 random augmentations of one image."""

    def __init__(self, trans_weak, trans_strong0, trans_strong1):
        self.trans_weak = trans_weak
        self.trans_strong0 = trans_strong0
        self.trans_strong1 = trans_strong1

    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong0(x)
        x3 = self.trans_strong1(x)

        return [x1, x2, x3]




def load_data_train(num_classes, dataset='CIFAR10', dspth='./data', bagsize=16):
    """
    不改变接口与返回值：
      return data_u, label_prob, labels_real, labels_idx, dataset_length, indices_u, input_dim

    现在只用 1 个参数（Dirichlet 总浓度 alpha0）控制“主导程度”：
      - alpha0 越小（<1）：越容易出现某一个簇权重很大 -> bag 更“被一个簇主导”
      - alpha0 越大（>1）：越均匀 -> bag 更混

    严格全局无放回：
      - 同一个样本不会出现在不同 bag（也不会在同一个 bag 里重复）
    """
    import os.path as osp
    import numpy as np
    import pickle
    from collections import Counter, OrderedDict

    input_dim = 1

    # -----------------------------
    # 1) 读取原始数据（保持你原来的风格）
    # -----------------------------
    if dataset == 'Corel16k':
        datalist = [osp.join(dspth, f'Corel16k{idx:03d}-train.arff') for idx in range(1, 11)]
        n_class = 374
    elif dataset == 'Corel5k':
        datalist = [osp.join(dspth, 'Corel5k-train.arff')]
        n_class = 374
    elif dataset == 'Delicious':
        datalist = [osp.join(dspth, 'delicious-train.arff')]
        n_class = 983
    elif dataset == 'Bookmarks':
        datalist = [osp.join(dspth, 'bookmarks.arff')]
        n_class = 208
    elif dataset == 'Eurlex_DC':
        datalist = [osp.join(dspth, f'eurlex-dc-leaves-fold{i}-train.arff') for i in range(1, 11)]
        n_class = 201
    elif dataset == 'Eurlex_SM':
        datalist = [osp.join(dspth, f'eurlex-sm-fold{i}-train.arff') for i in range(1, 11)]
        n_class = 281
    elif dataset == 'Scene':
        datalist = [osp.join(dspth, 'scene-train.arff')]
        n_class = 6
    elif dataset == 'Yeast':
        datalist = [osp.join(dspth, 'yeast-train.arff')]
        n_class = 14
    elif dataset == 'CIFAR10':
        datalist = [osp.join(dspth, 'cifar-10-batches-py', f'data_batch_{i+1}') for i in range(5)]
        n_class = 10
    elif dataset == 'CIFAR100':
        datalist = [osp.join(dspth, 'cifar-100-python', 'train')]
        n_class = 100
    elif dataset == 'SVHN':
        data, labels= load_svhn_data(dspth)
    elif dataset == 'MNIST':
        data, labels = [], []
        datalist = [osp.join(dspth, 'MNIST', 'raw', 'train-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'MNIST', 'raw', 'train-labels-idx1-ubyte')]
        n_class = num_classes
    elif dataset == 'FashionMNIST':
        data, labels = [], []
        datalist = [osp.join(dspth, 'FashionMNIST', 'raw', 'train-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'FashionMNIST', 'raw', 'train-labels-idx1-ubyte')]
        n_class = 10
    elif dataset == 'KMNIST':
        data, labels = [], []
        datalist = [osp.join(dspth, 'KMNIST', 'raw', 'train-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'KMNIST', 'raw', 'train-labels-idx1-ubyte')]
        n_class = 10
    elif dataset == 'EMNISTBalanced':
        data, labels = [], []
        datalist = [osp.join(dspth, 'EMNIST', 'raw', 'emnist-balanced-train-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'EMNIST', 'raw', 'emnist-balanced-train-labels-idx1-ubyte')]
        n_class = 47
    elif dataset in ['AGNEWS', 'TinyImageNet', 'miniImageNet']:
        raise ValueError(f"{dataset} loader not provided in this snippet.")
    else:
        raise ValueError("Unsupported dataset")

    # ---------- 读取具体数据 ----------
    if dataset in ['CIFAR10', 'CIFAR100']:
        data, labels = [], []
        for data_batch in datalist:
            with open(data_batch, 'rb') as fr:
                entry = pickle.load(fr, encoding='latin1')
                lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
                data.append(entry['data'])
                labels.append(lbs)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

        if n_class == 2:
            machine_classes = np.array([0, 1, 8, 9])
            labels = np.isin(labels, machine_classes).astype(np.int64)

    elif dataset in ['MNIST']:
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)
                fr_label.read(8)
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        if n_class == 2:
            labels = np.where(np.isin(labels, [0, 2, 4, 6, 8]), 0, 1)

    elif dataset in ['FashionMNIST', 'KMNIST', 'EMNISTBalanced']:
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)
                fr_label.read(8)
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

    elif dataset in ['Corel16k','Corel5k','Delicious','Bookmarks','Eurlex_DC','Eurlex_SM','Scene','Yeast']:
        import pandas as pd
        from scipy.io import arff

        data_parts, label_parts = [], []
        for arff_path in datalist:
            raw, _ = arff.loadarff(arff_path)
            df = pd.DataFrame(raw)

            X_part = df.iloc[:, :-n_class].values.astype(np.float32)
            Y_part = (df.iloc[:, -n_class:]
                      .applymap(lambda x: int(x.decode() if isinstance(x, bytes) else x))
                      .values.astype(np.int32))

            data_parts.append(X_part)
            label_parts.append(Y_part)

        data = np.concatenate(data_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0)

        if labels.shape[1] > 15:
            freq = labels.sum(axis=0)
            keep_idx = np.argsort(freq)[::-1][:15]
            remove_idx = np.setdiff1d(np.arange(labels.shape[1]), keep_idx)

            mask_samples = (labels[:, remove_idx].sum(axis=1) == 0)
            data = data[mask_samples]
            labels = labels[mask_samples]
            labels = labels[:, keep_idx]

        n_class = labels.shape[1]
        input_dim = data.shape[1]

    # -----------------------------
    # 2) 用聚类簇来生成 bag（严格无放回）
    # -----------------------------
    dataset_length = len(data)
    num_bags = dataset_length // bagsize
    data_length = num_bags * bagsize

    # 全长打乱再截断：保证参与造 bag 的样本是唯一的一批（不会重复）
    perm_all = np.arange(dataset_length)
    np.random.shuffle(perm_all)
    perm = perm_all[:data_length]

    data = data[perm]
    labels = labels[perm]

    # ---- 读取 cluster map（必须提前生成）----
    def _find_cluster_npz(_dspth, _dataset):
        base = osp.join(_dspth, "cluster_maps")
        if _dataset == "CIFAR100":
            candidates = [256, 128, 64]
        else:
            candidates = [32, 64]
        for K in candidates:
            p = osp.join(base, f"{_dataset}_train_K{K}.npz")
            if osp.exists(p):
                return p
        raise FileNotFoundError(
            f"Cannot find cluster map for {_dataset} under {base}. "
            f"Expected like {_dataset}_train_K32.npz / {_dataset}_train_K256.npz"
        )

    cluster_npz = _find_cluster_npz(dspth, dataset)
    z = np.load(cluster_npz, allow_pickle=True)
    clusters_all = z["clusters"].astype(np.int64)  # 原始顺序 cluster id
    clusters = clusters_all[perm]                  # 对齐 data/labels 的重排截断

    # ---- 建立每簇池：pool 是“可用样本索引列表”，取过就不会再用 ----
    rng = np.random.default_rng()
    pools = {}
    for i, c in enumerate(clusters):
        pools.setdefault(int(c), []).append(i)
    # 转 numpy 并 shuffle
    for c in list(pools.keys()):
        arr = np.array(pools[c], dtype=np.int64)
        rng.shuffle(arr)
        pools[c] = arr

    ptr = {c: 0 for c in pools.keys()}  # 每簇取样指针

    def remaining(c: int) -> int:
        return int(len(pools[c]) - ptr[c])

    def take(c: int, k: int) -> np.ndarray:
        """从簇 c 无放回取 k 个（若不够会自动取剩余全部）"""
        rem = remaining(c)
        k = min(k, rem)
        s = ptr[c]
        e = s + k
        ptr[c] = e
        return pools[c][s:e]

    cluster_ids_all = np.array(sorted(pools.keys()), dtype=np.int64)

    # -----------------------------
    # 只用一个参数：Dirichlet 总浓度 alpha0
    # -----------------------------
    alpha0 = 1
    # 建议范围：
    #   0.1 ~ 0.3: 强主导（一个簇占很多）
    #   0.5:      中等主导
    #   1.0:      接近均匀混合（主导不明显）

    def _pi_from_remaining(cluster_ids):
        """base measure π：按剩余量归一化（更稳，不容易抽到快空的簇）"""
        rems = np.array([remaining(int(c)) for c in cluster_ids], dtype=np.float64)
        rems = np.maximum(rems, 0.0)
        s = rems.sum()
        if s <= 0:
            return np.ones(len(cluster_ids), dtype=np.float64) / max(1, len(cluster_ids))
        return rems / s

    # -----------------------------
    # 3) 生成 bag：先全局抽 counts 矩阵 C，再严格无放回分配（不需要 deficit）
    # -----------------------------
    data_u, label_prob = [], []
    labels_real = []
    labels_idx = []
    indices_u = []

    # cluster id 列表（存在于截断数据中的簇）
    cluster_ids_all = np.array(sorted(pools.keys()), dtype=np.int64)
    K = len(cluster_ids_all)
    B = num_bags
    m = bagsize

    # 每个簇的供给（一定满足 sum = B*m，因为你已截断到 data_length=B*m）
    cluster_sizes = np.array([len(pools[int(c)]) for c in cluster_ids_all], dtype=np.int64)
    assert cluster_sizes.sum() == B * m, "Internal error: cluster sizes don't sum to data_length."

    # Dirichlet 总浓度（你原来写死 alpha0=1，这里保留）
    alpha0 = 1.0

    # base measure pi：按簇大小（“她那种方法”通常用固定 base，不用 remaining）
    pi = cluster_sizes.astype(np.float64)
    pi = pi / pi.sum()

    def _sample_counts_matrix_dirichlet(B, m, alpha0, pi, rng):
        """
        先为每个 bag 抽 w_b ~ Dir(alpha0*pi)
        再把 w_b 变成整数行和为 m 的 counts（先 floor，再按小数部分补齐）
        返回：C shape (B, K)，每行和=m
        """
        dir_param = np.maximum(alpha0 * pi, 1e-12)
        W = rng.dirichlet(dir_param, size=B)  # (B, K)

        # 先取 floor
        F = W * m
        C = np.floor(F).astype(np.int64)
        row_sum = C.sum(axis=1)
        rem = (m - row_sum).astype(np.int64)  # 每行还差多少个

        # 按小数部分做“补齐”（不改变 Dirichlet 风味太多）
        frac = F - np.floor(F)
        for b in range(B):
            r = int(rem[b])
            if r <= 0:
                continue
            p = frac[b].copy()
            s = p.sum()
            if s <= 0:
                # 万一全是整数，退化就按 W 补
                p = W[b].copy()
                p = p / p.sum()
            else:
                p = p / s
            add = rng.choice(np.arange(K), size=r, replace=True, p=p)
            # add 里可能有重复，直接累加
            for k in add:
                C[b, int(k)] += 1

        # 校验行和
        assert np.all(C.sum(axis=1) == m)
        return C

    def _balance_columns_to_targets(C, targets, rng):
        """
        把 C 的列和配平到 targets（严格等于），不改变行和。
        通过在某些 bag 内做 “从 surplus 列挪 1 个到 deficit 列” 来实现。
        """
        C = C.copy()
        cur = C.sum(axis=0)
        diff = targets - cur  # >0 缺，<0 多

        deficit = np.where(diff > 0)[0].tolist()
        surplus = np.where(diff < 0)[0].tolist()

        # 安全上限：最多需要移动 sum(diff>0) 次（每次移动 1）
        max_moves = int(diff[diff > 0].sum())
        moves = 0

        while deficit:
            d = deficit[-1]  # 取一个缺的列
            # 找一个多的列
            if not surplus:
                raise RuntimeError("Balancing failed: no surplus but still deficit.")
            s = surplus[-1]

            # 找一个 bag，使得该 bag 在 s 列有至少 1 个可以挪
            candidates = np.where(C[:, s] > 0)[0]
            if len(candidates) == 0:
                # 这个 surplus 列其实挪不动（理论上不该发生），换一个 surplus
                surplus.pop()
                continue
            b = int(rng.choice(candidates))

            # 在 bag b 内：从 s 挪 1 个到 d
            C[b, s] -= 1
            C[b, d] += 1

            diff[s] += 1
            diff[d] -= 1
            moves += 1
            if moves > max_moves + 10 * K:
                raise RuntimeError("Balancing seems stuck; check inputs.")

            # 更新 deficit/surplus 列表
            if diff[d] == 0:
                deficit.pop()
            if diff[s] == 0:
                surplus.pop()

        # 校验列和精确匹配 targets
        assert np.all(C.sum(axis=0) == targets)
        # 行和不变
        assert np.all(C.sum(axis=1) == m)
        return C

    # 1) 先抽一个“行和正确”的 counts 矩阵
    C0 = _sample_counts_matrix_dirichlet(B, m, alpha0, pi, rng)

    # 2) 再把列和配平到 cluster_sizes（这样就保证截断后的样本 100% 用完）
    C = _balance_columns_to_targets(C0, cluster_sizes, rng)

    # 3) 按 C 严格无放回分配（不需要 deficit）
    ptr = {int(c): 0 for c in cluster_ids_all}

    for j in range(B):
        bag_idx_list = []
        for k, cid in enumerate(cluster_ids_all):
            need = int(C[j, k])
            if need <= 0:
                continue
            cid = int(cid)
            s = ptr[cid]
            e = s + need
            chosen = pools[cid][s:e]  # 一定够，因为列和已经匹配供给
            ptr[cid] = e
            bag_idx_list.extend(chosen.tolist())

        bag_indices = np.array(bag_idx_list, dtype=np.int64)
        rng.shuffle(bag_indices)

        # --- 组装 bag_data（沿用你原来的分支）---
        if dataset in ['MNIST', 'FashionMNIST', 'EMNISTBalanced', 'KMNIST']:
            bag_data = [data[i].reshape(28, 28) for i in bag_indices]
        elif dataset == 'SVHN':
            bag_data = [data[i] for i in bag_indices]
        elif dataset == 'TinyImageNet':
            bag_data = [data[i] for i in bag_indices]
        elif dataset in ['Corel16k', 'Corel5k', 'Delicious', 'Bookmarks', 'Eurlex_DC', 'Eurlex_SM', 'Scene', 'Yeast']:
            bag_data = [data[i] for i in bag_indices]
        elif dataset == 'miniImageNet':
            bag_data = [data[i] for i in bag_indices]
        else:
            bag_data = [data[i].reshape(3, 32, 32).transpose(1, 2, 0) for i in bag_indices]

        bag_labels = np.array([labels[i] for i in bag_indices])
        labels_real.append(bag_labels)
        labels_idx.append(bag_indices)

        label_counts = Counter(bag_labels)
        label_counts = OrderedDict(sorted(label_counts.items()))
        label_proportions = [
            label_counts.get(label, 0) / len(bag_labels)
            for label in range(0, num_classes)
        ]

        data_u.append(bag_data)
        indices_u.append(j)
        label_prob.append(label_proportions)

    return data_u, label_prob, labels_real, labels_idx, dataset_length, indices_u, input_dim


def load_data_val(dataset, dspth='./data',n_classes=10):
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'test_batch')
        ]
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'test')
        ]
    elif dataset == 'Corel16k':
        datalist = [osp.join(dspth, f'Corel16k{idx:03d}-test.arff')
                    for idx in range(1, 11)]  # 一共有 10 个子集
    elif dataset == 'Corel5k':
        datalist = [osp.join(dspth, 'Corel5k-test.arff')]

    elif dataset == 'Delicious':
        datalist = [osp.join(dspth, 'delicious-test.arff')]

    elif dataset == 'Bookmarks':
        datalist = [osp.join(dspth, 'bookmarks.arff')]  # 原数据即 train+test，若你有拆分请自行替换

    elif dataset == 'Eurlex_DC':
        datalist = [osp.join(dspth,
                             f'eurlex-dc-leaves-fold{i}-test.arff') for i in range(1, 11)]

    elif dataset == 'Eurlex_SM':
        datalist = [osp.join(dspth,
                             f'eurlex-sm-fold{i}-test.arff') for i in range(1, 11)]

    elif dataset == 'Scene':
        datalist = [osp.join(dspth, 'scene-test.arff')]

    elif dataset == 'Yeast':
        datalist = [osp.join(dspth, 'yeast-test.arff')]
    elif dataset == 'SVHN':
        data, labels= load_svhn_val(dspth)
    elif dataset == "TinyImageNet":
        data, labels, n_class = load_tiny_imagenet_val(dspth)
    elif dataset == 'miniImageNet':
        # 加载miniImageNet数据集的训练、验证和测试集
        train_data, train_labels = merge_train_val_test(dspth)
        test_data_list = []
        test_labels_list = []
        n_class = 100
        for i in range(0, len(train_data), 600):
            # Get the last 100 samples from the current chunk
            chunk_data = train_data[i:i + 600][-100:]
            chunk_labels = np.array(train_labels[i:i + 600][-100:])

            # Append the data and labels to the lists
            test_data_list.append(chunk_data)
            test_labels_list.append(chunk_labels)

        # Concatenate the subsets into final arrays
        data = np.concatenate(test_data_list, axis=0)
        labels = np.concatenate(test_labels_list, axis=0)
        # 使用和训练集相同的 class_label 映射


    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        data, labels = [], []
        for data_batch in datalist:
            with open(data_batch, 'rb') as fr:
                entry = pickle.load(fr, encoding='latin1')
                lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
                data.append(entry['data'])
                labels.append(lbs)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [
            el.reshape(3, 32, 32).transpose(1, 2, 0)
            for el in data
        ]

        if n_classes == 2:
            # 机器类映射为 1，其余为 0
            machine_classes = np.array([0, 1, 8, 9])
            labels = np.isin(labels, machine_classes).astype(np.int64)
    elif dataset == 'MNIST':
        data, labels = [], []
        datalist = [osp.join(dspth, 'MNIST', 'raw', 't10k-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'MNIST', 'raw', 't10k-labels-idx1-ubyte')]
        n_class = 2
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        if n_class == 2:
            labels = np.where(np.isin(labels, [0, 2, 4, 6, 8]), 0, 1)
        data = [
            el.reshape(28, 28)
            for el in data
        ]
    elif dataset == 'FashionMNIST':
        data, labels = [], []
        datalist = [osp.join(dspth, 'FashionMNIST', 'raw', 't10k-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'FashionMNIST', 'raw', 't10k-labels-idx1-ubyte')]
        n_class = 10
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [
            el.reshape(28, 28)
            for el in data
        ]
    elif dataset == 'KMNIST':
        data, labels = [], []
        datalist = [osp.join(dspth, 'KMNIST', 'raw', 't10k-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'KMNIST', 'raw', 't10k-labels-idx1-ubyte')]
        n_class = 10
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [
            el.reshape(28, 28)
            for el in data
        ]

    elif dataset in [
        'Corel16k',  # Corel16kXXX-train/test.arff
        'Corel5k',
        'Delicious',
        'Bookmarks',
        'Eurlex_DC',  # eurlex-dc-leaves-fold*-train/test.arff
        'Eurlex_SM',  # eurlex-sm-fold*-train/test.arff
        'Scene',
        'Yeast'
    ]:
        import pandas as pd
        from scipy.io import arff

        # --------------------------------------------------
        # 0⃣  先给出 **train** / **test** 文件列表
        #     —— 与训练那段逻辑保持完全一致 ——
        # --------------------------------------------------
        if dataset == 'Corel16k':
            train_datalist = [osp.join(dspth, f'Corel16k{idx:03d}-train.arff')
                              for idx in range(1, 11)]
            test_datalist = [osp.join(dspth, f'Corel16k{idx:03d}-test.arff')
                             for idx in range(1, 11)]
            n_class = 374
        elif dataset == 'Corel5k':
            train_datalist = [osp.join(dspth, 'Corel5k-train.arff')]
            test_datalist = [osp.join(dspth, 'Corel5k-test.arff')]
            n_class = 374
        elif dataset == 'Delicious':
            train_datalist = [osp.join(dspth, 'delicious-train.arff')]
            test_datalist = [osp.join(dspth, 'delicious-test.arff')]
            n_class = 983
        elif dataset == 'Bookmarks':
            train_datalist = test_datalist = [osp.join(dspth, 'bookmarks.arff')]
            n_class = 208
        elif dataset == 'Eurlex_DC':
            train_datalist = [osp.join(dspth, f'eurlex-dc-leaves-fold{i}-train.arff')
                              for i in range(1, 11)]
            test_datalist = [osp.join(dspth, f'eurlex-dc-leaves-fold{i}-test.arff')
                             for i in range(1, 11)]
            n_class = 201
        elif dataset == 'Eurlex_SM':
            train_datalist = [osp.join(dspth, f'eurlex-sm-fold{i}-train.arff')
                              for i in range(1, 11)]
            test_datalist = [osp.join(dspth, f'eurlex-sm-fold{i}-test.arff')
                             for i in range(1, 11)]
            n_class = 281
        elif dataset == 'Scene':
            train_datalist = [osp.join(dspth, 'scene-train.arff')]
            test_datalist = [osp.join(dspth, 'scene-test.arff')]
            n_class = 6
        elif dataset == 'Yeast':
            train_datalist = [osp.join(dspth, 'yeast-train.arff')]
            test_datalist = [osp.join(dspth, 'yeast-test.arff')]
            n_class = 14

        # --------------------------------------------------
        # 1⃣  读取 **训练集**，统计出现频次 → keep_idx
        # --------------------------------------------------
        tr_X_parts, tr_Y_parts = [], []

        for p in train_datalist:
            raw, _ = arff.loadarff(p)
            df = pd.DataFrame(raw)

            X = df.iloc[:, :-n_class].values.astype(np.float32)
            Y = (df.iloc[:, -n_class:]
                 .applymap(lambda z: int(z.decode() if isinstance(z, bytes) else z))
                 .values.astype(np.int32))

            tr_X_parts.append(X)
            tr_Y_parts.append(Y)

        data = np.concatenate(tr_X_parts, axis=0)
        labels = np.concatenate(tr_Y_parts, axis=0)

        if labels.shape[1] > 15:  # 只有标签数 >15 时才裁剪
            freq = labels.sum(axis=0)  # 每列出现次数
            keep_idx = np.argsort(freq)[::-1][:15]  # 前 15 高频标签
            remove_idx = np.setdiff1d(np.arange(labels.shape[1]), keep_idx)

            mask_tr = (labels[:, remove_idx].sum(axis=1) == 0)
            data = data[mask_tr]
            labels = labels[mask_tr][:, keep_idx]
        else:
            keep_idx = np.arange(labels.shape[1])  # 全保留

        # -------- 更新维度信息 --------
        input_dim = data.shape[1]
        n_class = labels.shape[1]  # ≤15

        # 到这里：data / labels 即 **训练集**，已满足论文要求

        # --------------------------------------------------
        # 2⃣  读取 **测试集**，用同一个 keep_idx 同步裁剪
        # --------------------------------------------------
        te_X_parts, te_Y_parts = [], []
        rem_idx = np.setdiff1d(np.arange(n_class if len(keep_idx) == n_class else max(keep_idx) + 1), keep_idx)

        for p in test_datalist:
            raw, _ = arff.loadarff(p)
            df = pd.DataFrame(raw)

            X_full = df.iloc[:, :-n_class].values.astype(np.float32)
            Y_full = (df.iloc[:, -n_class:]
                      .applymap(lambda z: int(z.decode() if isinstance(z, bytes) else z))
                      .values.astype(np.int32))

            mask_te = (Y_full[:, rem_idx].sum(axis=1) == 0) if len(rem_idx) else np.ones(len(Y_full), bool)
            X_keep = X_full[mask_te]
            Y_keep = Y_full[mask_te][:, keep_idx]

            te_X_parts.append(X_keep)
            te_Y_parts.append(Y_keep)

        data = np.concatenate(te_X_parts, axis=0)
        labels = np.concatenate(te_Y_parts, axis=0)

    elif dataset == 'EMNISTBalanced':
        data, labels = [], []
        # 更新为EMNIST Balanced数据集的文件路径
        datalist = [osp.join(dspth, 'EMNIST', 'raw', 'emnist-balanced-test-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'EMNIST', 'raw', 'emnist-balanced-test-labels-idx1-ubyte')]
        n_class = 47  # EMNIST Balanced有47个类别

        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)  # 跳过头部信息
                fr_label.read(8)  # 跳过头部信息
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 28 * 28))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [el.reshape(28, 28) for el in data]  # 将每个样本重塑为28x28

    return data, labels

def load_svhn_val(dspth='./data/svhn'):
    svhn_path = osp.join(dspth, 'svhn')
    with open(osp.join(svhn_path, 'test_32x32.mat'), 'rb') as fr:
        svhn_data = sio.loadmat(fr)
        data = svhn_data['X']
        labels = svhn_data['y']
    data = np.transpose(data, (3, 0, 1, 2))

    labels = labels % 10
    labels = labels.squeeze()

    return data, labels
def load_svhn_data(dspth):
    svhn_path = osp.join(dspth, 'svhn')

    # 加载训练数据
    with open(osp.join(svhn_path, 'train_32x32.mat'), 'rb') as fr:
        svhn_train = sio.loadmat(fr)
        train_data = svhn_train['X']
        train_labels = svhn_train['y']


    # 转换数据维度
    train_data = np.transpose(train_data, (3, 0, 1, 2))

    # 调整标签（从1-10改为0-9）
    train_labels = (train_labels ) % 10

    # 压缩标签数组
    train_labels = train_labels.squeeze()

    # 合并训练数据和额外数据
    return train_data, train_labels



def compute_mean_var():
    data_x, label_x, data_u, label_u = load_data_train()
    data = data_x + data_u
    data = np.concatenate([el[None, ...] for el in data], axis=0)

    mean, var = [], []
    for i in range(3):
        channel = (data[:, :, :, i].ravel() / 127.5) - 1
        #  channel = (data[:, :, :, i].ravel() / 255)
        mean.append(np.mean(channel))
        var.append(np.std(channel))

    print('mean: ', mean)
    print('var: ', var)


class Cifar(Dataset):
    def __init__(self, dataset, data, labels, labels_real,labels_idx,indices_u, mode):
        super(Cifar, self).__init__()
        self.data, self.labels, self.labels_real,self.labels_idx,self.indices_u = data, labels, labels_real,labels_idx,indices_u
        self.mode = mode
        assert len(self.data) == len(self.labels)
        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == 'CIFAR100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        elif dataset == 'FashionMNIST':
            mean, std = (0.1307), (0.3081)
        elif dataset == 'EMNISTBalanced':
            mean, std = (0.1307), (0.3081)
        elif dataset == 'MNIST':
            mean, std = (0.1307), (0.3081)
        elif dataset == 'KMNIST':
            mean, std = (0.1307), (0.3081)
        elif dataset =='miniImageNet':
            mean, std=(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean = (0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225)
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            trans_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['FashionMNIST','KMNIST']:
            trans_weak = T1.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomHorizontalFlip(p=0.5),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomHorizontalFlip(p=0.5),
                RandomAugment1(2, 10),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['MNIST','EMNISTBalanced']:
            trans_weak = T.Compose([
                T1.Resize((28, 28)),
                #T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                #T1.RandomAffine(
                #    degrees=15,  # +/- 5度的旋转
                #    translate=(0.1, 0.1),  # 最多10%的水平和垂直平移
                #    scale_range=(0.9, 1.1)  # 0.9到1.1倍的缩放
                #),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomAffine(
                    degrees=15,  # +/- 5度的旋转
                    translate=(0.1, 0.1),  # 最多10%的水平和垂直平移
                    scale_range=(0.9, 1.1)  # 0.9到1.1倍的缩放
                ),
                RandomAugment1(2, 10),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
                transforms.RandomAffine(
                    degrees=15,  # +/- 5度的旋转
                    translate=(0.1, 0.1),  # 最多10%的水平和垂直平移
                    scale=(0.9, 1.1)  # 0.9到1.1倍的缩放
                ),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['TinyImageNet']:
            trans_weak = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['miniImageNet']:
            trans_weak = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in [
            'Corel16k', 'Corel5k', 'Delicious', 'Bookmarks',
            'Eurlex_DC', 'Eurlex_SM', 'Scene', 'Yeast'
        ]:
            # 这些数据集本身就是已提取好的扁平特征向量，
            # 在训练前不需要任何数据增强或归一化。
            trans_weak = T.Compose([
                T.ToTensor(),  # ← 只是把 ndarray → FloatTensor
            ])
            trans_strong0 = T.Compose([
                T.ToTensor(),  # 强弱同样：啥也不干
            ])
            trans_strong1 = T.Compose([
                T.ToTensor(),
            ])

        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_DLLP':
            self.trans = OneCropsTransform(trans_weak)
        elif self.mode == 'train_u_co':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_L^2P-AHIL':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            if dataset in ['MNIST', 'EMNISTBalanced', 'FashionMNIST','KMNIST']:
                self.trans = T.Compose([
                    T1.Resize((28, 28)),
                    T1.Normalize(mean, std),
                    T1.ToTensor(),
                ])
            elif dataset in ['CIFAR10', 'CIFAR100']:
                self.trans = T.Compose([
                    T.Resize((32, 32)),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])
            else:
                self.trans = T.Compose([
                    T.Resize((64, 64)),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])

    def __getitem__(self, idx):
        # 获取一组图片和对应的标签
        ims, lb_prob,lb_idx,indices_u = self.data[idx], self.labels[idx],self.labels_idx[idx],self.indices_u[idx]
        labels = self.labels_real[idx]
        # 对图片进行变换，这里假设使用了名为 self.trans 的图像变换函数
        if self.mode == 'train_u_co':
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            x_strong0 = torch.stack([self.trans(im)[1] for im in ims])
            x_strong1 = torch.stack([self.trans(im)[2] for im in ims])
            ims_transformed = [x_weak, x_strong0, x_strong1]
            return ims_transformed, lb_prob, labels,lb_idx,indices_u
        elif self.mode == 'train_u_L^2P-AHIL':
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            x_strong0 = torch.stack([self.trans(im)[1] for im in ims])
            ims_transformed = [x_weak, x_strong0]
            return ims_transformed, lb_prob, labels,lb_idx,indices_u
        elif self.mode == 'train_u_DLLP':
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            ims_transformed = [x_weak]
            return ims_transformed, lb_prob, labels, lb_idx,indices_u


    def __len__(self):
        leng = len(self.data)
        return leng

class SVHN(Dataset):
    def __init__(self, dataset, data, labels, labels_real, labels_idx, indices_u, mode):
        super(SVHN, self).__init__()
        self.data, self.labels, self.labels_real, self.labels_idx, self.indices_u = data, labels, labels_real, labels_idx, indices_u
        self.mode = mode
        assert len(self.data) == len(self.labels)

        mean, std = (0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744)  # SVHN uses different mean and std

        trans_weak = T.Compose([
            T.Resize((32, 32)),  # 调整图像大小为 32x32 像素
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),  # 添加填充并随机裁剪，用于数据增强
            T.RandomAffine(
                degrees=15,  # +/- 5度的旋转
                translate=(0.125, 0.125)),
            T.Normalize(mean, std),  # 标准化图像（mean和std是均值和标准差）
            T.ToTensor(),  # 将图像转换为张量
        ])

        # 定义强数据增强方法（trans_strong0）
        trans_strong0 = T.Compose([
            T.Resize((32, 32)),  # 调整图像大小为 32x32 像素
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),  # 添加填充并随机裁剪，用于数据增强
            RandomAugment(3, 5),
            # 自定义数据增强方法（在代码中未提供具体实现）
            T.Normalize(mean, std),  # 标准化图像（mean和std是均值和标准差）
            T.ToTensor(),  # 将图像转换为张量
        ])

        # 定义更强的数据增强方法（trans_strong1）
        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),  # 将张量转换为图像
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),  # 随机裁剪和缩放，增强数据多样性
            # 删除水平翻转操作
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # 随机颜色调整，增加数据多样性
            ], p=0.8),  # 80%的概率应用颜色调整
            transforms.RandomGrayscale(p=0.2),  # 20%的概率转换为灰度图像
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean, std),  # 标准化图像（mean和std是均值和标准差）
        ])
        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_co':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_L^2P-AHIL':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        elif self.mode == 'train_u_DLLP':
            self.trans = OneCropsTransform(trans_weak)
        else:
            if dataset in ['MNIST', 'EMNISTBalanced', 'FashionMNIST']:
                self.trans = T.Compose([
                    T1.Resize((64, 64)),
                    T1.Normalize(mean, std),
                    T1.ToTensor(),
                ])
            else:
                self.trans = T.Compose([
                    T.Resize((64, 64)),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])

    def __getitem__(self, idx):
        # 获取一组图片和对应的标签
        ims, lb_prob, lb_idx, indices_u = self.data[idx], self.labels[idx], self.labels_idx[idx], self.indices_u[idx]
        labels = self.labels_real[idx]
        # 对图片进行变换，这里假设使用了名为 self.trans 的图像变换函数
        if self.mode == 'train_u_co':
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            x_strong0 = torch.stack([self.trans(im)[1] for im in ims])
            x_strong1 = torch.stack([self.trans(im)[2] for im in ims])
            ims_transformed = [x_weak, x_strong0, x_strong1]
            return ims_transformed, lb_prob, labels, lb_idx, indices_u
        elif self.mode == 'train_u_L^2P-AHIL':
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            x_strong0 = torch.stack([self.trans(im)[1] for im in ims])
            ims_transformed = [x_weak, x_strong0]
            return ims_transformed, lb_prob, labels, lb_idx, indices_u
        elif self.mode == 'train_u_DLLP':
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            ims_transformed = [x_weak]
            return ims_transformed, lb_prob, labels, lb_idx,indices_u

    def __len__(self):
        leng = len(self.data)
        return leng


def get_train_loader(classes,dataset, batch_size, bag_size, root='data', method='co',supervised=False):
    data_u, label_prob, labels,label_idx,dataset_length,indices_u,input_dim = load_data_train(classes, dataset=dataset, dspth=root, bagsize=bag_size)
    if dataset != 'SVHN':
        ds_u = Cifar(
                dataset=dataset,
                data=data_u,
                labels=label_prob,
                labels_real=labels,
                labels_idx=label_idx,
                indices_u=indices_u,
                mode='train_u_%s' % method
            )
    else:
        ds_u = SVHN(
            dataset=dataset,
            data=data_u,
            labels=label_prob,
            labels_real=labels,
            labels_idx=label_idx,
            indices_u=indices_u,
            mode='train_u_%s' % method
        )
    #sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
    sampler_u = RandomSampler(ds_u, replacement=False)
    batch_sampler_u = BatchSampler(sampler_u, batch_size, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u,
        batch_sampler=batch_sampler_u,
        num_workers=16,
        pin_memory=True
    )
    return dl_u,dataset_length,input_dim


def get_val_loader(dataset, batch_size, num_workers, pin_memory=True, root='data',n_classes=10):
    data, labels = load_data_val(dataset, dspth=root,n_classes=n_classes)
    if dataset !='SVHN':
        ds = Cifar2(
            dataset=dataset,
            data=data,
            labels=labels,
            mode='test'
        )
    else:
        ds = SVHN2(
            dataset=dataset,
            data=data,
            labels=labels,
            mode='test'
        )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl


class SVHN2(Dataset):
    def __init__(self, dataset, data, labels, mode):
        super(SVHN2, self).__init__()
        self.data, self.labels = data, labels
        self.mode = mode
        assert len(self.data) == len(self.labels)

        # 根据 SVHN 数据集的均值和标准差进行设置
        mean, std = (0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744)
        trans_weak = T.Compose([
            T.Resize((32, 32)),
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        trans_strong0 = T.Compose([
            T.Resize((32, 32)),  # 调整图像大小为 32x32 像素
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),  # 添加填充并随机裁剪，用于数据增强

            RandomAugment(2, 10),
            # 自定义数据增强方法（在代码中未提供具体实现）
            T.Normalize(mean, std),  # 标准化图像（mean和std是均值和标准差）
            T.ToTensor(),  # 将图像转换为张量
        ])
        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_co':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_L^2P-AHIL':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            if dataset in ['MNIST', 'EMNISTBalanced', 'FashionMNIST']:
                self.trans = T.Compose([
                    T1.Resize((64, 64)),
                    T1.Normalize(mean, std),
                    T1.ToTensor(),
                ])
            else:
                self.trans = T.Compose([
                    T.Resize((32, 32)),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng


class Cifar2(Dataset):
    def __init__(self, dataset, data, labels, mode):
        super(Cifar2, self).__init__()
        self.data, self.labels = data, labels
        self.mode = mode
        assert len(self.data) == len(self.labels)
        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == 'CIFAR100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        elif dataset == 'FashionMNIST':
            mean, std = (0.1307), (0.3081)
        elif dataset == 'EMNISTBalanced':
            mean, std = (0.1307), (0.3081)
        elif dataset == 'MNIST':
            mean, std = (0.1307), (0.3081)
        elif dataset == 'KMNIST':
            mean, std = (0.1307), (0.3081)
        elif dataset =='miniImageNet':
            mean, std=(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean = (0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225)
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            trans_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in [
            'Corel16k', 'Corel5k', 'Delicious', 'Bookmarks',
            'Eurlex_DC', 'Eurlex_SM', 'Scene', 'Yeast'
        ]:
            # 这些数据集本身就是已提取好的扁平特征向量，
            # 在训练前不需要任何数据增强或归一化。
            trans_weak = T.Compose([
                T.ToTensor(),  # ← 只是把 ndarray → FloatTensor
            ])
            trans_strong0 = T.Compose([
                T.ToTensor(),  # 强弱同样：啥也不干
            ])
            trans_strong1 = T.Compose([
                T.ToTensor(),
            ])

        elif dataset in ['FashionMNIST','KMNIST']:
            trans_weak = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomHorizontalFlip(p=0.5),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomHorizontalFlip(p=0.5),
                RandomAugment1(2, 10),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['MNIST', 'EMNISTBalanced']:
            trans_weak = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomAffine(
                    degrees=15,  # +/- 5度的旋转
                    translate=(0.1, 0.1),  # 最多10%的水平和垂直平移
                    scale_range=(0.9, 1.1)  # 0.9到1.1倍的缩放
                ),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomAffine(
                    degrees=15,  # +/- 5度的旋转
                    translate=(0.1, 0.1),  # 最多10%的水平和垂直平移
                    scale_range=(0.9, 1.1)  # 0.9到1.1倍的缩放
                ),
                RandomAugment1(2, 10),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
                transforms.RandomAffine(
                    degrees=15,  # +/- 5度的旋转
                    translate=(0.1, 0.1),  # 最多10%的水平和垂直平移
                    scale=(0.9, 1.1)  # 0.9到1.1倍的缩放
                ),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['TinyImageNet']:
            trans_weak = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['miniImageNet']:
            trans_weak = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_co':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_L^2P-AHIL':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            if dataset in ['MNIST', 'EMNISTBalanced', 'FashionMNIST','KMNIST']:
                self.trans = T.Compose([
                    T1.Resize((28, 28)),
                    T1.Normalize(mean, std),
                    T1.ToTensor(),
                ])
            elif dataset in ['CIFAR10', 'CIFAR100','SVHN']:
                self.trans = T.Compose([
                    T.Resize((32, 32)),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])
            else:
                self.trans = T.Compose([
                    T.Resize((64, 64)),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng
