# Training Arguments

This project supports the following command-line arguments for training.

---

## Dataset & Data Settings

| Argument      | Type | Default  | Description                                    |
| ------------- | ---- | -------- | ---------------------------------------------- |
| `--root`      | str  | `./data` | Dataset directory                              |
| `--dataset`   | str  | `KMNIST` | Dataset name                                   |
| `--n-classes` | int  | `10`     | Number of classes in the dataset               |
| `--n-epoches` | int  | `1024`   | Number of training epochs                      |
| `--batchsize` | int  | `64`      | Training batch size (number of bags per batch) |
| `--bagsize`   | int  | `16`    | Number of instances per bag                    |

---

## Optimization Settings

| Argument     | Type  | Default  | Description                                      |
| ------------ | ----- | -------- | ------------------------------------------------ |
| `--lr`       | float | `2.5e-3` | Learning rate                                    |
| `--momentum` | float | `0.9`    | Momentum for optimizer                           |
| `--eps`      | double | `1e-30`  | Numerical stability epsilon                      |
| `--seed`     | int   | `13`     | Random seed (negative value disables fixed seed) |

---

## Warmup Settings

| Argument        | Type  | Default | Description                                                |
| --------------- | ----- | ------- | ---------------------------------------------------------- |
| `--warmup-frac` | float | `0.08`  | Fraction of total iterations used for learning rate warmup |
| `--warmup-lr`   | float | `5e-5`  | Initial learning rate during warmup                        |

The number of warmup iterations is computed as:

```
warmup_iter = warmup_frac × total_iterations
```

The warmup learning rate ratio is computed as:

```
warmup_ratio = warmup_lr / lr
```

---

## Bag Construction Strategy

| Argument      | Type | Default  | Description                                                         |
| ------------- | ---- | -------- | ------------------------------------------------------------------- |
| `--pl_method` | str  | `random` | Bag-generation strategy. Options: `random`, `cluster`, `alphafirst` |
| `--pi`        | str  | `10`     | Bag purity / concentration control for `alphafirst` and `cluster`. Smaller `pi` ⇒ more homogeneous bags; larger `pi` ⇒ more mixed bags. |
---

## Experiment Management

| Argument       | Type | Default | Description                              |
| -------------- | ---- | ------- | ---------------------------------------- |
| `--exp-dir`    | str  | `LLP-PVC`    | Experiment identifier (output directory) |
| `--checkpoint` | str  | `""`    | Path to pretrained model checkpoint      |

---

# Example Usage

```bash
python LLP-PVC.py \
    --dataset KMNIST \
    --n-classes 10 \
    --bagsize 256 \
    --batchsize 4 \
    --lr 2.5e-3 \
    --warmup-frac 0.08 \
    --warmup-lr 5e-5 \
    --pl_method random \
    --exp-dir experiment_1
```
