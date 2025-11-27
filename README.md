# 🧩 adj2img

This document provides an overview of how to configure and run experiments with **adj2img**, a framework that transforms adjacency information into image-like representations for graph-based learning.

---

## ⚙️ Arguments

### Structural and Graph Parameters
| Argument | Type | Default | Description |
|-----------|------|----------|-------------|
| `--hops` | `int` | `3` | Number of hops used to extract the local structural subgraph around each node. Determines the receptive field for generating the adjacency image. |
| `--img_size` | `int` | `512` | Size (in pixels) of the image representation generated from the adjacency matrix. Larger values capture finer structural details. |
| `--dataset_name` | `str` | `'chameleon'` | Name of the dataset used for training and evaluation. Supported examples include `cora`, `citeseer`, `chameleon`, etc. |

---

### Model Architecture
| Argument | Type | Default | Description |
|-----------|------|----------|-------------|
| `--embed_dim` | `int` | `64` | Dimension of the embedding space for both node and image representations. |
| `--num_layers` | `int` | `12` | Number of Transformer or GNN layers used in the architecture. Controls model depth. |
| `--num_heads` | `int` | `4` | Number of attention heads in the Transformer encoder. A higher number increases representational granularity. |
| `--mlp_dim` | `int` | `64` | Hidden dimension size of the MLP blocks used in the Transformer architecture. |
| `--dropout` | `float` | `0.7` | Dropout rate applied throughout the model for regularization. Higher values reduce overfitting. |

---

### Training Configuration
| Argument | Type | Default | Description |
|-----------|------|----------|-------------|
| `--lr` | `float` | `0.001` | Learning rate for model optimization. Adjust for training stability and convergence speed. |
| `--seed` | `int` | `2026` | Random seed to ensure experiment reproducibility. |
| `--missing_rate` | `int` | `0` | Option of node features to randomly mask during training, used to simulate incomplete feature settings. |

---

### Prototype and Image Parameters
| Argument | Type | Default | Description |
|-----------|------|----------|-------------|
| `--img_embed_num` | `int` | `100` | Number of structural prototype tokens used for image embedding. Controls the granularity of structural pattern representation. |

---
## Before running the examples
```bash
mkdir graphon
mkdir image
mkdir results
```

## 🧠 Example Usage
```bash
python main.py \
  --dataset_name chameleon \
  --hops 3 \
  --img_size 512 \
  --embed_dim 64 \
  --num_layers 2 \
  --num_heads 1 \
  --mlp_dim 128 \
  --dropout 0.7 \
  --lr 0.001 \
  --img_embed_num 200 \
  --missing_rate 0 \
  --seed 42
```

## Hyperparameters for each dataset

| Dataset       | hops | img_size | embed_dim | num_layers | num_heads | mlp_dim | dropout | lr      | img_embed_num | 
|---------------|------|----------|-----------|------------|-----------|---------|---------|----------|----------------|
| cora          | 3    | 512      | 256        | 1          | 1         | 512     | 0.9     | 0.001    | 20            | 
| citeseer      | 3    | 512      | 256        | 4          | 1         | 128     | 0.7     | 0.001    | 50            | 
| chameleon     | 3    | 512      | 64        | 2          | 1         | 128     | 0.7     | 0.001    | 200            |
| squirrel      | 3    | 512      | 64        | 4          | 1         | 64     | 0.7     | 0.001    | 50            | 
| ogbn-arxiv    | 3    | 512      | 512       | 1          | 16         | 256     | 0.9     | 0.0001   | 20            |

## Dataset statistics

| Dataset     | Nodes  | Edges  | Features | Classes |
|-------------|--------|--------|----------|---------|
| cora        | 2,708  | 5,278  | 1,433    | 7       |
| citeseer    | 3,327  | 4,552  | 3,703    | 6       | 
| chameleon   | 2,277  | 31,371 | 2,325    | 5       | 
| squirrel    | 5,201  | 198,353| 2,089    | 5       | 
| ogbn-arxiv  | 169,343| 1,166,243 | 128    | 40      |
