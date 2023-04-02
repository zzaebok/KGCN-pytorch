# KGCN-pytorch-updated

This is an updated PyTorch implementation of the KGCN model, building upon two previous works: [TensorFlow KGCN](https://github.com/hwwang55/KGCN) and [PyTorch KGCN](https://github.com/zzaebok/KGCN-pytorch). The former is the offical one while the latter is a PyTorch version implemented by [@zzaebok](https://github.com/zzaebok). This repo modifies directly on his [PyTorch KGCN](https://github.com/zzaebok/KGCN-pytorch).

- [KGCN-pytorch-updated](#kgcn-pytorch-updated)
  - [1.1. KGCN at a Glance](#11-kgcn-at-a-glance)
  - [1.2. Running the Code](#12-running-the-code)
  - [1.3. Dataset](#13-dataset)
    - [1.3.1. `movie`](#131-movie)
    - [1.3.2. `music`](#132-music)
    - [1.3.3. `product`](#133-product)
    - [1.3.4. Other Dataset](#134-other-dataset)
  - [1.4. Structure](#14-structure)
  - [1.5. Comparison with PyTorch KGCN](#15-comparison-with-pytorch-kgcn)
    - [1.5.1. Add a new handcrafted dataset `product`](#151-add-a-new-handcrafted-dataset-product)
    - [1.5.2. Add a new mixer `transe`](#152-add-a-new-mixer-transe)
    - [1.5.3. Add `batch_experiments.ipynb` for convenient experiment](#153-add-batch_experimentsipynb-for-convenient-experiment)
    - [1.5.4. Fix RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (CPU)](#154-fix-runtimeerror-indices-should-be-either-on-cpu-or-on-the-same-device-as-the-indexed-tensor-cpu)
    - [1.5.5. Fix IndexError: index out of range in self](#155-fix-indexerror-index-out-of-range-in-self)


## 1.1. KGCN at a Glance

KGCN is **K**nowledge **G**raph **C**onvolutional **N**etworks for recommender systems, which uses the technique of graph convolutional networks (GCN) to proces knowledge graphs for the purpose of recommendation.

![KGCN Framework](./assets/framework.png)

Figure 1: KGCN Framework

Reference:

> Knowledge Graph Convolutional Networks for Recommender Systems
> Hongwei Wang, Miao Zhao, Xing Xie, Wenjie Li, Minyi Guo.
> In Proceedings of The 2019 Web Conference (WWW 2019)
> * ACM: https://dl.acm.org/citation.cfm?id=3313417
> * arXiv: https://arxiv.org/abs/1904.12575
> * Paper With Code: https://paperswithcode.com/paper/190412575

## 1.2. Running the Code

For showing result under one specific hyper parameter setting, use `KGCN.ipynb` or `KGCN.py`.

For batch experiments, use `batch_experiments.ipynb`.

p.s. `KGCN.ipynb` and `KGCN.py` have the same functionality, but the latter is modularized for easy debugging and reuse.

## 1.3. Dataset

### 1.3.1. `movie`

Raw rating file for movie is too large to be contained in this repo.

Downlad the rating data first

```bash
$ wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
$ unzip ml-20m.zip
$ mv ml-20m/ratings.csv data/movie/
```

### 1.3.2. `music`

Nothing to do

### 1.3.3. `product`

This dataset is built upon the Rec-Tmall dataset. Check [./data/product/preprocessing.ipynb](./data/product/preprocessing.ipynb) for more information.

### 1.3.4. Other Dataset

If you want to use your own dataset, you need to prepare 3 files.

- Rating Data
   - Each row should contain (user-item-rating)
   - In this repo, it is pandas dataframe structure. (look at `data_loader.py`)
- Knowledge Graph
   - Each triple(head-relation-tail) consists of knowledge graph
   - In this repo, it is dictionary type. (look at `data_loader.py`)
- Item Index to Entity Index Mapping
   - Check [./data/product/preprocessing.ipynb](./data/product/preprocessing.ipynb) to see my solutions.

## 1.4. Structure

Core files:

- `data_loader.py`
  - data loader class for movie / music dataset
  - you don't need it if you make custom dataset
- `aggregator.py`
  - aggregator class which implements 3 aggregation functions
  - and 2 mixers
- `model.py`
  - KGCN model network

![Dependency Graph](./assets/Dependency%20Graph.svg)

Figure 2: Dependency Graph

## 1.5. Comparison with [PyTorch KGCN](https://github.com/zzaebok/KGCN-pytorch/tree/3b0bb56da4b6759d204de06f1d4547e9b4abe3ce)

### 1.5.1. Add a new handcrafted dataset `product`

* Dataset source: [Rec-Tmall](https://tianchi.aliyun.com/dataset/140281)
* Preprocessing script: [preprocessing.ipynb](data/product/preprocessing.ipynb)

### 1.5.2. Add a new mixer `transe`

`transe` mixer has a better performace on divergence speed, time complexity, divergence loss value and AUC value, final-epoch loss value and AUC value in most cases.

See `_mix_neighbor_vectors_TransE()` in [aggregator.py](./aggregator.py)

### 1.5.3. Add `batch_experiments.ipynb` for convenient experiment

Automatically conduct multiple experiments.

1. Config many parameter sets once
2. Run
3. Check results in a well-formatted print-out

### 1.5.4. Fix RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (CPU)

*Before*

https://github.com/zzaebok/KGCN-pytorch/blob/3b0bb56da4b6759d204de06f1d4547e9b4abe3ce/model.py#L79-L80

*After*

```python
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h].cpu()]).view((self.batch_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h].cpu()]).view((self.batch_size, -1)).to(self.device)
```

### 1.5.5. Fix IndexError: index out of range in self

*Before*

https://github.com/zzaebok/KGCN-pytorch/blob/3b0bb56da4b6759d204de06f1d4547e9b4abe3ce/model.py#L34-L35

*After*

```python
        self.adj_ent = torch.zeros(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.zeros(self.num_ent, self.n_neighbor, dtype=torch.long)
```
