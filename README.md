# KGCN-pytorch-updated

This is an updated PyTorch implementation of the KGCN model, building upon two previous works: [TensorFlow KGCN](https://github.com/hwwang55/KGCN) (Original) and [PyTorch KGCN](https://github.com/zzaebok/KGCN-pytorch).

Reference:

> Knowledge Graph Convolutional Networks for Recommender Systems  
Hongwei Wang, Miao Zhao, Xing Xie, Wenjie Li, Minyi Guo.  
In Proceedings of The 2019 Web Conference (WWW 2019)  
https://dl.acm.org/citation.cfm?id=3313417  
https://arxiv.org/abs/1904.12575

## Dataset

- ### Movie

    Raw rating file for movie is too large to be contained in this repo.

    Downlad the rating data first
    ```bash
    $ wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
    $ unzip ml-20m.zip
    $ mv ml-20m/ratings.csv data/movie/
    ```

- ### Music

    Nothing to do

- ### Other dataset

    If you want to use your own dataset, you need to prepare 2 data.

    1. Rating data
        - Each row should contain (user-item-rating)
        - In this repo, it is pandas dataframe structure. (look at `data_loader.py`)
    2. Knowledge graph
        - Each triple(head-relation-tail) consists of knowledge graph
        - In this repo, it is dictionary type. (look at `data_loader.py`)

## Structure
1.  `data_loader.py`
    - data loader class for movie / music dataset
    - you don't need it if you make custom dataset

2. `aggregator.py`
    - aggregator class which implements 3 aggregation functions

3. `model.py`
    - KGCN model network

## Running the code

Look at the `KGCN.ipynb`.

It contains
- how to construct Datset
- how to construct Data loader
- how to train network

## Debugging

- [x] model.py > _aggregate() > relations > 存在 relation 的 tensor 分量有离群值
- [x] 调查 PyTorch torch.nn.Embedding
- [x] .view 方法是什么
- [x] 再次确认数据预处理的正确性：绝对正确，为了保证更加符合数据，甚至直接把 i2e 事先运行完毕
- [x] 研究 sklearn 里的 label encoder：https://scikit-learn.org/stable/modules/preprocessing_targets.html#preprocessing-targets
- [x] kg 是否被正确地构建_construct_kg
- [x] gen_adj 是否生成正确的邻接列表
- [ ] forward 中的 gen_neighbor 是否正确生成
- [ ] _aggregate 的后半部分再看一下理解以下
- [ ] 给 model.py 重新注释