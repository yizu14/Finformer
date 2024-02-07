# Finformer: A Static-dynamic Spatiotemporal Framework for Stock Trend Prediction

The implementation for IEEE BigData '23 paper "Finformer: A Static-dynamic Spatiotemporal Framework for Stock Trend Prediction".

## What's new

[2024.02.07] We release our codes for stock trend prediction.

## Model Overview

The Finformer is composed by the Temporal Encoder, the Sparse Static-Dynamic Transformer Encoder, and the Gated Fusion module.

<img width="700" alt="image" src="https://github.com/yizu14/Finformer/blob/0b784b80397e843b1415aa6ca9e3669bc7112370/imgs/Finformer.png">

## Datasets

Our tests are based on CSI300 and CSI100 datasets. Unfortunately, due to the origin of our data, we do not have the right to make data publicly available. However, we are trying our best to provide guidance on obtaining the dataset. You may use data from alternative sources, but please be cautious about the accuracy of constituent stock information for the indices during different periods.

## How to Run Our Model

eval_notebook.ipynb - A performance evaluation notebook

config.yaml - Model configuration include hyperparameters, train/valid/test dates and portfolio configurations can be set here.

loader.py - Cross-sectional dataloader. We used a modified version of MTSDataset provided by TRA (Lin, Hengxu, et al. "Learning multiple stock trading patterns with temporal routing adaptor and optimal transport." SIGKDD '21).

metrics.py - Metrics implementation and a modified version of TopkDropoutStrategy (See paper V.E. Portfolio Evaluation).

model.py - Architecture of Finformer

utils.py - Graph and dataset process functions

train.py - Training framework

It can be run by execute the train.py after configure the data path correctly.

## Acknowledgement
We thank
1. Qlib - for implementation framework. https://github.com/microsoft/qlib
2. TRA (Lin, Hengxu, et al. "Learning multiple stock trading patterns with temporal routing adaptor and optimal transport." SIGKDD '21) - for evaluation notebook. https://arxiv.org/pdf/2106.12950.pdf

### Citation

If you find our paper useful, please consider citing our work

```bibtex
@inproceedings{zu2023finformer,
  title={Finformer: A Static-dynamic Spatiotemporal Framework for Stock Trend Prediction},
  author={Zu, Yi and Mi, Jiacong and Song, Lingning and Lu, Shan and He, Jieyue},
  booktitle={2023 IEEE International Conference on Big Data (BigData)},
  pages={1460--1469},
  year={2023},
  organization={IEEE}
}
```
