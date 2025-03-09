# DAPNet: multi-view graph contrastive network incorporating disease clinical and molecular associations for disease progression prediction

## Introduction

This repository contains source code and datasets for paper "DAPNet: multi-view graph contrastive network incorporating disease clinical and molecular associations for disease progression prediction". In this study, we proposed DAPNet, a deep learning-based disease progression prediction model that solely utilizes the comorbidity duration (without relying on multi-modal data or comprehensive medical records) and disease associations from biomedical knowledge graphs to deliver high-performance prediction. DAPNet is the first to apply multi-view graph contrastive learning to disease progression prediction tasks. Compared with other studies on comorbidities, DAPNet innovatively integrates molecular-level disease association information, combines disease co-occurrence and ICD-10, and fully explores the associations between diseases;

  
## A Quick Start
Before our program runs, we need to make some preparations.

### Install Python libraries needed


```shell
$ conda create -n DP python=3.7
$ conda activate DP
$ pip install -r requirements.txt
```
We recommend using version 1.15 of TensorFlow 1.
### Diseases network  creation

Based on the relationship data of diseases, this study sorted and constructed three disease networks from different perspectives. Although their data sources are different, the file format is consistent. 

- Adjacency matrix(data/adj_matrix.csv). And their row and column sizes are the same.
- Feature matrix(data/Graph_embeddingfull.npy). The rows correspond to the number of diseases, and the columns correspond to initial weights, such as text embedding.
### Datasets creation
To better represent the phenotypes of patients, this study proposed a new method for calculating illness duration to refect the impact of disease duration. The process of constructing the dataset is detailed in the  '/pre' folder. The final file can refer to 'data/train_demo'.csv.

### Running
Jump to /model folder, and run the Python file.
```shell
nohup python -u train.py > DAPNet.txt 2>&1 &
```
