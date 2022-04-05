# Data Mining: Graph Learning on Biological Data

This is the fourth homework of the [Data Mining](http://aris.me/index.php/data-mining-2021) course at Sapienza University of Rome.

The work is centered on the **Node Classification** task of **Graph Learning** while building and comparing two different approaches to modeling: **Node2Vec** embbedings + basic classifier of choice (MLP, SVM, etc.) vs **Graph Neural Networks (GNN)** with a subsequent application of the graph **Explainability** technique.

The specifications of the task assignment can be found in the accompanying pdf: [task_assignment.pdf](https://github.com/olga-sorokoletova/Data-Mining/blob/main/Homework%204/task_assignment.pdf).

## Data

Biological Data used to learn on: 
1. **Protein–Protein Interaction (PPI)** files of the [Biogrid](https://github.com/olga-sorokoletova/Data-Mining/blob/main/Homework%204/Biogrid_4.4.199_Hs.txt) dataset;
2. **Gene–Disease Association (GDA)** data from [DisGeNET](https://www.disgenet.org/downloads).

## Implementation

The detailed description of the **architectural choices, approaches to data inspection and preparation, challenges and issue resolution tactis, experiments, model performances** and **graphical results** is provided in the [report.pdf](https://github.com/olga-sorokoletova/Data-Mining/blob/main/Homework%204/report.pdf), which is suggested for reading before diving deep into code implementation: [```code.ipynb```](https://github.com/olga-sorokoletova/Data-Mining/blob/main/Homework%204/code.ipynb).

<p align="center">
  <img src="./report%20images/subgraph.png" width="504" height="504"/>
</p>

## Author
- Olga Sorokoletova - 1937430
