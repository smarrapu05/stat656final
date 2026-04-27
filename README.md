# Enzyme Classification Using Graphs

An exploration of graph based neural networks as predictors for classifying enzymes



Graph based protein analysis is a vital area of bioinformatics because even incredibly structurally similar proteins can have incredibly different functions. For instance myoglobin is incredibly close to hemoglobin, but has half of the functionality since it can’t deposit oxygen and only holds onto it. Understanding these sorts of specific residue and structural differences helps in fields like drug discovery and synthetic biology. In specific we are looking at a subclass of proteins called enzymes. Enzymes are a specific type of protein that catalyzes reactions in order to allow for them to occur easier. Enzymes are classified into 1 of 6 top level enzyme classes that categorize them based on the type of reaction they catalyze and how they accomplish it. Our project is classification based, where we are trying to classify these protein networks into each of these Enzyme Commission (EC) numbers. This project explores the implementation and differences between traditional machine learning methods and two types of GNNs (a GCN and a GAT).





**eda.R** - This file contains the code for the exploratory data analysis of the graph properties based on EC class

**randomForest.py** - This file contains the code for the random forest model

**gcn\_enzymes.py** - This file contains a custom GCN implementation in pytorch. It includes 6-fold cross validation for this model as well as some diagnostics (confusion matrix, loss curve, and accuracy curve).

**gat\_enzymes.py** - This file contains a similar custom GAT implementation. The data parsing and formatting is shared from gcn\_enzymes so it imports the functions. It also has the same diagnostic plots



*/ENZYMES* - This is the ENZYMES dataset from TUDataset \[https://chrsmrrs.github.io/datasets/docs/datasets/]. It is here because the gat and gcn require it to function

*/outputs*  - This folder contains the outputs from the eda and model files.



Both csvs in the directory are outputs from eda.R, but they are stored in the main folder because randomForest depends on them to complete its function.

