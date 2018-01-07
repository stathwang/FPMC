# FPMC: Factorized Personalized Markov Chains for Next Basket Recommendation
Paper: [Factorized Personalized Markov Chains for Next Basket Recommendation](https://www.ismll.uni-hildesheim.de/pub/pdfs/RendleFreudenthaler2010-FPMC.pdf) (Rendle et al. 2010)

This repository contains my implementation of FPMC in R that predicts a user's next purchase based on the past purchase history. The algorithm combines a matrix factorization of user-item matrix to model user preferences and factorized (first order) Markov chains to consider sequential dynamics.

### Problem Formulation
![FPMC](/img/fpmc_model.png)

### Main Ideas
FPMC models both long-term **user preference** (matrix factorization) and short-term **sequential dynamics** (markov chains). It factorizes two matrices: the user-item matrix and the item-item transition matrix using a S-BPR (Sequential Bayesian Personalized Ranking) loss and sums up the similarity based on both. S-BPR uses a sigmoid function to characterize the probability that a true item is ranked higher than a false item given a user and the model parameters, assuming independence of users and time steps.

It has been shown using multiple datasets that **FOSSIL** (Factorized Sequential Prediction with Item Similarity Models) performs better than FPMC, but the performance could vary depending on the nature of data generating process and the task domain.

### Instructions
Run the *preprocess.R* script on *movielens_trunc.csv* file and then the *fpmc.R* on the training set produced. I'm going to upload the python version of the algorithm as well as functions for evaluation metrics (i.e. AUC, precision, recall) soon, so stay tuned.
