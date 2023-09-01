# DANN (Domain Adversarial Neural Network)-Pytorch
This repository provides a Pytorch implementation of DANN (Domain Adversarial Neural Networks) introduced by Ganin et al. [1]. The code includes the supporting scripts for reproducing the results obtained in [1] for the domain adaptation task, explicitly using the datasets mnist and mnist modified. 

# Pre-requisites
1- Python 3.7.4

2- Pytorch 1.13

Aiming at simplifying Python environment issues, we provide the [docker container](https://hub.docker.com/r/psoto87/pytorch1.13) used to conduct the experiments' results obtained with this code.

# Experiments
This code reproduces the experiments carried out among the datasets mnist and mnist modified. The difference between such datasets is represented in the following figure, taken from [1].


# References
[1] Ganin and V. Lempitsky, “Unsupervised   domain   adaptation  by backpropagation,”arXiv preprint arXiv:1409.7495, 2014.

