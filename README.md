# Bayesian\_INN
This repository contains the code to reproduce the results of **Understanding Event-Generation Networks via Uncertainties** <https://arxiv.org/pdf/2104.04543.pdf>

Implementation is in PyTorch 1.8.0. Features included are:
* Bayesian linear layers
* MMD loss
* Train on weighted points

## Usage
```
clone the repository
git clone git@github.com:marcobellagente93/Bayesian_INN.git
generate linear and quadratic toy datasets via data/generate\_toy
Run the code
python train.py
```
