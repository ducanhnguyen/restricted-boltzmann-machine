# restricted-boltzmann-machine
The implementation of rbm, which is an improvement of Boltzmann Machine. RBM is used in dimensionality reduction, collaborative filtering, etc.

RBM has one visible layer (v) and one hidden layer (h). We can calculate h from v. Otherwise, we also can calculate v from h.


## 1. Bernoulli RBM

Both sides only have values of 0 or 1 (boolean values).

<p align="center">
<img src="https://github.com/ducanhnguyen/restricted-boltzmann-machine/blob/master/img/rbm.png" width="250">
</p>

#### Notation

| Notation | Description |
| --- | --- |
|v (NxD)| the visible layer|
|W (DxM)| kernel|
|h (NxM)| the hidden layer|
|b (Dx1), c (Mx1)| biases|
|F| energy function|
|N| the number of observations|
|D| the number of features|
|M| the number of hidden units|
|p(v\| h)| the probability of v given h (is a vector of probabilities)|
|p(h\| v)| the probability of h given v (is a vector of probabilities)|

#### Function activation 

The activation functions of p(v|h) and p(h|v) are sigmoid.

<p align="center">
<img src="https://github.com/ducanhnguyen/restricted-boltzmann-machine/blob/master/img/probability_sigmoid.png" width="200">
</p>

#### Loss function

Rather than using cross-entropy, the authors use another kind of loss function denoted by L. It is observed that minimizing L also means that minimizing the cross-entropy.

We try to minimize the following loss function:

<p align="center">
L = F(v) - F(v')
</p>

v' is a sample of (v, h). We generate v' by performing Gibbs sampling with one step. More than one steps are good, but it is not necessary since one step is good enough.

The formula of the energy function F is as follows:

<p align="center">
<img src="https://github.com/ducanhnguyen/restricted-boltzmann-machine/blob/master/img/energy_function.png" width="350">
</p>

#### Experiments

The experiment is performed on PyCharm 2018.3.4, python 3, mac osx.

<img src="https://github.com/ducanhnguyen/restricted-boltzmann-machine/blob/master/img/rbm_cost.png" width="550">

The left images are the original ones. The right images ars the reconstructed images by using RBM.

| Example 1 | Example 2 | Example 3 |
| --- | --- | --- |
|<img src="https://github.com/ducanhnguyen/restricted-boltzmann-machine/blob/master/img/rbm_0_reconstruction.png" width="350">|<img src="https://github.com/ducanhnguyen/restricted-boltzmann-machine/blob/master/img/rbm_1_construction.png" width="350">|<img src="https://github.com/ducanhnguyen/restricted-boltzmann-machine/blob/master/img/rbm_4_reconstruction.png" width="350">|

## 2. Stacked RBM

It is similar to stacked autoencoder.

Stacked RBM = continuous layer of RBM

#### Experiment

I test with three RBM on digit-recognizer dataset. The first hidden layer has 64 units. The second has 32 units. The last hidden layer has 16 unit. The number of classes on output layer is 10 classes.

Each hidden layer is trained in turn from the first hidden layer to the last hidden layer. In each training phase, epoch = 100, learning_rate = 0.001.

<img src="https://github.com/ducanhnguyen/restricted-boltzmann-machine/blob/master/img/stacked_rbm.png" width="550">

Kaggle score: 0.95757% on the dataset digit-recognizer.
