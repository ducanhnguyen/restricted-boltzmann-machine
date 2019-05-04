# restricted-boltzmann-machine
The implementation of rbm, which is an improvement of Boltzmann Machine.

Like autoencoder, RBM is a dimensionality reduction technique. RBM has one visible layer (v) and one hidden layer (h). We can calculate h from v. Otherwise, we also can calculate v from h. Both sides only have values of 0 or 1 (yes or no)

<img src="https://github.com/ducanhnguyen/restricted-boltzmann-machine/blob/master/img/rbm.png" width="350">

### Environment

PyCharm 2018.3.4, python 3, mac osx

### Experiments

<img src="https://github.com/ducanhnguyen/restricted-boltzmann-machine/blob/master/img/rbm_cost.png" width="550">

The left images are the original ones. The right images ars the reconstructed images by using RBM.

<img src="https://github.com/ducanhnguyen/restricted-boltzmann-machine/blob/master/img/rbm_0_reconstruction.png" width="550">

<img src="https://github.com/ducanhnguyen/restricted-boltzmann-machine/blob/master/img/rbm_4_reconstruction.png" width="550">
