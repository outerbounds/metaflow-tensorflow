# Introduction

The following four files showcase how to leverage tensorflow's `MirroredStrategy` with `@kubernetes`. This enables distributed training on multiple GPUs of a single machine. Note that it doesn't use the `@tensorflow` decorator.

1. `gpu_profile.py` contains the `@gpu_profile` decorator, and is available [here](https://github.com/outerbounds/metaflow-gpu-profile). It is used in the file `flow.py`

2. `train_mnist.py` contains the main snippet for how to use the `MirroredStrategy` while training a model on the MNIST dataset.

3. `flow.py` contains a flow that uses the training code from `train_mnist.py` and uses the docker image `tensorflow/tensorflow:2.15.0-gpu` for GPU setup.

- This can be run using `python flow.py --environment=pypi run`
- If you are on the [Outerbounds](https://outerbounds.com/) platform, you can leverage `fast-bakery` for blazingly fast docker image builds. This can be used by `python flow.py --environment=fast-bakery run`

4. `reload.ipynb` showcases how to use the trained model for inference later on. Please make sure to have `tensorflow==2.15.1` installed locally to be able to run this notebook correctly.
