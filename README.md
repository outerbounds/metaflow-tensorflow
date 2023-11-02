# Metaflow TensorFlow decorator
The [`tf.distribute.Strategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) allows TensorFlow developers to distribute model training to multiple GPUs/TPUs and machines. 

### Features

### Installation

Install this experimental module:
```
pip install metaflow-tensorflow
```

### Getting Started
This package will add a Metaflow extension to your already installed Metaflow, so you can use the `tensorflow` decorator.
```
from metaflow import FlowSpec, step, tensorflow, ...
```

The rest of this `README.md` file describes how you can use TensorFlow with Metaflow in the single node and multi-node cases which require `@tensorflow`.

# TensorFlow Distributed on Metaflow guide
The examples in this repository are based on the [original TensorFlow Examples](https://www.tensorflow.org/guide/distributed_training#examples_and_tutorials).

### Examples and guides

| Directory | TensorFlow script description |
| :--- | ---: |
| [MirroredStrategy](examples/single-node/README.md) | Synchronous distributed training on multiple GPUs on one machine. |  
| [MultiWorkerMirroredStrategy](examples/multi-node/README.md) | Synchronous distributed training across multiple workers, each with potentially multiple GPUs. |  

#### Parameter Server
Not yet tested, please reach out to the Outerbounds team if you need help.

#### Installing TensorFlow for GPU usage in Metaflow
> From [TensorFlow documentation](https://www.tensorflow.org/install/pip): Do not install TensorFlow with conda. It may not have the latest stable version. pip is recommended since TensorFlow is only officially released to PyPI.

We have found the easiest way to install TensorFlow for GPU is to use the pre-made Docker image `tensorflow/tensorflow:latest-gpu`.

#### Fault Tolerance
See [TensorFlow documentation on this matter](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#fault_tolerance).
The TL;DR is to use a flavor of `tf.distribute.Strategy`, which implement mechanisms to handle worker failures gracefully.

### License 
`metaflow-tensorflow` is distributed under the <u>Apache License</u>.