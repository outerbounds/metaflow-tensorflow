# Getting started
Install this experimental module:
```
pip install metaflow-tensorflow
```

This package will add a Metaflow extension to your already installed Metaflow, so you can use the `tensorflow_parallel` decorator.
```
from metaflow import FlowSpec, step, tensorflow_parallel, ...
```

The rest of this `README.md` file describes how you can use TensorFlow with Metaflow in the single node and multi-node cases which require `@tensorflow_parallel`.

# TensorFlow Distributed on Metaflow guide
The examples in this repository are based on the [original TensorFlow Examples](https://www.tensorflow.org/guide/distributed_training#examples_and_tutorials).

## Important notes

### Installing TensorFlow for GPU usage in Metaflow
> From [TensorFlow documentation](https://www.tensorflow.org/install/pip): Do not install TensorFlow with conda. It may not have the latest stable version. pip is recommended since TensorFlow is only officially released to PyPI.

We have found the easiest way to install TensorFlow for GPU is to use the pre-made Docker image `tensorflow/tensorflow:latest-gpu`.

### Fault Tolerance
See [TensorFlow documentation on this matter](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#fault_tolerance).
The TL;DR is to use a flavor of `tf.distribute.Strategy`, which implement mechanisms to handle worker failures gracefully.

## MirroredStrategy
Synchronous distributed training on multiple GPUs on one machine.

```
python single-node/flow.py run
```

## MultiWorkerMirroredStrategy
Synchronous distributed training across multiple workers, each with potentially multiple GPUs.

```
python multi-node/flow.py run
```

## Parameter Server
Not yet implemented, please reach out to the Outerbounds team if you need this.