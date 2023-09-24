# Original Source: https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras

import os
import tensorflow as tf
import numpy as np

def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the [0, 255] range.
    # You need to convert them to float32 with values in the [0, 1] range.
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset

def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model

def _is_main(task_type, task_id):
  return (task_type == 'worker' and task_id == 0) or task_type is None

def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir

def write_filepath(filepath, task_type, task_id):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_main(task_type, task_id):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)

def keras_model_path_to_tar(
    local_model_dir: str = '/model',
    local_tar_name = 'model.tar.gz'
):
    import tarfile
    with tarfile.open(local_tar_name, mode="w:gz") as _tar:
        _tar.add(local_model_dir, recursive=True)
    return local_tar_name

def main(
    per_worker_batch_size = 64, 
    epochs=25, 
    steps_per_epoch=70,
    num_workers=1,
    run=None,
    local_model_dir = '/tmp',
    local_tar_name = 'model.tar.gz'
):

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = mnist_dataset(global_batch_size)

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        model = build_and_compile_cnn_model()

    # The training state is backed up at epoch boundaries by default.
    # Restore the last checkpoint, and continue training from the beginning of the epoch 
        # and step at which the training state was last saved.
    callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir='/tmp/backup')]
    model.fit(multi_worker_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    print("Model training complete.")

    # save model
    task_type, task_id = (strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id)
    keras_model_path = write_filepath(local_model_dir, task_type, task_id)
    model.save(keras_model_path)

    # push tar file to s3
    if _is_main(task_type, task_id):
        from metaflow import S3
        tar_file = keras_model_path_to_tar(keras_model_path, local_tar_name)
        key = tar_file.split('/')[-1]
        s3 = S3(run=run)    
        s3.put_files([(key, tar_file)])

if __name__ == '__main__':
    main()