import os
import tensorflow as tf

try:
    import tensorflow_datasets as tfds
except ImportError:
    import os
    import subprocess

    with open(os.devnull, "wb") as devnull:
        subprocess.check_call(
            ["pip", "install", "tensorflow-datasets"],
            stdout=devnull,
            stderr=subprocess.STDOUT,
        )
    import tensorflow_datasets as tfds


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


def keras_model_path_to_tar(
    local_model_dir: str = "/model", local_tar_name="model.tar.gz"
):
    import tarfile

    with tarfile.open(local_tar_name, mode="w:gz") as _tar:
        _tar.add(local_model_dir, recursive=True)
    return local_tar_name


def main(
    checkpoint_dir="./training_checkpoints",
    local_model_dir="/tmp",
    local_tar_name="model.tar.gz",
    run=None,
):
    # download data
    datasets, info = tfds.load(name="mnist", with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets["train"], datasets["test"]

    # Define the distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Set up the input pipeline
    num_train_examples = info.splits["train"].num_examples
    num_test_examples = info.splits["test"].num_examples

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    train_dataset = (
        mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    )
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

    # Create the model and instantiate the optimizer
    with strategy.scope():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, 3, activation="relu", input_shape=(28, 28, 1)
                ),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )

    # Define the name of the checkpoint files.
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    # Define a callback for printing the learning rate at the end of each epoch.
    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(
                "\nLearning rate for epoch {} is {}".format(
                    epoch + 1, model.optimizer.lr.numpy()
                )
            )

    # Put all the callbacks together.
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True
        ),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR(),
        tf.keras.callbacks.BackupAndRestore(backup_dir="/tmp/backup"),
    ]

    # Train and evaluate
    EPOCHS = 12
    model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)

    checkpoints_dir_res = os.listdir(checkpoint_dir)
    print(checkpoints_dir_res)

    # Restore the latest checkpoint
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    eval_loss, eval_acc = model.evaluate(eval_dataset)
    print("Eval loss: {}, Eval accuracy: {}".format(eval_loss, eval_acc))

    # Save the model
    model.save(local_model_dir)

    # Zip the model dir and send to S3 for future use
    from metaflow import S3

    tar_file = keras_model_path_to_tar(local_model_dir, local_tar_name)
    key = tar_file.split("/")[-1]
    s3 = S3(run=run)
    s3.put_files([(key, tar_file)])

    # load model without scope
    unreplicated_model = tf.keras.models.load_model(local_model_dir)

    unreplicated_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)
    print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))

    # load model with scope
    with strategy.scope():
        replicated_model = tf.keras.models.load_model(local_model_dir)
        replicated_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
        print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))


if __name__ == "__main__":
    main()
