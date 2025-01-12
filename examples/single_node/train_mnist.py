import os
import tarfile
from metaflow import S3
import tensorflow as tf
import tensorflow_datasets as tfds


EPOCHS = 12
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64


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


def save_model_to_s3(
    local_model_dir="/tmp",
    local_tar_name="model.tar.gz",
    run=None,
):
    tar_file = keras_model_path_to_tar(local_model_dir, local_tar_name)
    key = tar_file.split("/")[-1]
    s3 = S3(run=run)
    s3.put_files([(key, tar_file)])


def keras_model_path_to_tar(
    local_model_dir: str = "/model",
    local_tar_name="model.tar.gz"
):
    with tarfile.open(local_tar_name, mode="w:gz") as _tar:
        _tar.add(local_model_dir, recursive=True)
    return local_tar_name


def evaluate_model(
    local_model_dir,
    eval_dataset,
    with_scope: bool,
    strategy,
):
    model_path = os.path.join(local_model_dir, "model.keras")

    if with_scope:
        with strategy.scope():
            replicated_model = tf.keras.models.load_model(model_path)
            replicated_model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"],
            )
        eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
        print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))
    else:
        unreplicated_model = tf.keras.models.load_model(model_path)
        unreplicated_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)
        print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))


def train_and_save_model(
    strategy,
    train_dataset,
    eval_dataset,
    checkpoint_dir="./training_checkpoints",
    local_model_dir="/tmp",
    local_tar_name="model.tar.gz",
    run=None,
):
    # Create necessary directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(local_model_dir, exist_ok=True)
    os.makedirs("/tmp/backup", exist_ok=True)

    # create the model and instantiate the optimizer
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

    # define the name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.keras")

    # define a callback for printing the learning rate at the end of each epoch
    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(
                "\nLearning rate for epoch {} is {}".format(
                    epoch + 1, model.optimizer.learning_rate.numpy()
                )
            )

    # put all the callbacks together
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=False
        ),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR(),
        tf.keras.callbacks.BackupAndRestore(backup_dir="/tmp/backup"),
    ]

    # train and evaluate
    model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)

    # restore the latest checkpoint
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Loading checkpoint from {latest_checkpoint}")
        model.load_weights(latest_checkpoint)

    # evaluate the model
    eval_loss, eval_acc = model.evaluate(eval_dataset)
    print("Eval loss: {}, Eval accuracy: {}".format(eval_loss, eval_acc))

    # save the model
    model_save_path = os.path.join(local_model_dir, "model.keras")
    model.save(model_save_path)
    save_model_to_s3(local_model_dir, local_tar_name, run)


def main(
    local_model_dir="/tmp",
    local_tar_name="model.tar.gz",
    checkpoint_dir="./training_checkpoints",
    run=None,
):
    # define the distribution strategy
    strategy = tf.distribute.MirroredStrategy()

    # download data
    datasets, _ = tfds.load(name="mnist", with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets["train"], datasets["test"]
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # set up input pipeline
    batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    train_dataset = (
        mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(batch_size)
    )
    eval_dataset = mnist_test.map(scale).batch(batch_size)

    train_and_save_model(
        strategy,
        train_dataset,
        eval_dataset,
        checkpoint_dir,
        local_model_dir,
        local_tar_name,
        run=run,
    )

    # load model without scope
    evaluate_model(
        local_model_dir,
        eval_dataset,
        with_scope=False,
        strategy=None
    )

    # load model with scope
    evaluate_model(
        local_model_dir,
        eval_dataset,
        with_scope=True,
        strategy=strategy
    )


if __name__ == "__main__":
    main()
