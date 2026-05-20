from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


IMG_SIZE = (224, 224)
CLASSES = ["NORMAL", "BACTERIA", "VIRUS"]


def build_transfer_model(num_classes: int = 3) -> tf.keras.Model:
    base = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


def make_dataset(data_dir: str, subset: str):
    return tf.keras.utils.image_dataset_from_directory(
        Path(data_dir) / subset,
        image_size=IMG_SIZE,
        batch_size=32,
        label_mode="categorical",
    )


def train(data_dir: str, epochs: int = 8):
    train_ds = make_dataset(data_dir, "train")
    val_ds = make_dataset(data_dir, "val")
    model = build_transfer_model(num_classes=len(train_ds.class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("models/pneumonia_vgg16.keras", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    ]
    Path("models").mkdir(exist_ok=True)
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)


if __name__ == "__main__":
    train("data/chest_xray")
