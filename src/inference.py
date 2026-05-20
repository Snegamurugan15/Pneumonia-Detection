from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]


def load_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def predict_image(model_path: str | Path, image_path: str | Path) -> dict:
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(load_image(image_path), verbose=0)[0]
    return {
        "prediction": CLASS_NAMES[int(np.argmax(probs))],
        "probabilities": {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)},
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pneumonia_vgg16.keras")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(predict_image(args.model, args.image))
