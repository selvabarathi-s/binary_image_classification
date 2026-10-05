import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Train a binary image classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("PetImages"),
        help="Dataset folder containing one subfolder per class, e.g. Cat and Dog.",
    )
    parser.add_argument("--image-size", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models") / "binary_image_classifier.keras",
        help="Where the trained model will be saved for future use.",
    )
    parser.add_argument(
        "--class-names-path",
        type=Path,
        default=Path("models") / "class_names.json",
        help="Where class names will be saved and loaded.",
    )
    parser.add_argument(
        "--predict",
        type=Path,
        default=None,
        help="Optional image path to classify after training or loading a saved model.",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load --model-path instead of training a new model. Useful with --predict.",
    )
    return parser.parse_args()


def remove_corrupted_images(data_dir):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    removed = 0

    for image_path in data_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in valid_extensions:
            continue

        try:
            image_bytes = tf.io.read_file(str(image_path))
            tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        except Exception:
            image_path.unlink()
            removed += 1

    if removed:
        print(f"Removed {removed} corrupted image(s).")


def load_datasets(data_dir, image_size, batch_size, validation_split, seed):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="binary",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="binary",
    )

    class_names = train_ds.class_names
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(image_size):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_image(model, image_path, image_size, class_names):
    image = tf.keras.utils.load_img(image_path, target_size=(image_size, image_size))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    probability = float(model.predict(image_array, verbose=0)[0][0])
    predicted_index = int(probability >= 0.5)
    confidence = probability if predicted_index == 1 else 1.0 - probability

    print(f"Prediction: {class_names[predicted_index]} ({confidence:.2%} confidence)")


def save_class_names(class_names, class_names_path):
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with class_names_path.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)


def load_class_names(class_names_path):
    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class names file not found: {class_names_path}. Train the model first."
        )

    with class_names_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    args = parse_args()

    if args.load_model:
        model = tf.keras.models.load_model(args.model_path)
        class_names = load_class_names(args.class_names_path)
    else:
        if not args.data_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {args.data_dir}")

        remove_corrupted_images(args.data_dir)
        train_ds, val_ds, class_names = load_datasets(
            args.data_dir,
            args.image_size,
            args.batch_size,
            args.validation_split,
            args.seed,
        )

        model = build_model(args.image_size)
        model.summary()
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model_path)
        save_class_names(class_names, args.class_names_path)
        print(f"Saved model to {args.model_path}")
        print(f"Saved class names to {args.class_names_path}")
        print(f"Class order: {class_names}")

    if args.predict:
        predict_image(model, args.predict, args.image_size, class_names)


if __name__ == "__main__":
    main()
