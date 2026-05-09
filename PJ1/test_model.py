"""Evaluate a saved model on the MNIST test set.

Run examples:
        python test_model.py --model cnn
        python test_model.py --model mlp
"""

from pathlib import Path
import argparse
import gzip
import pickle
from struct import unpack

import mynn as nn
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "MNIST"
BEST_MODEL_DIR = BASE_DIR / "best_models"
MODEL_BEST_MODEL_PATHS = {
        "mlp": BEST_MODEL_DIR / "without_lr_delay" / "mlp" / "best_model.pickle",
        "cnn": BEST_MODEL_DIR / "without_lr_delay" / "cnn" / "best_model.pickle",
}
LR_MODE_SUBDIRS = {
        "with": "with_lr_delay",
        "without": "without_lr_delay",
}


def parse_args():
        parser = argparse.ArgumentParser(description="Evaluate a trained MLP or CNN.")
        parser.add_argument("--model", choices=["mlp", "cnn"], default="cnn", help="model architecture to load")
        parser.add_argument("--model-path", default=None, help="path to the saved model")
        parser.add_argument(
                "--lr",
                choices=["default", "with", "without"],
                default="without",
                help="which default checkpoint set to load when --model-path is not provided (default: without)",
        )
        return parser.parse_args()


def resolve_default_model_path(model_name, lr_scheduler_mode="without"):
        if lr_scheduler_mode == "default":
                lr_scheduler_mode = "without"

        if lr_scheduler_mode == "without":
                return MODEL_BEST_MODEL_PATHS[model_name]

        subdir = LR_MODE_SUBDIRS[lr_scheduler_mode]
        return BEST_MODEL_DIR / subdir / model_name / "best_model.pickle"


def load_model(model_path, model_name=None):
        if model_name is None:
                with open(model_path, "rb") as f:
                        header = pickle.load(f)

                if isinstance(header, list) and len(header) > 0 and header[0] == "Model_CNN":
                        model_name = "cnn"
                else:
                        model_name = "mlp"

        if model_name == "cnn":
                model = nn.models.Model_CNN()
        else:
                model = nn.models.Model_MLP()

        model.load_model(model_path)
        return model, model_name


def load_mnist(images_path, labels_path):
        with gzip.open(images_path, "rb") as f:
                magic, num, rows, cols = unpack(">4I", f.read(16))
                images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

        with gzip.open(labels_path, "rb") as f:
                magic, num = unpack(">2I", f.read(8))
                labels = np.frombuffer(f.read(), dtype=np.uint8)

        return images, labels


def main():
        args = parse_args()

        model_path = (
                Path(args.model_path)
                if args.model_path is not None
                else resolve_default_model_path(args.model, args.lr)
        )
        print(f"Loading model from: {model_path}")

        model, resolved_model_name = load_model(model_path, args.model)
        test_images_path = DATA_DIR / "t10k-images-idx3-ubyte.gz"
        test_labels_path = DATA_DIR / "t10k-labels-idx1-ubyte.gz"
        test_imgs, test_labs = load_mnist(test_images_path, test_labels_path)

        test_imgs = test_imgs / test_imgs.max()
        if resolved_model_name == "cnn":
                test_imgs = test_imgs.reshape(-1, 1, 28, 28)

        logits = model(test_imgs)
        print(nn.metric.accuracy(logits, test_labs))


if __name__ == "__main__":
        main()