"""Error analysis and model visualization utilities.

Run examples:
        python weight_visualization.py
        python weight_visualization.py --model cnn --model-path ./best_models/cnn/best_model.pickle --output-dir ./figs
        python weight_visualization.py --model mlp --model-path ./best_models/mlp/best_model.pickle --output-dir ./figs
"""

from pathlib import Path
from struct import unpack
import argparse
import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


BASE_DIR = Path(__file__).resolve().parent
BEST_MODEL_DIR = BASE_DIR / "best_models"
MODEL_PATHS = {
        "mlp": BEST_MODEL_DIR / "without_lr_delay" / "mlp" / "best_model.pickle",
        "cnn": BEST_MODEL_DIR / "without_lr_delay" / "cnn" / "best_model.pickle",
}
LR_MODE_SUBDIRS = {
        "with": "with_lr_delay",
        "without": "without_lr_delay",
}
TEST_IMAGES_PATH = BASE_DIR / "dataset" / "MNIST" / "t10k-images-idx3-ubyte.gz"
TEST_LABELS_PATH = BASE_DIR / "dataset" / "MNIST" / "t10k-labels-idx1-ubyte.gz"
DEFAULT_OUTPUT_DIR = BASE_DIR / "figs"


def parse_args():
        parser = argparse.ArgumentParser(description="Generate confusion matrix and model visualizations.")
        parser.add_argument("--model", choices=["mlp", "cnn"], default="cnn", help="explicitly select the model type")
        parser.add_argument("--model-path", default=None, help="path to the saved model")
        parser.add_argument(
                "--lr",
                choices=["default", "with", "without"],
                default="without",
                help="which default checkpoint set to load when --model-path is not provided (default: without)",
        )
        parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="directory to save figures")
        parser.add_argument("--max-misclassified", type=int, default=9, help="maximum misclassified examples to plot")
        return parser.parse_args()


def resolve_default_model_path(model_name, lr_scheduler_mode="without"):
        if lr_scheduler_mode == "default":
                lr_scheduler_mode = "without"

        if lr_scheduler_mode == "without":
                return MODEL_PATHS[model_name]

        subdir = LR_MODE_SUBDIRS[lr_scheduler_mode]
        return BEST_MODEL_DIR / subdir / model_name / "best_model.pickle"


def load_mnist(images_path, labels_path):
        with gzip.open(images_path, 'rb') as f:
                magic, num, rows, cols = unpack('>4I', f.read(16))
                images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

        with gzip.open(labels_path, 'rb') as f:
                magic, num = unpack('>2I', f.read(8))
                labels = np.frombuffer(f.read(), dtype=np.uint8)

        images = images.astype(np.float32) / 255.0
        return images, labels


def load_any_model(model_path):
        with open(model_path, 'rb') as f:
                header = pickle.load(f)

        if isinstance(header, list) and len(header) > 0 and header[0] == 'Model_CNN':
                model = nn.models.Model_CNN()
        else:
                model = nn.models.Model_MLP()

        model.load_model(model_path)
        return model


def prepare_inputs(model, images):
        if len(model.layers) > 0 and isinstance(model.layers[0], nn.op.conv2D):
                return images.reshape(-1, 1, 28, 28)
        return images.reshape(images.shape[0], -1)


def predict(model, images, batch_size=256):
        inputs = prepare_inputs(model, images)
        logits_list = []
        for start in range(0, inputs.shape[0], batch_size):
                batch = inputs[start:start + batch_size]
                logits_list.append(model(batch))
        logits = np.concatenate(logits_list, axis=0)
        preds = np.argmax(logits, axis=1)
        return logits, preds


def confusion_matrix(labels, preds, num_classes=10):
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for label, pred in zip(labels, preds):
                matrix[label, pred] += 1
        return matrix


def plot_confusion_matrix(matrix, class_names=None):
        if class_names is None:
                class_names = [str(i) for i in range(matrix.shape[0])]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        threshold = matrix.max() * 0.6 if matrix.max() > 0 else 0
        for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                        color = 'white' if matrix[i, j] > threshold else 'black'
                        ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color=color, fontsize=8)

        ax.set_title('Confusion Matrix')
        fig.tight_layout()
        return fig, ax


def show_misclassified_examples(images, labels, preds, max_examples=25):
        wrong_idx = np.where(labels != preds)[0]
        if wrong_idx.size == 0:
                print('No misclassified examples found.')
                return None, None

        count = min(max_examples, wrong_idx.size)
        rows = int(np.ceil(np.sqrt(count)))
        cols = int(np.ceil(count / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
        axes = np.atleast_1d(axes).reshape(rows, cols)

        for idx in range(rows * cols):
                ax = axes.flat[idx]
                ax.axis('off')
                if idx < count:
                        sample_idx = wrong_idx[idx]
                        ax.imshow(images[sample_idx].reshape(28, 28), cmap='gray')
                        ax.set_title(f't:{labels[sample_idx]} p:{preds[sample_idx]}', fontsize=9)

        fig.suptitle('Misclassified Examples')
        fig.tight_layout()
        return fig, axes


def save_fig(fig, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches='tight')


def resolve_output_dir(base_output_dir, lr_scheduler_mode, model_type):
        if lr_scheduler_mode == 'default':
                lr_scheduler_mode = 'without'

        return Path(base_output_dir) / lr_scheduler_mode / model_type


def load_model(model_path, model_type=None):
        with open(model_path, 'rb') as f:
                header = pickle.load(f)

        saved_model_type = 'cnn' if isinstance(header, list) and len(header) > 0 and header[0] == 'Model_CNN' else 'mlp'
        if model_type is not None and model_type != saved_model_type:
                print(f'Warning: requested model type {model_type} does not match saved model type {saved_model_type}. Using the saved model structure.')

        model_type = saved_model_type

        if model_type == 'cnn':
                model = nn.models.Model_CNN()
        else:
                model = nn.models.Model_MLP()

        model.load_model(model_path)
        return model, model_type


def plot_mlp_weights(model):
        linear_layers = [layer for layer in model.layers if hasattr(layer, 'W') and getattr(layer.W, 'ndim', 0) == 2]
        if not linear_layers:
                return []

        figures = []

        first_layer = linear_layers[0]
        if first_layer.W.shape[0] == 28 * 28:
                count = min(16, first_layer.W.shape[1])
                rows = 4
                cols = int(np.ceil(count / rows))
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
                axes = np.atleast_1d(axes).reshape(rows, cols)
                for idx in range(rows * cols):
                        ax = axes.flat[idx]
                        ax.axis('off')
                        if idx < count:
                                ax.imshow(first_layer.W[:, idx].reshape(28, 28), cmap='gray')
                fig.suptitle('First MLP Layer Weights')
                fig.tight_layout()
                figures.append(fig)
        return figures


def plot_cnn_kernels(model, max_filters=16):
        figures = []
        conv_layers = [layer for layer in model.layers if isinstance(layer, nn.op.conv2D)]
        for layer_idx, conv in enumerate(conv_layers):
                if conv.W.shape[1] == 1:
                        kernels = conv.W[:, 0, :, :]
                else:
                        kernels = conv.W.mean(axis=1)

                count = min(max_filters, kernels.shape[0])
                rows = int(np.ceil(np.sqrt(count)))
                cols = int(np.ceil(count / rows))
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
                axes = np.atleast_1d(axes).reshape(rows, cols)

                for idx in range(rows * cols):
                        ax = axes.flat[idx]
                        ax.axis('off')
                        if idx < count:
                                ax.imshow(kernels[idx], cmap='gray')

                fig.suptitle(f'CNN Layer {layer_idx + 1} Kernels')
                fig.tight_layout()
                figures.append(fig)

        return figures


if __name__ == '__main__':
        args = parse_args()
        model_path = (
                Path(args.model_path)
                if args.model_path is not None
                else resolve_default_model_path(args.model, args.lr)
        )
        print(f'Loading model from: {model_path}')

        model, model_type = load_model(model_path, args.model)
        output_dir = resolve_output_dir(args.output_dir, args.lr, model_type)
        test_imgs, test_labs = load_mnist(TEST_IMAGES_PATH, TEST_LABELS_PATH)
        logits, preds = predict(model, test_imgs)
        cm = confusion_matrix(test_labs, preds, num_classes=int(test_labs.max()) + 1)

        fig, _ = plot_confusion_matrix(cm)
        save_fig(fig, output_dir / f'{model_type}_confusion_matrix.png')

        fig, _ = show_misclassified_examples(test_imgs, test_labs, preds, max_examples=args.max_misclassified)
        if fig is not None:
                save_fig(fig, output_dir / f'{model_type}_misclassified_examples.png')

        if model_type == 'mlp':
                figures = plot_mlp_weights(model)
                for idx, fig in enumerate(figures, start=1):
                        save_fig(fig, output_dir / f'mlp_weights_layer_{idx}.png')
        elif model_type == 'cnn':
                figures = plot_cnn_kernels(model)
                for idx, fig in enumerate(figures, start=1):
                        save_fig(fig, output_dir / f'cnn_kernels_layer_{idx}.png')

        plt.show()