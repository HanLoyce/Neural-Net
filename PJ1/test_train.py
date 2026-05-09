"""Train a model and save the learning curve to figs/.

Run examples:
        python test_train.py --model cnn
        python test_train.py --model mlp
"""

from pathlib import Path
import argparse
import gzip
import pickle
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from draw_tools.plot import plot


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "MNIST"
BEST_MODEL_DIR = BASE_DIR / "best_models"
FIGS_DIR = BASE_DIR / "figs"
MODEL_BEST_MODEL_DIRS_WITH = {
        "mlp": BEST_MODEL_DIR / "with_lr_delay" / "mlp",
        "cnn": BEST_MODEL_DIR / "with_lr_delay" / "cnn",
}
MODEL_BEST_MODEL_DIRS_WITHOUT = {
        "mlp": BEST_MODEL_DIR / "without_lr_delay" / "mlp",
        "cnn": BEST_MODEL_DIR / "without_lr_delay" / "cnn",
}
DEFAULT_EPOCHS_WITH_SCHEDULER = {
        "mlp": 15,
        "cnn": 15,
}
DEFAULT_EPOCHS_WITHOUT_SCHEDULER = 15


def parse_args():
        parser = argparse.ArgumentParser(description="Train an MLP or CNN on MNIST.")
        parser.add_argument("--model", choices=["mlp", "cnn"], default="cnn", help="model architecture to train")
        parser.add_argument(
                "--epochs",
                type=int,
                default=None,
                help="number of training epochs (auto: no scheduler=15, with scheduler mlp=15/cnn=15)",
        )
        parser.add_argument("--log-iters", type=int, default=100, help="logging interval")
        parser.add_argument("--save-dir", default=None, help="directory for the best checkpoint")
        scheduler_group = parser.add_mutually_exclusive_group()
        scheduler_group.add_argument(
                "--use-lr-scheduler",
                dest="use_lr_scheduler",
                action="store_true",
                help="enable learning-rate scheduler",
        )
        scheduler_group.add_argument(
                "--no-lr-scheduler",
                dest="use_lr_scheduler",
                action="store_false",
                help="disable learning-rate scheduler",
        )
        parser.set_defaults(use_lr_scheduler=False)
        return parser.parse_args()


def load_mnist(images_path, labels_path):
        with gzip.open(images_path, "rb") as f:
                magic, num, rows, cols = unpack(">4I", f.read(16))
                images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

        with gzip.open(labels_path, "rb") as f:
                magic, num = unpack(">2I", f.read(8))
                labels = np.frombuffer(f.read(), dtype=np.uint8)

        return images, labels


def build_model(model_name, train_imgs, train_labs, use_lr_scheduler=True):
        num_classes = int(train_labs.max()) + 1
        if model_name == "mlp":
                model = nn.models.Model_MLP([train_imgs.shape[-1], 600, num_classes], "ReLU", [1e-4, 1e-4])
                optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
                scheduler = None
                if use_lr_scheduler:
                        scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=6, gamma=0.5)
                loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=num_classes)
                return model, optimizer, scheduler, loss_fn, train_imgs, False

        train_imgs = train_imgs.reshape(-1, 1, 28, 28)
        model = nn.models.Model_CNN(lambda_list=[1e-4, 1e-4, 1e-4])
        optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
        scheduler = None
        if use_lr_scheduler:
                scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=6, gamma=0.5)
        loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=num_classes)
        return model, optimizer, scheduler, loss_fn, train_imgs, True


def resolve_default_save_dir(model_name, use_lr_scheduler):
        if use_lr_scheduler:
                return MODEL_BEST_MODEL_DIRS_WITH[model_name]
        return MODEL_BEST_MODEL_DIRS_WITHOUT[model_name]


def main():
        args = parse_args()

        np.random.seed(309)

        train_imgs_path = DATA_DIR / "train-images-idx3-ubyte.gz"
        train_labels_path = DATA_DIR / "train-labels-idx1-ubyte.gz"
        train_imgs, train_labs = load_mnist(train_imgs_path, train_labels_path)

        idx = np.random.permutation(np.arange(train_imgs.shape[0]))
        with open(BASE_DIR / "idx.pickle", "wb") as f:
                pickle.dump(idx, f)

        train_imgs = train_imgs[idx]
        train_labs = train_labs[idx]
        valid_imgs = train_imgs[:10000]
        valid_labs = train_labs[:10000]
        train_imgs = train_imgs[10000:]
        train_labs = train_labs[10000:]

        train_imgs = train_imgs / train_imgs.max()
        valid_imgs = valid_imgs / valid_imgs.max()

        model, optimizer, scheduler, loss_fn, train_imgs, is_cnn = build_model(
                args.model,
                train_imgs,
                train_labs,
                use_lr_scheduler=args.use_lr_scheduler,
        )

        if args.epochs is not None:
                num_epochs = args.epochs
        elif args.use_lr_scheduler:
                num_epochs = DEFAULT_EPOCHS_WITH_SCHEDULER[args.model]
        else:
                num_epochs = DEFAULT_EPOCHS_WITHOUT_SCHEDULER

        if is_cnn:
                valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

        save_dir = (
                Path(args.save_dir)
                if args.save_dir is not None
                else resolve_default_save_dir(args.model, args.use_lr_scheduler)
        )

        runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
        runner.train(
                [train_imgs, train_labs],
                [valid_imgs, valid_labs],
                num_epochs=num_epochs,
                log_iters=args.log_iters,
                save_dir=str(save_dir),
        )

        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, 2)
        axes = axes.reshape(-1)
        figure.set_tight_layout(True)
        plot(runner, axes)

        fig_path = FIGS_DIR / f"train_curve_{args.model}.png"
        figure.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
        main()