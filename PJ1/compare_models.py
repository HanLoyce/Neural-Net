"""Compare saved MLP and CNN test results and plot accuracy comparison.

Run:
    python compare_models.py
    python compare_models.py --lr with
"""
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

import test_model as tm
import mynn as nn


BASE_DIR = Path(__file__).resolve().parent
COMPARE_DIR = BASE_DIR / "figs"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare MLP and CNN test accuracy.")
    parser.add_argument(
        "--lr",
        choices=["default", "with", "without"],
        default="without",
        help="which default checkpoint set to load (default: without)",
    )
    return parser.parse_args()


def evaluate_model(model_name, lr_scheduler_mode):
    model_path = tm.resolve_default_model_path(model_name, lr_scheduler_mode)
    model, resolved = tm.load_model(model_path, model_name)

    test_images_path = tm.DATA_DIR / "t10k-images-idx3-ubyte.gz"
    test_labels_path = tm.DATA_DIR / "t10k-labels-idx1-ubyte.gz"
    test_imgs, test_labs = tm.load_mnist(test_images_path, test_labels_path)

    test_imgs = test_imgs / test_imgs.max()
    if resolved == "cnn":
        test_imgs = test_imgs.reshape(-1, 1, 28, 28)

    logits = model(test_imgs)
    acc = nn.metric.accuracy(logits, test_labs)
    return float(acc)


def main():
    args = parse_args()
    models = ["mlp", "cnn"]
    accs = []
    for model_name in models:
        print(f"Evaluating {model_name} (lr_mode={args.lr})...")
        acc = evaluate_model(model_name, args.lr)
        print(f"{model_name} accuracy: {acc:.4f}")
        accs.append(acc)

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    if args.lr in {"default", "without"}:
        out_path = COMPARE_DIR / "compare_accuracy.png"
    else:
        out_path = COMPARE_DIR / f"compare_accuracy_{args.lr}_lr.png"

    x = np.arange(len(models))
    plt.figure(figsize=(6, 4.5))
    bars = plt.bar(x, accs, color=["#4C72B0", "#DD8452"])
    plt.xticks(x, models)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("MLP vs CNN Test Accuracy", pad=12, fontsize=12)

    for bar, acc in zip(bars, accs):
        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = acc + 0.02
        plt.text(
            x_pos,
            y_pos,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
