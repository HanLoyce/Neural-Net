"""Compare test accuracies of MLP and CNN with and without LR scheduler.

Run:
    python compare_lr_modes.py
    python compare_lr_modes.py --save-figs
"""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

import test_model as tm
import mynn as nn


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "figs"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare MLP/CNN accuracies with/without LR scheduler")
    parser.add_argument("--save-figs", action="store_true", help="save comparison figures to figs/")
    return parser.parse_args()


def evaluate_model(model_name, lr_mode):
    model_path = tm.resolve_default_model_path(model_name, lr_mode)
    print(f"Loading {model_name} ({lr_mode}) from: {model_path}")
    model, resolved = tm.load_model(model_path, model_name)

    test_images_path = tm.DATA_DIR / "t10k-images-idx3-ubyte.gz"
    test_labels_path = tm.DATA_DIR / "t10k-labels-idx1-ubyte.gz"
    test_imgs, test_labs = tm.load_mnist(test_images_path, test_labels_path)

    test_imgs = test_imgs / test_imgs.max()
    if resolved == "cnn":
        test_imgs = test_imgs.reshape(-1, 1, 28, 28)

    logits = model(test_imgs)
    acc = float(nn.metric.accuracy(logits, test_labs))
    return acc


def main():
    args = parse_args()

    models = ["mlp", "cnn"]
    modes = ["with", "without"]

    results = {m: {} for m in models}
    for m in models:
        for mode in modes:
            try:
                acc = evaluate_model(m, mode)
            except Exception as e:
                print(f"Error evaluating {m} ({mode}): {e}")
                acc = float('nan')
            results[m][mode] = acc

    # Print table
    print("\nTest accuracy table:")
    print("Model\twith\twithout")
    for m in models:
        print(f"{m}\t{results[m]['with']:.4f}\t{results[m]['without']:.4f}")

    # Plot grouped bars
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    with_vals = [results[m]["with"] for m in models]
    without_vals = [results[m]["without"] for m in models]

    bars1 = ax.bar(x - width/2, without_vals, width, label='without LR', color='#4C72B0')
    bars2 = ax.bar(x + width/2, with_vals, width, label='with LR', color='#DD8452')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_title('MLP vs CNN: with vs without LR scheduler')
    ax.legend()

    for bar, val in zip(bars1 + bars2, list(without_vals) + list(with_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'compare_lr_modes.png'
    if args.save_figs:
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
        print(f"Saved figure to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
