import pickle
import sys

import matplotlib.pyplot as plt


print(f"Reading initial from {sys.argv[1]}")
with open(sys.argv[1], "rb") as infile:
    normal_metrics = pickle.load(infile)

sep = None
if len(sys.argv) > 2:
    sep = len(normal_metrics["train"]["perplexity"]) - 1
    print(f"Reading normal from {sys.argv[2]}")
    with open(sys.argv[2], "rb") as infile:
        new_normal_metrics = pickle.load(infile)
    for m, l in normal_metrics["train"].items():
        l.extend(new_normal_metrics["train"][m])
    for m, l in normal_metrics["valid"].items():
        l.extend(new_normal_metrics["valid"][m])

    print(f"Reading focal from {sys.argv[3]}")
    with open(sys.argv[3], "rb") as infile:
        new_focal_metrics = pickle.load(infile)


num_epochs = len(normal_metrics["train"]["perplexity"])
epochs = list(range(num_epochs))
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

axs[0].plot(epochs, normal_metrics["train"]["perplexity"], color="red")
axs[0].set_ylabel("perplexity", color="red")
ax2 = axs[0].twinx()
ax2.plot(epochs, normal_metrics["train"]["ml_loss"], color="blue")
ax2.set_ylabel("ml_loss", color="blue", rotation=-90, va="bottom")
if sep is not None:
    new_epochs = list(range(sep + 1, num_epochs))
    axs[0].plot(new_epochs, new_focal_metrics["train"]["perplexity"], linestyle="--", color="red")
    ax2.plot(new_epochs, new_focal_metrics["train"]["ml_loss"], linestyle="--", color="blue")
axs[0].set_title("Training Metrics")

for m, l in normal_metrics["valid"].items():
    axs[1].plot(epochs, l, label=m)
    if sep is not None:
        axs[1].plot(new_epochs, new_focal_metrics["valid"][m], linestyle="--", label=f"{m} w/ focal loss")
axs[1].legend()
axs[1].set_title("Validation Metrics")

axs[0].axvline(sep)
axs[1].axvline(sep)

plt.xlabel("Epoch")
plt.tight_layout()
plt.savefig("metrics.png", dpi=200, bbox_inches="tight")
plt.show()
