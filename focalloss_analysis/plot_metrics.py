import pickle
import sys

import matplotlib.pyplot as plt


print(f"Reading from {sys.argv[1]}")
with open(sys.argv[1], "rb") as infile:
    metrics = pickle.load(infile)

# Epochs separating runs
seps = []

if len(sys.argv) > 2:
    for f in sys.argv[2:]:
        seps.append(len(metrics["train"]["perplexity"]))
        print(f"Reading from {f}")
        with open(f, "rb") as infile:
            new_metrics = pickle.load(infile)
            for m, l in metrics["train"].items():
                l.extend(new_metrics["train"][m])
            for m, l in metrics["valid"].items():
                l.extend(new_metrics["valid"][m])


num_epochs = len(metrics["train"]["perplexity"])
epochs = list(range(num_epochs))
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

axs[0].plot(epochs, metrics["train"]["perplexity"], color="red")
axs[0].set_ylabel("perplexity", color="red")
ax2 = axs[0].twinx()
ax2.plot(epochs, metrics["train"]["ml_loss"], color="blue")
ax2.set_ylabel("ml_loss", color="blue", rotation=-90, va="bottom")
axs[0].set_title("Training Metrics")

for m, l in metrics["valid"].items():
    axs[1].plot(epochs, l, label=m)
axs[1].legend()
axs[1].set_title("Validation Metrics")

for sep in seps:
    axs[0].axvline(sep)
    axs[1].axvline(sep)

plt.xlabel("Epoch")
plt.tight_layout()
plt.savefig("metrics.png", dpi=200, bbox_inches="tight")
plt.show()
