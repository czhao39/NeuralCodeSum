import pickle
import sys

import matplotlib.pyplot as plt


print(f"Reading initial from {sys.argv[1]}")
with open(sys.argv[1], "rb") as infile:
    normal_metrics = pickle.load(infile)

sep = None
if len(sys.argv) > 2:
    sep = len(normal_metrics["train"]["perplexity"])
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

# Only plot starting from epoch 120
epochs = epochs[120:]
for m in normal_metrics["train"].keys():
    normal_metrics["train"][m] = normal_metrics["train"][m][120:]
for m in normal_metrics["valid"].keys():
    normal_metrics["valid"][m] = normal_metrics["valid"][m][120:]

fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# Plot training metrics
ax = axs[0, 0]
ax.plot(epochs, normal_metrics["train"]["perplexity"], label="perplexity w/ CE")
ax.set_ylabel("perplexity")
if sep is not None:
    new_epochs = list(range(sep, num_epochs))
    ax.plot(new_epochs, new_focal_metrics["train"]["perplexity"], linestyle="--", label="perplexity w/ focal")
    ax.axvline(sep, color="black", linestyle=":")
ax.set_xlabel("Epoch")
ax.set_title("Training Perplexity")
ax.legend()

# Plot validation precision, recall, F1
ax = axs[0, 1]
ax.plot(epochs, normal_metrics["valid"]["precision"], label="precision w/ CE")
ax.plot(epochs, normal_metrics["valid"]["recall"], label="recall w/ CE")
ax.plot(epochs, normal_metrics["valid"]["f1"], label="F1 w/ CE")
if sep is not None:
    ax.plot(new_epochs, new_focal_metrics["valid"]["precision"], linestyle="--", label="precision w/ focal")
    ax.plot(new_epochs, new_focal_metrics["valid"]["recall"], linestyle="--", label="recall w/ focal")
    ax.plot(new_epochs, new_focal_metrics["valid"]["f1"], linestyle="--", label="F1 w/ focal")
    ax.axvline(sep, color="black", linestyle=":")
ax.set_xlabel("Epoch")
ax.set_title("Validation Precision, Recall, and F1")
ax.legend()

# Plot validation BLEU
ax = axs[1, 0]
ax.plot(epochs, normal_metrics["valid"]["bleu"], label="BLEU w/ CE")
if sep is not None:
    ax.plot(new_epochs, new_focal_metrics["valid"]["bleu"], linestyle="--", label="BLEU w/ focal")
    ax.axvline(sep, color="black", linestyle=":")
ax.set_xlabel("Epoch")
ax.set_ylabel("BLEU")
ax.set_title("Validation BLEU Score")
ax.legend()

# Plot validation ROUGE-L
ax = axs[1, 1]
ax.plot(epochs, normal_metrics["valid"]["rouge_l"], label="ROUGE-L w/ CE")
if sep is not None:
    ax.plot(new_epochs, new_focal_metrics["valid"]["rouge_l"], linestyle="--", label="ROUGE-L w/ focal")
    ax.axvline(sep, color="black", linestyle=":")
ax.set_xlabel("Epoch")
ax.set_ylabel("ROUGE-L")
ax.set_title("Validation ROUGE-L Score")
ax.legend()

plt.tight_layout()
plt.savefig("metrics.png", dpi=400, bbox_inches="tight")
plt.show()
