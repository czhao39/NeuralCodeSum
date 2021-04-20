import pickle
import sys


assert len(sys.argv) == 3

metrics = {"train": {"perplexity": [],
                     "ml_loss": []},
           "valid": {"bleu": [],
                     "rouge_l": [],
                     "precision": [],
                     "recall": [],
                     "f1": []}}

print(f"Reading from {sys.argv[1]}")
with open(sys.argv[1], "r") as infile:
    for line in infile:
        if "[ train: " in line:
            ppl_ind = line.index("perplexity")
            loss_ind = line.index("ml_loss")
            end_ind = line.index(" | Time")
            metrics["train"]["perplexity"].append(float(line[ppl_ind+13:loss_ind-3]))
            metrics["train"]["ml_loss"].append(float(line[loss_ind+10:end_ind]))
        elif "[ dev valid official: " in line:
            bleu_ind = line.index("bleu")
            rouge_ind = line.index("rouge_l")
            prec_ind = line.index("Precision")
            rec_ind = line.index("Recall")
            f1_ind = line.index("F1")
            end_ind = line.index(" | examples")
            metrics["valid"]["bleu"].append(float(line[bleu_ind+7:rouge_ind-3]))
            metrics["valid"]["rouge_l"].append(float(line[rouge_ind+10:prec_ind-3]))
            metrics["valid"]["precision"].append(float(line[prec_ind+12:rec_ind-3]))
            metrics["valid"]["recall"].append(float(line[rec_ind+9:f1_ind-3]))
            metrics["valid"]["f1"].append(float(line[f1_ind+5:end_ind]))

num_epochs = len(metrics["train"]["perplexity"])
assert len(metrics["valid"]["bleu"]) == num_epochs
print(f"Got metrics for {num_epochs} epochs")

with open(sys.argv[2], "wb") as outfile:
    pickle.dump(metrics, outfile)
print(f"Wrote to {sys.argv[2]}")
