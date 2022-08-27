import json

from datasets import load_dataset
import re


dataset_path = "PiC/phrase_similarity"
data_list = [example for example in load_dataset(dataset_path)["test"]]

file_names = ["predictions_contextual", "predictions_non_contextual"]

for file_name in file_names:

    # False positives + False negatives
    fps, fns = [], []

    with open("../results/{}.txt".format(file_name), "r") as input_file:
        for line in input_file:
            # Line pattern: "Idx 0: GT: 0.0 -- Pred: 1.0 -- Conf: 0.6496"
            match = re.findall('Idx (.+?): GT: (.+?) -- Pred: (.+?) -- Conf: ([0-9][.][0-9]+)', line.strip())
            match = [float(num) for num in match[0]] if match else None
            idx, gt, pred, sigmoid_score = int(match[0]), match[1], match[2], match[3]
            example = data_list[idx]
            example["label"] = "positive" if example["label"] == 1 else "negative"
            example["sigmoid_score"] = sigmoid_score

            if pred != gt:
                if pred == 1.0:
                    fps.append(example)
                else:
                    fns.append(example)

    fps = sorted(fps, key=lambda d: d["sigmoid_score"], reverse=True)
    fns = sorted(fns, key=lambda d: d["sigmoid_score"], reverse=False)

    with open("../results/{}_incorrect_FPs.json".format(file_name), "w") as output_file:
        json.dump(fps, output_file, indent=4)

    with open("../results/{}_incorrect_FNs.json".format(file_name), "w") as output_file:
        json.dump(fns, output_file, indent=4)