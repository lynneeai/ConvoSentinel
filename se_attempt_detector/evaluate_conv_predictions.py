import argparse
import json
from sklearn.metrics import classification_report


parser = argparse.ArgumentParser()
parser.add_argument("--prediction_file", type=str, required=True)
args = parser.parse_args()

LABEL_MAP = {
    "benign": 0,
    "malicious": 1
}

objs = json.load(open(args.prediction_file, "r"))
preds, labels = [], []
for obj in objs:
    preds.append(LABEL_MAP[obj["prediction"]])
    labels.append(LABEL_MAP[obj["label"]])
    
print(classification_report(labels, preds, target_names=["benign", "malicious"]))