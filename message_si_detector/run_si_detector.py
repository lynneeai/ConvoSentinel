import argparse
import os
import json
import sys
sys.path.append("../")
from tqdm import tqdm

from utils import configure_gpu_device


parser = argparse.ArgumentParser()
# required arguments
parser.add_argument("--devices", required=True, type=str, help="GPU ids separated by comma, e.g. 0,1 or 1")
parser.add_argument("--model_checkpoint", required=True)
parser.add_argument("--input_file", required=True)
parser.add_argument("--output_file", required=True)
args = parser.parse_args()

configure_gpu_device(devices_str=args.devices)


from message_si_detector.pii_detector import PIIDetector

print(f"Setting up SI detector...")
base_model_path = "google/flan-t5-large"
pii_detector = PIIDetector(base_model_path=base_model_path, model_checkpoint=args.model_checkpoint)

if "/" in args.output_file:
    output_folder = "/".join(args.output_file.split("/")[:-1])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

print(f"Generating predictions for {args.input_file}...")
data = json.load(open(args.input_file, "r"))["Conversations"]
predictions = []
for sample in tqdm(data, dynamic_ncols=True):
    pred_obj = pii_detector.conversation_level_detection(sample, load_labels=False)
    predictions.append(pred_obj["pred_labels"])
    json.dump(predictions, open(args.output_file, "w"), indent=4)

print(f"Finished! Predictions saved to {args.output_file}.")