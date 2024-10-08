
import sys
sys.path.append("../..")
import argparse
import os
import json
from datetime import datetime

from utils import configure_gpu_device
from message_si_detector.train_si_detector.si_detector_training_config import Training_Config


parser = argparse.ArgumentParser()
# required arguments
parser.add_argument("--devices", required=True, type=str, help="GPU ids separated by comma, e.g. 0,1 or 1")
# optional arguments
parser.add_argument("--model_path", type=str, default="google/flan-t5-base")
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--eval_batch_size", type=int, default=4)
parser.add_argument("--eval_steps", type=int, default=100)
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--output_dir", default="./model_outputs")
args = parser.parse_args()

configure_gpu_device(devices_str=args.devices)


##########################
## set and save configs ##
##########################
now_dt = datetime.now()
timestamp = now_dt.strftime("%m-%d-%y-%H:%M")

training_config = Training_Config(
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    fp16=args.fp16,
    output_dir=f"{args.output_dir}/{args.model_path.split('/')[-1]}_{timestamp}",
    logging_dir=f"{args.output_dir}/tensorboard_logs/{args.model_path.split('/')[-1]}_{timestamp}"
)

if not os.path.exists(training_config.output_dir):
    os.makedirs(training_config.output_dir)
json.dump(training_config.to_dict(), open(f"{training_config.output_dir}/config.json", "w"), indent=4)


#####################
## package imports ##
#####################
"""
The following imports needs to be done after configuring gpu devices 
so torch will only see specified devices
"""
import inspect
from tqdm import tqdm
from torch.utils.data import random_split
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, ProgressCallback

from message_si_detector.se_datasets import PIIDetectorMessageLevelDataset
from message_si_detector.pii_detector import PIIDetector
from message_si_detector.train_si_detector.si_evaluator import SIEvaluator


############################
## prepare data and model ##
############################
print("Loading data...")
dataset = PIIDetectorMessageLevelDataset(args.model_path)
train_data, val_data = random_split(dataset.train_dataset, [0.9, 0.1])

print(f"Initializing model {args.model_path}...")
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, device_map="auto")
model.resize_token_embeddings(len(dataset.tokenizer))
model.config.bos_token_id = dataset.tokenizer.bos_token_id
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params}")


#####################
## prepare trainer ##
#####################
class NoLossLoggingInTerminalCallback(ProgressCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Overrides parent class to disable loss string printing in terminal.
        """
        if state.is_world_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            
training_args = {
    key: val for key, val in training_config.to_dict().items()
    if key in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
}
training_args = Seq2SeqTrainingArguments(**training_args)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=dataset.collate_fn,
    train_dataset=train_data,
    eval_dataset=val_data
)
trainer.callback_handler.remove_callback(ProgressCallback)
trainer.callback_handler.add_callback(NoLossLoggingInTerminalCallback)


###########
## train ##
###########
print("Training...")
trainer.train()


##########
## test ##
##########
print("Testing...")
trainer._load_best_model()
pii_detector = PIIDetector(base_model_path=args.model_path, model=trainer.model)
pii_evaluator = SIEvaluator("cuda:0") # gpu is configured to have only 1 device
test_data = json.load(open("../data/v3/labeled_testV3.json", "r"))["Conversations"]

pred_obj_list = []
for sample in tqdm(test_data, dynamic_ncols=True):
    pred_obj = pii_detector.conversation_level_detection(sample)["pred_labels"]
    pred_obj_list.append(pred_obj)

print("Evaluating...")
scores = pii_evaluator.evaluate(pred_obj_list, test_data)
with open(f"{training_config.output_dir}/test_results.txt", "w") as outfile:
    outfile.write(f"{scores['classification_report']}\n")
    outfile.write(f"Message level PII name cosine similarity: {scores['message_level_cosines']}\n")
    outfile.write(f"Conversation level PII name cosine similarity: {scores['conversation_level_cosines']}\n")