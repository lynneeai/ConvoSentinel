import json
from collections import defaultdict
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from message_si_detector.prompters import PIIDetectorPrompter


class PIIDetectorMessageLevelDataset(object):
    def __init__(self, model_path, train_file="../data/v3/labeled_trainV3.json"):
        
        self.prompter = PIIDetectorPrompter()
        self._setup_tokenizer(model_path)
        
        self.train_file = train_file
        
    def _setup_tokenizer(self, model_path):
        print(f"Setting up tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens({"bos_token": "<s>"})
    
    def _process_conversation(self, conv, load_labels=True):
        messages = conv["Conversation"]
        idx2piis = defaultdict(list)
        
        if load_labels:
            annotation = conv["GroundTruth"]
            for pii in annotation["PII"]:
                for message_i in pii["Messages"]:
                    idx2piis[message_i].append(pii["Name"].strip())
                
        processed_messages = []
        for i, message in enumerate(messages):
            # if "Message" not in message: continue
            if i % 2 == 0:
                prompt, answer = self.prompter.generate_prompt(
                    message["Message"], piis=idx2piis.get(i, None)
                )
                
                if load_labels:
                    processed_messages.append({
                        "prompt": prompt,
                        "answer": answer
                    })
                else:
                    processed_messages.append({"prompt": prompt})
                
        return processed_messages
    
    def _tokenize(self, message, load_labels=True):
        prompt_toks = self.tokenizer(message["prompt"])
        
        if load_labels:
            labels = self.tokenizer(self.tokenizer.bos_token + message["answer"])["input_ids"]
            return {
                **prompt_toks,
                **{"labels": labels}
            }
        return prompt_toks
    
    def _load_data(self, data_file, load_labels=True):
        data = json.load(open(data_file, "r"))["Conversations"]
        messages = []
        for conv in data:
            if "Conversation" in conv:
                messages += self._process_conversation(conv, load_labels=load_labels)
        dataset = Dataset.from_list(messages)
        dataset = dataset.map(lambda x: self._tokenize(x, load_labels=load_labels))
        dataset = dataset.remove_columns(["prompt"])
        if load_labels:
            dataset = dataset.remove_columns(["answer"])
        
        return dataset
    
    def collate_fn(self, batch):
        seq2seq_collate = DataCollatorForSeq2Seq(self.tokenizer, padding=True, return_tensors="pt")
        return seq2seq_collate(batch)
    
    @property
    def train_dataset(self):
        return self._load_data(self.train_file)