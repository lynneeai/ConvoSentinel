import torch
from collections import defaultdict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from message_si_detector.prompters import PIIDetectorPrompter


class PIIDetector(object):
    def __init__(self, base_model_path, model=None, model_checkpoint=None):
        try:
            assert bool(model != None) ^ bool(model_checkpoint != None)
        except:
            raise AssertionError("Please initialize PIIDetector with either a trained model object or a model_checkpoint directory.")
        
        self.prompter = PIIDetectorPrompter()
        self._setup_tokenizer(base_model_path)
        self.generation_config = GenerationConfig(max_new_tokens=16)
        
        if model:
            self.model = model
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, device_map="auto")
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
            self.model.eval()
        
    def _setup_tokenizer(self, base_model):
        print(f"Setting up tokenizer from {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.add_special_tokens({"bos_token": "<s>"})
        
    def message_level_detection(self, message):
        prompt, _ = self.prompter.generate_prompt(message)
        input_ids = self.tokenizer(prompt)["input_ids"]
        generated = self.model.generate(
            input_ids=torch.tensor([input_ids]).to(self.model.device),
            decoder_input_ids = torch.tensor(
                [self.tokenizer.bos_token_id]
            ).unsqueeze(0).to(self.model.device),
            generation_config=self.generation_config
        )
        generated_text = self.tokenizer.decode(generated[0])
        pred_piis = self.prompter.get_response(generated_text)
        return pred_piis
    
    def conversation_level_detection(self, conversation, load_labels=True):
        messages = conversation["Conversation"]
        idx2piis = defaultdict(list)
        
        if load_labels:
            annotation = conversation["GroundTruth"]
            for pii in annotation["PII"]:
                for message_i in pii["Messages"]:
                    idx2piis[message_i].append(pii["Name"].strip())
        
        pred_obj = {
            "IsMalicious": None,
            "Ambiguity": None,
            "Multi": None,
            "FirstPII": -1,
            "NumPIIMessages": 0,
            "PII": []
        }         
        pred_piis_list, gold_piis_list = [], []
        pred_pii2indices = defaultdict(list)
        for i, message in enumerate(messages):
            # if "Message" not in message: continue
            if i % 2 == 0:
                pred_piis = self.message_level_detection(message["Message"])
                pred_piis_list.append(pred_piis)
                
                if pred_obj["FirstPII"] == -1 and bool(pred_piis):
                    pred_obj["FirstPII"] = i
                pred_obj["NumPIIMessages"] += int(bool(pred_piis))
                
                for pii in pred_piis:
                    pred_pii2indices[pii].append(i)
                
                if load_labels:
                    gold_piis_list.append(idx2piis.get(i, []))
                
        pred_obj["PII"] = [{"Name": pii, "Messages": indices} for pii, indices in pred_pii2indices.items()]
        
        return {
            "pred_labels": pred_obj,
            "gold_labels": conversation["GroundTruth"] if load_labels else None,
            "pred_piis_list": pred_piis_list,
            "gold_piis_list": gold_piis_list if load_labels else None
        }