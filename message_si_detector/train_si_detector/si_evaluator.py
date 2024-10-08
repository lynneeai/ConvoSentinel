import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import classification_report
from collections import defaultdict
from tqdm import tqdm


class SIEvaluator(object):
    def __init__(self, device="cpu"):
        self.device = device
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
    
    def message_level_score(self, pred_piis, gold_piis):
        score_obj = {
            # "binary": (
            #     bool(not pred_piis and not gold_piis)
            #     or
            #     bool(pred_piis and gold_piis)
            # ), 
            "cosine": 0.0,
            "pred_label": int(bool(pred_piis)),
            "gold_label": int(bool(gold_piis))
        }
        
        if bool(pred_piis and gold_piis):
            max_cos_sim = []
            gold_piis_embed = self.model.encode(gold_piis)
            for pii in pred_piis:
                pii_embed = self.model.encode(pii)
                max_cos_sim.append(torch.max(util.pytorch_cos_sim(pii_embed, gold_piis_embed)).numpy())
            score_obj["cosine"] = np.mean(max_cos_sim)
        elif bool(not pred_piis and not gold_piis):
            score_obj["cosine"] = 1
        
        return score_obj
    
    def conversation_level_score(self, pred_obj, gold_obj):
        pred_piis = [obj["Name"] for obj in pred_obj["PII"] if all(idx % 2 == 0 for idx in obj["Messages"])]
        gold_piis = [obj["Name"] for obj in gold_obj["PII"] if all(idx % 2 == 0 for idx in obj["Messages"])]
        
        score_obj = {"cosine": 0.0}
        if bool(pred_piis and gold_piis):
            max_cos_sim = []
            gold_piis_embed = self.model.encode(gold_piis)
            for pii in pred_piis:
                pii_embed = self.model.encode(pii)
                max_cos_sim.append(torch.max(util.pytorch_cos_sim(pii_embed, gold_piis_embed)).numpy())
            score_obj["cosine"] = np.mean(max_cos_sim)
        elif bool(not pred_piis and not gold_piis):
            score_obj["cosine"] = 1
        return score_obj
    
    def evaluate(self, pred_obj_list, gold_obj_list):
        conversation_level_cosines, message_level_cosines = [], []
        message_labels, message_preds = [], []
        for pred_obj, gold_obj in tqdm(list(zip(pred_obj_list, gold_obj_list)), dynamic_ncols=True):
            # if "Conversation" not in gold_obj: continue
            message_len = len(gold_obj["Conversation"])
            gold_obj = gold_obj["GroundTruth"]
            conv_level_score = self.conversation_level_score(pred_obj, gold_obj)
            conversation_level_cosines.append(conv_level_score["cosine"])
            
            idx2piis_pred = defaultdict(list)
            for pii in pred_obj["PII"]:
                for message_i in pii["Messages"]:
                    idx2piis_pred[message_i].append(pii["Name"].strip())
        
            idx2piis_gold = defaultdict(list)
            for pii in gold_obj["PII"]:
                for message_i in pii["Messages"]:
                    idx2piis_gold[message_i].append(pii["Name"].strip())
            
            for i in range(message_len):
                if i % 2 == 0:
                    pred_piis = idx2piis_pred.get(i, [])
                    gold_piis = idx2piis_gold.get(i, [])
                    message_score = self.message_level_score(pred_piis, gold_piis)
                    message_level_cosines.append(message_score["cosine"])
                    message_labels.append(message_score["gold_label"])
                    message_preds.append(message_score["pred_label"])

        
        avg_message_cosine = np.mean(message_level_cosines).item()
        avg_conversation_cosine = np.mean(conversation_level_cosines).item()
        
        report = classification_report(message_labels, message_preds, target_names=["NO PII", "PII"], zero_division=0)
        print("Message level PII binary detection classification report:")
        print(report)
        print(f"Message level PII name cosine similarity: {avg_message_cosine}")
        print(f"Conversation level PII name cosine similarity: {avg_conversation_cosine}")
        
        return {
            "message_level_cosines": avg_message_cosine,
            "conversation_level_cosines": avg_conversation_cosine,
            "classification_report": classification_report(message_labels, message_preds, target_names=["NO PII", "PII"], zero_division=0, output_dict=True)
        }