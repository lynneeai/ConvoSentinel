import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class VectorizeModel:
    def __init__(self, device = "cpu") -> None:
        self.model = SentenceTransformer('sentence-transformers/sentence-t5-base')
        self.model.eval()
        # self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.DEVICE = device
        self.model.to(self.DEVICE)
    
    def predict(self,query):
        return self.model.encode(query)

    def predict_request(self, query):
        q_id_pred = self.predict_vec(query)
        return q_id_pred.cpu().numpy().tolist()
    
    def get_sim(self, q1, q2):
        q1_v = self.predict_vec(q1)
        q2_v = self.predict_vec(q2)
        sim = F.cosine_similarity(q1_v[0], q2_v[0], dim=-1)
        return sim.numpy().tolist()

if __name__ == "__main__":
    vec_model = VectorizeModel()
    print(vec_model.predict("this really really works!"))