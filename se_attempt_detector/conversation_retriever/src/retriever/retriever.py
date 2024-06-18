import json,requests,copy
import numpy as np
import sys
import os
from src.retriever.vector.vector_searcher import VecSearcher
from src.models.vectorizer import VectorizeModel
import torch


class Retriever:
    def __init__(self, vec_search_path):
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.vec_model = VectorizeModel(device)
        # print("setup worked")
        self.vec_searcher = VecSearcher()
        
        self.vec_searcher.load(vec_search_path)

    def rank(self, query, recall_result):
        rank_result = []
        for idx in range(len(recall_result)):
            new_sim = self.vec_model.get_sim(query, recall_result[idx][1][0]) # reranking with cosine similarity
            rank_item = copy.deepcopy(recall_result[idx])
            rank_item.append(new_sim)
            rank_result.append(copy.deepcopy(rank_item))
        rank_result.sort(key=lambda x: x[3], reverse=True)
        return rank_result
    
    def search(self, query, nums=3):
        q_vec = self.vec_model.predict(query).cpu().numpy()
        recall_result = self.vec_searcher.search(q_vec, nums)
        rank_result = self.rank(query, recall_result)
        print("response: {}".format(rank_result))
        result = []
        for answer in rank_result:
            tmp_result = {}
            # tmp_result["query"] = answer[0]
            tmp_result["answer"] = answer[1][1]["answer"]
            tmp_result["match_query"] = answer[1][0]
            tmp_result["score"] = str(answer[3])
            result.append(copy.deepcopy(tmp_result))
        return result
    

if __name__ == "__main__":
    # VEC_INDEX_DATA = "semafor_index"
    print("import succesful")
    # retriever = Retriever(vec_search_path =VEC_INDEX_DATA)
    # q = "I want your personal info! give me your date of birth!"
    # print(q)
    # print(retriever.search(q))