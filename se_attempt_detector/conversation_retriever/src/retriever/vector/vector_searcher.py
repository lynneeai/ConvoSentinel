import os, json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))
from src.retriever.vector.faiss_index import VecIndex

class VecSearcher:
    def __init__(self):
        self.invert_index = VecIndex() 
        self.forward_index = [] 
        self.INDEX_FOLDER_PATH = "data/index/{}"

    def build(self, index_dim, index_name):
        self.index_name = index_name
        self.index_folder_path = self.INDEX_FOLDER_PATH.format(index_name)
        if not os.path.exists(self.index_folder_path) or not os.path.isdir(self.index_folder_path):
            os.mkdir(self.index_folder_path)
        self.invert_index.build(index_dim)
    
    def insert(self, vec, doc):
        self.invert_index.insert(vec)
        self.forward_index.append(doc)
    
    def save(self):
        with open(self.index_folder_path + "/forward_index.txt", "w") as f:
            for data in self.forward_index:
                # print(data)
                f.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
        print("forward_done")
        self.invert_index.save(self.index_folder_path + "/invert_index.faiss")
        print("invert_done")
    
    def load(self, index_name):
        self.index_name = index_name
        self.index_folder_path = self.INDEX_FOLDER_PATH.format(index_name)
        print("this worked")
        self.invert_index.load(self.index_folder_path + "/invert_index.faiss")
        print("this worked too?")
        with open(self.index_folder_path + "/forward_index.txt", encoding="utf8") as f:
            for line in f:
                self.forward_index.append(json.loads(line.strip()))
    
    def search(self, vecs, nums = 5):
        search_res = self.invert_index.search(vecs, nums)
        print(search_res[1][0])
        print(len(search_res[1][0]))
        recall_list = []
        print(self.forward_index)
        for idx in range(nums): #get top 5
            recall_list.append([search_res[1][0][idx], self.forward_index[search_res[1][0][idx]], search_res[0][0][idx]])

        return recall_list