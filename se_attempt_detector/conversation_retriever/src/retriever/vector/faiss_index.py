import faiss

class VecIndex:
    def __init__(self) -> None:
        self.index = ""
    
    def build(self, index_dim):
        description = "HNSW64"
        measure = faiss.METRIC_L2
        self.index = faiss.index_factory(index_dim, description, measure)
    
    def insert(self, vec):
        self.index.add(vec) 
    
    def load(self, read_path):
        self.index = faiss.read_index(read_path)

    def save(self, save_path):
        faiss.write_index(self.index, save_path)
    
    def search(self, vec, num):
        return self.index.search(vec, num)