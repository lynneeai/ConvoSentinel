import sys
sys.path.append("../../")
from utils import configure_gpu_device
configure_gpu_device(devices_str="0")

import json
import torch
from tqdm import tqdm
from src.models.vectorizer import VectorizeModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

def conv_pii_dict(pii):
    result = {}
    for items in  pii:
        for index in items['Messages']:
            if index not in result:
                result[index] = []
            result[index].append(items['Name'])
    return result


def save_conversations(conv,conversations):
    data = conv['Conversation']
    is_malicious = conv['IsMalicious']
    pii_dict = conv_pii_dict(conv['PII'])
    scenario = conv['Scenario']
    # List to store conversation groups
    current_conversation = []
    
    # Iterate through the data
    for i, entry in enumerate(data):
        # Add the current message to the current conversation
        current_conversation.append(entry)
        
        # Check if we are at the start and don't have enough for a 6-turn, but can save a 2-turn conversation
        # if i == 2:
        if i < 5 and len(current_conversation) % 2 == 0:
            conversations.append([current_conversation.copy(),{'Is_Malicious':is_malicious, 'pii_type':pii_dict.get(i, []), 'Scenario':scenario}])
        # If the current conversation has 6 turns, save it and reset
        if len(current_conversation) == 6:
            conversations.append([current_conversation.copy(),{'Is_Malicious':is_malicious, 'pii_type':pii_dict.get(i, []), 'Scenario':scenario}])
            current_conversation = current_conversation[2:]  # Keep the conversation flowing by sliding window
            

if __name__ == "__main__":
    SOURCE_INDEX_DATA_PATH = ""
    VEC_INDEX_DATA = "semafor_index"
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    vec_model = VectorizeModel(device)
    print("this worked")
    index_dim = VectorizeModel(device).predict("whatever works")[0].size
    source_index_data = []
    #script for load data here
    with open("../../data/v5/annotated_train.json", "r") as file:
        lbl_train_data = json.load(file)
    import pandas as pd
    conversation_data = []

    for idx, conversation in tqdm(list(enumerate(lbl_train_data['Conversations'])), dynamic_ncols=True):

        try:
            row = {"Conversation": conversation['Conversation']}
        except:
            row = {"Conversation": conversation['Conversations']}

        row.update(conversation['GroundTruth'])
        conversation_data.append(row)

    conversations_df = pd.DataFrame(conversation_data)
    source_index_data = []
    for idx, row in conversations_df.iterrows():
        save_conversations(row,source_index_data)
    
    doc = []
    for query in tqdm(source_index_data, dynamic_ncols=True):
        vec = vec_model.predict(json.dumps(query[0]))
        doc.append(Document(page_content=json.dumps(query[0]), metadata=query[1]))
    db = FAISS.from_documents(doc, embeddings)

    db.save_local("../faiss_index_v5")