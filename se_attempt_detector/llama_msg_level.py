import sys
sys.path.append("../")
import argparse
import os
import logging
    
from utils import configure_gpu_device

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="0")
parser.add_argument("--si_file", type=str, required=True)
parser.add_argument("--test_file", type=str, default="../data/annotated_test.json")
parser.add_argument("--prediction_file", type=str, default="./predictions/llama_msg_top3.json")
parser.add_argument("--debug", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.ERROR)
args = parser.parse_args()

logging.basicConfig(level=args.loglevel)

os.makedirs(os.path.dirname(args.prediction_file), exist_ok=True)
configure_gpu_device(devices_str=args.device)


import json
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from huggingface_hub import login
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


log = logging.getLogger(__name__)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
login(open("../hf_token.txt", "r").read().strip())

INSTRUCTION = (
    "A conversation is malicious if one person is attempting to elicit personal information for illegitimate reasons from another person. "
    + "A conversation is benign when any personal information is being shared for legitimate reasons.\n"
    + "Determine if the following conversation is malicious or benign. Output answer in one word: [malicious|benign].\n\n"
)
EXAMPLE_TEMPLATE = "Example Conv:\n{conv}\nQuestion: Is this conversation malicious or benign?\nAnswer: This conversation is {label}.\n\n"
TEMPLATE = "Conv:\n{conv}\nQuestion: Is this conversation malicious or benign?\nAnswer: This conversation is"

MODEL = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
TOKENIZER = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
GEN_CONFIG = GenerationConfig(max_new_tokens=2)


class Retriever:
    def __init__(self, vec_search_path):
        self.db = FAISS.load_local(vec_search_path, embeddings, allow_dangerous_deserialization=True)
    
    def search(self, query, nums=3):
        results = self.db.similarity_search(query, nums)
        example_str = ""
        for res in results:
            example_str += self._process_example(json.loads(res.page_content), res.metadata["Is_Malicious"])
        return example_str
    
    def _process_example(self, conv_list, label):
        conv = "\n".join([f"{x['Name']}: {x['Message']}" for x in conv_list])
        label = "malicious" if label else "benign"
        return EXAMPLE_TEMPLATE.format(conv=conv, label=label)
    

def get_conv_snippets(conv_list, pii_messages):
    all_snippets, message_indices = [], []
    curr_snippet = []
    for i, message in enumerate(conv_list):
        curr_snippet.append(message)
        if i == 2:
            if i in pii_messages:
                all_snippets.append(curr_snippet.copy())
                message_indices.append(i)
        if len(curr_snippet) == 5:
            if i in pii_messages:
                all_snippets.append(curr_snippet)
                message_indices.append(i)
            curr_snippet = curr_snippet[2:]
    
    snippet_strs = [
        json.dumps(snippet)
        for snippet in all_snippets
    ]
    
    snippet_template_strs = [
        TEMPLATE.format(conv="\n".join([f"{x['Name']}: {x['Message']}" for x in snippet]))
        for snippet in all_snippets
    ]
    return snippet_strs, snippet_template_strs, message_indices


def generate_label(conv_str, example_str):
    prompt = INSTRUCTION + example_str + conv_str
    log.debug(f"Prompt: {prompt}")
    
    input_ids = TOKENIZER(prompt, return_tensors="pt")["input_ids"]
    generated = MODEL.generate(input_ids=input_ids.to(MODEL.device), generation_config=GEN_CONFIG)
    answer = generated[0][len(input_ids[0]):]
    answer = TOKENIZER.decode(answer)
    return answer


def main(args):
    pii_labels = json.load(open(args.si_file,'r'))
    data = json.load(open(args.test_file, "r"))["Conversations"]
    assert(len(pii_labels) == len(data))
    
    retriever = Retriever("faiss_index_v5")
    
    predictions = []
    for data, pii_label in tqdm(list(zip(data, pii_labels)), dynamic_ncols=True):
        pii_messages = set()
        for pii in pii_label["PII"]:
            pii_messages.update(set(pii["Messages"]))
        
        snippet_strs, snippet_template_strs, message_indices = get_conv_snippets(data["Conversation"], pii_messages)
        pred = {
            "id": data["GroundTruth"]["ConversationID"],
            "label": "malicious" if data["GroundTruth"]["IsMalicious"] else "benign",
            "scenario": data["GroundTruth"]["Scenario"],
            "ambiguity": data["GroundTruth"]["Ambiguity"],
            "multi": data["GroundTruth"]["Multi"],
            "predictions": {}
        }
        for str, template_str, message_idx in zip(snippet_strs, snippet_template_strs, message_indices):
            example_str = retriever.search(str)
            response = generate_label(
                conv_str=template_str,
                example_str=example_str,
            )
            pred["predictions"][message_idx] = response
        
        predictions.append(pred)
        json.dump(predictions, open(args.prediction_file, "w"), indent=4)
    

if __name__ == "__main__":
    
    
    main(args)