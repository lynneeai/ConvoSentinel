import sys
sys.path.append("../")
import logging
import json
from tqdm import tqdm
from collections import defaultdict

from openai_utils import call_gpt, get_cost_estimate


log = logging.getLogger(__name__)

TEMPLATE = (
    "A conversation is malicious if one person is attempting to elicit personal information for illegitimate reasons from another person. "
    + "A conversation is benign when any personal information is being shared for legitimate reasons.\n"
    + "Determine if the following conversation is malicious or benign, given the explanation.\n"
    + "Output answer in one word: [malicious|benign].\n\n"
    + "Conversation:\n{conv}\n\n"
    + "Explanation: {explanation}\n\n"
    + "Answer: This conversation is [malicious|benign].\n\n"
)
MESSAGE_TEMPLATE = "[Message {message_id}] {name}: {message}"
EXPLANATION_TEMPLATE = "In message {message_id}, {attacker} requested {piis}, which is likely to be {message_label}."
# EXPLANATION_TEMPLATE = "In message {message_id}, {attacker} requested {piis}."


def get_label(conv_str, explan_str, only_estimate_cost=False, model="gpt-4-turbo"):
    prompt = TEMPLATE.format(conv=conv_str, explanation=explan_str)
    log.debug(f"Prompt: {prompt}")
    
    if only_estimate_cost:
        return None, get_cost_estimate(
            len(prompt.split()), 
            output_word_count=1,
            model=model
        )
    
    response, cost = call_gpt(
        prompt=prompt,
        model=model,
        max_tokens=4
    )
    return response, cost


def get_conv_explan_str(conv_list, pii_label, msg_label):
    conv = "\n".join(
        [MESSAGE_TEMPLATE.format(message_id=i, name=x["Name"], message=x["Message"]) for i, x in enumerate(conv_list)]
    )
    
    attacker = conv_list[0]["Name"]
    msg2piis = defaultdict(list)
    for pii in pii_label["PII"]:
        for msg_id in pii["Messages"]:
            msg2piis[msg_id].append(pii["Name"])
    msg2piis = {k: ", ".join(v) for k, v in msg2piis.items()}
    if len(msg2piis) == 0:
        explan = "No personal information was requested."
    else:
        explan = []
        for msg_id, label in msg_label["predictions"].items():
            explan.append(EXPLANATION_TEMPLATE.format(
                message_id=msg_id,
                attacker=attacker,
                piis=msg2piis[int(msg_id)],
                message_label=label
            ))
        explan = " ".join(explan)
    
    return conv, explan
    

def main(args):
    pii_labels = json.load(open(args.si_file,'r'))
    msg_labels = json.load(open(args.msg_label_file, "r"))
    test_data = json.load(open(args.test_file, "r"))["Conversations"]
    assert len(pii_labels) == len(msg_labels) == len(test_data)
    
    predictions = []
    total_cost = {"cost": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for data, pii_label, msg_label in tqdm(zip(test_data, pii_labels, msg_labels), total=len(test_data), dynamic_ncols=True):
        conv_str, explan_str = get_conv_explan_str(data["Conversation"], pii_label, msg_label)
        response, cost = get_label(conv_str, explan_str, only_estimate_cost=args.only_estimate_cost)
        total_cost = {k: total_cost[k] + cost[k] for k in cost}
        
        if not args.only_estimate_cost:
            predictions.append({
                "id": data["GroundTruth"]["ConversationID"],
                "label": "malicious" if data["GroundTruth"]["IsMalicious"] else "benign",
                "scenario": data["GroundTruth"]["Scenario"],
                "ambiguity": data["GroundTruth"]["Ambiguity"],
                "multi": data["GroundTruth"]["Multi"],
                "prediction": response,
            })
            json.dump(predictions, open(args.prediction_file, "w"), indent=4)
    
    print(f"Total cost: ${total_cost['cost']:.2f}")
    print(f"Prompt tokens: {total_cost['prompt_tokens']}")
    

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_estimate_cost", action="store_true")
    parser.add_argument("--test_file", default="../data/annotated_test.json")
    parser.add_argument("--si_file", type=str, required=True)
    parser.add_argument("--msg_label_file", default="./predictions/gpt-4-turbo_msg_top3.json")
    parser.add_argument("--prediction_file", default="./predictions/gpt-4-turbo_conv_top3.json")
    parser.add_argument("--debug", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.ERROR)
    args = parser.parse_args()
    
    logging.basicConfig(level=args.loglevel)
    
    if not args.only_estimate_cost:
        os.makedirs(os.path.dirname(args.prediction_file), exist_ok=True)
    
    main(args)