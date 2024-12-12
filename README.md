# ConvoSentinel

### This is the official repository of the EMNLP 2024 paper: [Defending Against Social Engineering Attacks in the Age of LLMs](https://aclanthology.org/2024.emnlp-main.716/).

## SEConvo Data
The **SEConvo** data is available for download [here](https://zenodo.org/records/12170260). See `SEConvo/README.md` for data details.

## To Run ConvoSentinel
1. Recommended to use `python 3.10`.
2. Obtain the [`trained_si_model`](https://drive.google.com/file/d/1g1t5u-M1IvWk2bFO1U6PLNcb7qriAPEu/view?usp=sharing) and [`faiss_index_v5`](https://drive.google.com/file/d/1UeASLNxBNBwymkA4_1Oi5kyOeHpkuLhI/view?usp=sharing) (conversation snippet dataset). Place `trained_si_model` in the `message_si_detector/` folder and `faiss_index_v5` in the `se_attempt_detector/` folder. These models are publicly available but are not detailed here due to the anonymity policy.
3. Place your Hugging Face token in the `hf_token.txt` file and your OpenAI API key in the `openai_api_key.txt` file.
4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
5. Run the `ConvoSentinel` script:
    ```sh
    bash run_convosentinel.sh SEConvo/annotated_test.json outputs/message_si_preds.json outputs/message_se_preds.json outputs/conv_se_preds.json
    ```
    This script generates:
    - Message-level SI predictions: `outputs/message_si_preds.json`
    - Message-level SE predictions: `outputs/message_se_preds.json`
    - Conversation-level SE predictions: `outputs/conv_se_preds.json`
6. By default, the script runs with `cuda:0`. Modify this in the script if necessary.


### If you find our method useful, please cite us:
```
@article{ai2024defending,
  title={Defending Against Social Engineering Attacks in the Age of LLMs},
  author={Ai, Lin and Kumarage, Tharindu and Bhattacharjee, Amrita and Liu, Zizhou and Hui, Zheng and Davinroy, Michael and Cook, James and Cassani, Laura and Trapeznikov, Kirill and Kirchner, Matthias and others},
  journal={arXiv preprint arXiv:2406.12263},
  year={2024}
}
```
