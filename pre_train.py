import hashlib
import json
import os
import pickle
import random
import time
from dataclasses import dataclass
from os import listdir, sep
from os.path import isfile, join
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from filelock import FileLock
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import EarlyStoppingCallback  # transformers 4.5.1에서 가능
from transformers import (BertConfig, BertForPreTraining, BertTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)
from transformers.utils import logging
import pandas as pd

logger = logging.get_logger(__name__)

@dataclass
class Example:
    input_ids: Tensor
    token_type_ids: Tensor
    next_sentence_label: Tensor

@dataclass
class Dialogue:
    utterances: List[Tuple[str, str]]

class NextPredictionDataset(Dataset):
    def __init__(self, features) -> None:
        self.features = features


def load_chatbot_data(path: str):
    raw = pd.read_csv(path)
    raw = raw.sample(frac=1).reset_index(drop=True)
    print("Loading Chatbot data...")
    result = [Dialogue([(raw_line["Q"], raw_line["A"])]) for raw_line in tqdm(raw.iloc, total=len(raw))]
    return result

def load_wos_data(train_path: str, dev_path: str):
    with open(train_path, 'r') as fr:
        train_data = json.load(fr)
    with open(dev_path, 'r') as fr:
        dev_data = json.load(fr)

    train_result: List[Dialogue] = []
    for dialogue_raw in tqdm(train_data):
        utterance_list = []

        sys_utterance = ""
        for utterance in dialogue_raw["dialogue"]:
            if utterance["role"] == "sys":
                sys_utterance = utterance["text"]
            elif utterance["role"] == "user":
                utterance_list.append((sys_utterance, utterance["text"]))
                sys_utterance = ""

        train_result.append(Dialogue(utterance_list))
    
    return train_result, None



if __name__ == "__main__":
    chatbot_data = load_chatbot_data("/workspace/Chatbot_data/ChatbotData.csv")
    for index, item in enumerate(chatbot_data):
        print(item.utterances[0])
        if index == 10:
            break

    wos_train_data, wos_dev_data = load_wos_data(
        "/opt/ml/input/data/train_dataset/train_dials.json", 
        "/opt/ml/input/data/eval_dataset/eval_dials.json"
    )

    for index, data in enumerate(wos_train_data):
        print(*data.utterances, sep='\n')
        if index == 0:
            break
