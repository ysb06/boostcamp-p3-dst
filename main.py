import argparse
import copy
import os
from typing import Dict, Type

import torch
import yaml

import dst.train
from dst.data.loader import OpenVocabDSTFeature
from dst.train import TraineeBase


def initialize():
    training_config = {
        "data_path": {
            "root_dir": "/opt/ml/input/data",
            "training_dialogue_data": "/opt/ml/input/data/train_dataset/train_dials.json",
            "training_slot_meta_data": "/opt/ml/input/data/train_dataset/slot_meta.json",
            "training_ontology_data": "/opt/ml/input/data/train_dataset/ontology.json",
            "training_features_data": "/opt/ml/input/data/train_dataset/open_vocab_features.pkl",
            "evaluation_dialogue_data": "/opt/ml/input/data/eval_dataset/eval_dials.json",
            "evaluation_slot_meta_data": "/opt/ml/input/data/eval_dataset/slot_meta.json",
        },
        "save_path": {
            "root_dir": "./results",
            "checkpoints_dir": "./results/checkpoint",
            "tensorboard_log_dir": "./results/tensorboard",
            "yaml_log_dir": "./results/yaml_log",
        },
        "training_base": {
            "trainee_type": "TRADETrainee",
            "trainee_name": "No_name",
            "hyperparameters": {
                "model": {
                    "type" : "TRADE",
                    "prtrained_embedding_model": "monologg/koelectra-base-v3-discriminator",
                },
                "args": {
                    "vocab_size": 1,
                    "hidden_size": 1,
                    "hidden_dropout_prob": 1,
                    "n_gate": 1,
                    "vocab_size": 1,
                    "vocab_size": 1,
                    "vocab_size": 1,
                    "vocab_size": 1,
                },
                "dev_split" : {
                    "split_k" : 5,
                    "target" : None
                },
                "seed": 327459
            }
        },
        "trainings": [
            {
                "trainee_name": "No_name",
            }
        ]
    }

    with open(f"./config.yaml", 'w') as fw:
        yaml.dump(training_config, fw)
        print("config.yaml created!")

    return training_config


def load_config():
    training_config = {}
    with open(f"./config.yaml", 'r') as fr:
        training_config = yaml.load(fr, Loader=yaml.FullLoader)
        print("config.yaml loaded")

    return training_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=str, default="True")
    parser.add_argument("--inference", type=str, default="True")
    option_args = parser.parse_args()

    do_training: bool = option_args.training.lower() == "true"
    do_inference: bool = option_args.inference.lower() == "true"

    # Search target device
    print(f"PyTorch version: [{torch.__version__}]")
    if torch.cuda.is_available():   # 무조건 cuda만 사용
        target_device = torch.device("cuda:0")
    else:
        raise Exception("No CUDA Device")
    print(f"  Target device: [{target_device}]")
    print()

    # Load config file
    if not os.path.isfile("./config.yaml"):
        config = initialize()
    else:
        config = load_config()
    print()

    # 필요한 폴더 생성
    save_paths = config["save_path"]
    if not os.path.isdir(save_paths["root_dir"]):
        os.mkdir(save_paths["root_dir"])
    for key in save_paths:
        if not os.path.isdir(save_paths[key]):
            os.mkdir(save_paths[key])

    # 하이퍼파라미터 초기화
    training_settings = []
    training_base_setting: Dict = config["training_base"]
    for training in config["trainings"]:
        setting = copy.deepcopy(training_base_setting)
        setting.update(training)
        training_settings.append(setting)
    
    # Training
    for index, setting in enumerate(training_settings):
        if do_training:
            trainee_class: Type[TraineeBase] = getattr(dst.train, setting["trainee_type"])
            trainee: TraineeBase = trainee_class(
                setting["trainee_name"], 
                config["data_path"], 
                config["save_path"], 
                setting["hyperparameters"], 
                target_device
            )
            trainee.train()
        
        if do_inference:
            pass
            
