import argparse
import copy
from dst.train import TraineeBase
import dst.train
import os
from typing import Dict

import torch
import yaml


def initialize():
    training_config = {
        "data_path": {
            "root": "/opt/ml/input/data",
            "training_data": "/opt/ml/input/data/train_dataset",
            "evaluation_data": "/opt/ml/input/data/eval_dataset",
        },
        "save_path": {
            "root": "./results",
            "checkpoints_path": "./results/checkpoint",
            "tensorboard_log_path": "./results/tensorboard",
            "log_path": "./results/log",
        },
        "training_base": {
            "trainee_type": "BaselineTrainee",
            "trainee_name": "No_name",
            "hyperparameters": {
                "model": {
                    "name": "bert-base-multilingual-cased",
                    "type": "Bert",
                },
                "args": {
                    "num_train_epochs": 5,              # total number of training epochs
                    "learning_rate": 5e-5,              # learning_rate
                    "per_device_train_batch_size": 16,  # batch size per device during training
                    # number of warmup steps for learning rate scheduler
                    "warmup_steps": 500,
                    "weight_decay": 0.01,               # strength of weight decay
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
            trainee_class = getattr(dst.train, setting["trainee_type"])
            trainee: TraineeBase = trainee_class(target_device, config["data_path"]["training_data"], save_paths, setting["hyperparameters"])
            trainee.train()
        
        if do_inference:
            pass
            
