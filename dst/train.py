from dst.model import TRADE, TRADEConfig
import random
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from dst.data import loader


class TraineeBase:
    def __init__(
        self,
        trainee_name: str,
        dataset_paths: Dict,
        save_paths: Dict,
        hyperparameters: Dict,
        device: torch.device,
    ) -> None:
        self.name = trainee_name
        self.device = device
        self.save_paths = save_paths
        self.hyperparameters = hyperparameters

        seed_everything(hyperparameters["seed"])

    def train(self):
        raise NotImplementedError("Need to specify train function")


class TRADETrainee(TraineeBase):
    def __init__(
        self,
        trainee_name: str,
        dataset_paths: Dict,
        save_paths: Dict,
        hyperparameters: Dict,
        device: torch.device
    ) -> None:
        super().__init__(trainee_name, dataset_paths, save_paths, hyperparameters, device)

        tokenizer_model_name = hyperparameters["model"]["prtrained_embedding_model"]
        print(f"Loading [{tokenizer_model_name}] tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_model_name)
        print("Loading tokenizer finished")
        print()

        features_path = dataset_paths["training_features_data"]
        if features_path is not None:
            training_datasets, dev_datasets, self.tokenized_slot_meta = loader.load_TRADE_dataset_from_features(
                features_path,
                dataset_paths["training_slot_meta_data"],
                tokenizer
            )
        else:
            training_datasets, dev_datasets, self.tokenized_slot_meta = loader.load_TRADE_dataset_from_raw(
                dataset_paths["root_dir"],
                dataset_paths["training_dialogue_data"],
                dataset_paths["training_slot_meta_data"],
                tokenizer,
                dev_split_k=hyperparameters["dev_split"]["split_k"],
                seed=hyperparameters["seed"],
            )

        target = hyperparameters["dev_split"]["target"]
        if target is not None:
            self.training_datasets = [training_datasets[target]]
            self.dev_datasets = [dev_datasets[target]]
        else:
            self.training_datasets = training_datasets
            self.dev_datasets = dev_datasets

        print() # End initializing
    
    def train(self):
        for fold, (training_dataset, dev_dataset) in enumerate(zip(self.training_datasets, self.dev_datasets)):
            print(f"Training start with fold {fold}...")
            
            config = TRADEConfig(

            )
            config.vocab_size = 0
            config.hidden_size = 0
            config.hidden_dropout_prob = 0
            config.n_gate = 0
            config.proj_dim = 0

            # Model 선언
            model = TRADE(args, tokenized_slot_meta)
            model.set_subword_embedding(self.hyperparameters["model"]["prtrained_embedding_model"])  # Subword Embedding 초기화
            print(f"Subword Embeddings is loaded from {args.model_name_or_path}")
            model.to(self.device)
            print("Model is initialized")


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
