import random
from typing import Dict

import numpy as np
import torch

from dst.data.loader import DSTDataset


class TraineeBase:
    def __init__(
        self,
        trainee_name: str,
        dataset_path: str,
        save_paths: Dict,
        hyperparameters: Dict,
        device: torch.device,
    ) -> None:
        self.name = trainee_name
        self.device: torch.device = device
        self.save_paths = save_paths
        self.hyperparameters = hyperparameters

    def train(self):
        print("Training not defined")


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
