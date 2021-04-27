from torch.utils.data import Dataset
import pandas as pd
import json


class DSTDataset(Dataset):
    def __init__(self, raw: pd.DataFrame) -> None:
        self.raw_data: pd.DataFrame = raw
        self.targets = [x for x in range(len(raw))]

    def __getitem__(self, idx):
        return self.targets[idx]

    def __len__(self):
        return len(self.targets)


def load_dataset(path: str):
    with open(f"{path}/train_dials.json", 'r') as fr:
        dial_data = json.load(fr)

    with open(f"{path}/slot_meta.json", 'r') as fr:
        sltm_data = json.load(fr)
    
    with open(f"{path}/ontology.json", 'r') as fr:
        otlg_data = json.load(fr)

    print(type(dial_data))
    print(type(sltm_data))
    print(type(otlg_data))

if __name__ == "__main__":
    load_dataset("/opt/ml/input/data/train_dataset")