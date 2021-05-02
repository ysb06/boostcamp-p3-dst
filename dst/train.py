import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from dst.data import loader
from dst.evaluation import _evaluation
from dst.inference import inference
from dst.model import TRADE, TRADEConfig, masked_cross_entropy_for_value


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
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_model_name)
        print("Loading tokenizer finished")
        print()

        # 이 부분도 생각하지 못했음...좋지 않음
        with open(dataset_paths["training_slot_meta_data"], 'r') as fr:
            self.slot_meta: List[str] = json.load(fr)
        print("Features loaded")

        features_path = dataset_paths["training_features_data"]
        if features_path is not None:
            training_datasets, dev_datasets, self.tokenized_slot_meta = loader.load_TRADE_dataset_from_features(
                features_path,
                dataset_paths["training_slot_meta_data"],
                self.tokenizer
            )
        else:
            training_datasets, dev_datasets, self.tokenized_slot_meta = loader.load_TRADE_dataset_from_raw(
                dataset_paths["root_dir"],
                dataset_paths["training_dialogue_data"],
                dataset_paths["training_slot_meta_data"],
                self.tokenizer,
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
            
            pretrained_model_name = self.hyperparameters["model"]["prtrained_embedding_model"]

            config = TRADEConfig(
                vocab_size=len(self.tokenizer),
                n_gate=3,   # gate 갯수는 loader._convert_to_TRADE_feature의 gating2id Dictionary 길이와 같다.
                # gate를 yaml에 지정하는게 바람직한 일일까?
                **self.hyperparameters["model"]["args"]
            )

            # Model 선언
            model = TRADE(config, self.tokenized_slot_meta)
            model.set_subword_embedding(pretrained_model_name)  # Subword Embedding 초기화
            print(f"Subword Embeddings is loaded from {pretrained_model_name}")
            model.to(self.device)
            print("Model initialized")
            print()

            train_loader = DataLoader(
                training_dataset,
                batch_size=self.hyperparameters["train_batch_size"],
                collate_fn=loader.collate_TRADE_dataset,
            )
            print("# train:", len(training_dataset))

            dev_loader = DataLoader(
                dev_dataset,
                batch_size=self.hyperparameters["dev_batch_size"],
                collate_fn=loader.collate_TRADE_dataset,
            )
            print("# dev:", len(dev_dataset))

            # Optimizer 및 Scheduler 선언
            n_epochs = self.hyperparameters["epochs"]
            t_total = len(train_loader) * n_epochs
            warmup_steps = int(t_total * self.hyperparameters["warmup_ratio"])
            optimizer = AdamW(model.parameters(), **self.hyperparameters["optimizer"]["args"])
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )

            loss_fnc_1 = masked_cross_entropy_for_value  # generation
            loss_fnc_2 = nn.CrossEntropyLoss()  # gating

            best_score, best_checkpoint = 0, 0
            for epoch in range(n_epochs):
                model.train()
                for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                    input_ids, segment_ids, input_masks, gating_ids, target_ids, label_texts = batch
                    input_ids = input_ids.to(self.device)
                    segment_ids = segment_ids.to(self.device)
                    input_masks = input_masks.to(self.device)
                    gating_ids = gating_ids.to(self.device)
                    target_ids = target_ids.to(self.device)

                    # teacher forcing
                    if (
                        self.hyperparameters["teacher_forcing"] > 0.0
                        and random.random() < self.hyperparameters["teacher_forcing"]
                    ):
                        tf = target_ids
                    else:
                        tf = None

                    # Forward
                    all_point_outputs, all_gate_outputs = model(
                        input_ids, 
                        segment_ids, 
                        input_masks, 
                        target_ids.size(-1), 
                        tf
                    )

                    # generation loss
                    loss_1 = loss_fnc_1(
                        all_point_outputs.contiguous(),
                        target_ids.contiguous().view(-1),
                        self.tokenizer.pad_token_id,
                    )
                    
                    # gating loss
                    loss_2 = loss_fnc_2(
                        all_gate_outputs.contiguous().view(-1, 3),
                        # 주의: view의 두번째 인자(3)에는 gate 갯수가 들어가야 함.
                        gating_ids.contiguous().view(-1),
                    )
                    loss = loss_1 + loss_2

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.hyperparameters["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if step % 100 == 0:
                        print(f"Epoch: [{epoch}/{n_epochs}], Step: [{step}/{len(train_loader)}]")
                        print(f"loss: {loss.item()} gen_loss: {loss_1.item()} gate_loss: {loss_2.item()}")

                predictions, labels = inference(model, dev_loader, self.slot_meta, self.device, self.tokenizer)
                eval_result = _evaluation(predictions, labels, self.tokenized_slot_meta)
                for k, v in eval_result.items():
                    print(f"{k}: {v}")

                if best_score < eval_result['joint_goal_accuracy']:
                    print("Update Best checkpoint!")
                    best_score = eval_result['joint_goal_accuracy']
                    best_checkpoint = epoch

                    # 모델 저장
                    if not os.path.isdir(f"{self.save_paths['checkpoints_dir']}/{fold}"):
                        os.mkdir(f"{self.save_paths['checkpoints_dir']}/{fold}")

                    save_file_path = f"{self.save_paths['checkpoints_dir']}/{fold}/model_best.bin"

                    torch.save(model.state_dict(), save_file_path)
                    print(f"Best checkpoint saved at {save_file_path}")


def _evaluation(preds, labels: List[List[str]], slot_meta):
    evaluator = DSTEvaluator(slot_meta)

    evaluator.init()
    assert len(preds) == len(labels)

    for pred, label in zip(preds, labels):
        evaluator.update(label, pred)

    result = evaluator.compute()
    print(result)
    return result





class DSTEvaluator:
    def __init__(self, slot_meta):
        self.slot_meta = slot_meta
        self.init()

    def init(self):
        self.joint_goal_hit = 0
        self.all_hit = 0
        self.slot_turn_acc = 0
        self.slot_F1_pred = 0
        self.slot_F1_count = 0

    def update(self, gold, pred):
        self.all_hit += 1
        if set(pred) == set(gold):
            self.joint_goal_hit += 1

        temp_acc = compute_acc(gold, pred, self.slot_meta)
        self.slot_turn_acc += temp_acc

        temp_f1, _, _, count = compute_prf(gold, pred)
        self.slot_F1_pred += temp_f1
        self.slot_F1_count += count

    def compute(self):
        turn_acc_score = self.slot_turn_acc / self.all_hit
        slot_F1_score = self.slot_F1_pred / self.slot_F1_count
        joint_goal_accuracy = self.joint_goal_hit / self.all_hit
        eval_result = {
            "joint_goal_accuracy": joint_goal_accuracy,
            "turn_slot_accuracy": turn_acc_score,
            "turn_slot_f1": slot_F1_score,
        }
        return eval_result


def compute_acc(gold, pred, slot_meta):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_meta)
    ACC = len(slot_meta) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = (
            2 * precision * recall / float(precision + recall)
            if (precision + recall) != 0
            else 0
        )
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count




# ---------------------------------

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
