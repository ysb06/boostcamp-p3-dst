import json
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

pad_token_id = 3
# 이 부분이 tokenizer 부를 때마다 달라질 텐데
# 전역 변수로 사용하지 않을 방법이 있을까?

@dataclass(frozen=True)
class DSTInputExample:
    context_turns: List[str]
    current_turn: List[str]
    label: Optional[List[str]] = None
# dataclass는 서로 쉽게 값 비교할 수 있게 만들어 주는 Annotation
# 참조: https://www.daleseo.com/python-dataclasses/


@dataclass
class OpenVocabDSTFeature:
    input_idx_sent: List[int]
    segment_idx: List[int]
    gating: List[int]
    target: List[List[int]]
    target_origin: List[str]


class WOSDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]

def load_TRADE_dataset_from_raw(
        raw_root_dir: str,
        raw_dir_path: str,
        raw_slot_meta_path: str,
        tokenizer: BertTokenizer,
        gate: Dict[str, int],
        dev_split_k: int = 5, 
        seed: int = None
    ) -> Tuple[List[WOSDataset], List[WOSDataset], List[List[int]]]:
    # BERT 계열의 토크나이저만 사용이 가능한데 혹시 다른 모델 사용할 수도 있을까?
    # Baseline을 그대로 가지고 왔지만 저 모델은 Electra 계열인데....?

    print("Load data from raw...")
    if dev_split_k <= 1:
        # validation 데이터셋이 없는 경우 대응이 안됨.
        raise Exception("Currently dev_split_k cannot be lower than 2")
    
    # 무조건 straitified k fold로 나누고 전체에 대해서 토큰화를 진행하는 비효율적인 코드로 되어 있음.
    # 토큰화 후 straitified k fold로 나누는 게 더 효율적일듯.
    # 코드 고치기...
    train_input_folds, dev_input_folds = _convert_data_from_raw(raw_dir_path, dev_split_k, seed=seed)
    print("Converting raw finished")
    
    with open(raw_slot_meta_path, 'r') as fr:
        sl_m_data: List[str] = json.load(fr)
    print("Load data finished")

    # Feature 변환 (다른 DST 모델을 사용하면 아래 코드를 변경해야 함)
    print()
    print("Converting data to features...")
    train_feature_folds, dev_feature_folds = [], []

    print()
    if dev_input_folds is not None:     # Development (Validation) 데이터가 있는 경우
        for index, (train_input, dev_input) in enumerate(zip(train_input_folds, dev_input_folds)):
            print(f"Convert [{index}] training data to features")
            train_feature_folds.append([_convert_to_TRADE_feature(dialogue, sl_m_data, tokenizer, gate) for dialogue in tqdm(train_input)])

            print(f"Convert [{index}] development data to features")
            dev_feature_folds.append([_convert_to_TRADE_feature(dialogue, sl_m_data, tokenizer, gate) for dialogue in tqdm(dev_input)])

    else:                               # Development (Validation) 데이터가 없는 경우
        for index, train_input in enumerate(train_input_folds):
            print(f"Convert [{index}] training data to features")
            train_feature_folds.append([_convert_to_TRADE_feature(dialogue, sl_m_data, tokenizer, gate) for dialogue in tqdm(train_input)])

    print(f"Converting {len(train_feature_folds)} data finished")

    tokenized_data_path = f"{raw_root_dir}/train_dataset/ov_features_{dev_split_k}_{seed}.pkl"
    with open(tokenized_data_path, 'wb') as fwb:
        # 변환이 완료되면 저장
        pickle.dump(
            (
                train_feature_folds, 
                dev_feature_folds
            ), fwb)
    print(f"Features saved at {tokenized_data_path}")

    return load_TRADE_dataset_from_features(tokenized_data_path, raw_slot_meta_path, tokenizer)


def load_TRADE_dataset_from_features(
        tokenized_data_path: str,
        raw_slot_meta_path: str,
        tokenizer: BertTokenizer,
    ) -> Tuple[List[WOSDataset], List[WOSDataset], List[List[int]]]:
    # 강제로 다시 읽어 들임. pickle이 제대로 저장되어야 함
    print(f"Loading Features...[{tokenized_data_path}] {int(os.path.getsize(tokenized_data_path) / (1024**2))} MB")
    with open(tokenized_data_path, 'rb') as frb:
        train_feature_folds, dev_feature_folds = pickle.load(frb)
    
    with open(raw_slot_meta_path, 'r') as fr:
        sl_m_data: List[str] = json.load(fr)
    print("Features loaded")
    
    training_datasets = []
    dev_datasets = []
    for train_feature_fold, dev_feature_fold in zip(train_feature_folds, dev_feature_folds):
        training_datasets.append(WOSDataset(train_feature_fold))
        dev_datasets.append(WOSDataset(dev_feature_fold))

    tokenized_slot_meta = []
    for slot in sl_m_data:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )

    global pad_token_id
    pad_token_id = tokenizer.pad_token_id
    # 전역변수 쓰는 이 코드를 어떻게 해야 할까...

    return training_datasets, dev_datasets, tokenized_slot_meta

    
def collate_TRADE_dataset(batch: List[OpenVocabDSTFeature]):
    input_ids = torch.LongTensor(
        _pad_ids([b.input_idx_sent for b in batch], pad_token_id)
    )
    segment_ids = torch.LongTensor(
        _pad_ids([b.segment_idx for b in batch], pad_token_id)
    )
    input_masks = input_ids.ne(pad_token_id)

    gating_ids = torch.LongTensor([b.gating for b in batch])
    target_ids = _pad_id_of_matrix(
        [torch.LongTensor(b.target) for b in batch],
        pad_token_id,
    )

    label_texts = [b.target_origin for b in batch]
    return input_ids, segment_ids, input_masks, gating_ids, target_ids, label_texts





# --모듈 내부에서만 사용되는 함수들

# ----Pre-processing 관련 함수들
# recover_state는 뭐하는 함수일까? ans: 원래 텍스트로 다시 변환하는 함수


def _convert_to_TRADE_feature(
        input_segment: DSTInputExample, 
        slot_meta: List, 
        tokenizer: BertTokenizer,
        gating2id: Dict[str, int],
        max_tokenizing_length: int = 512,
    ):

    dialogue_context = " [SEP] ".join(input_segment.context_turns + input_segment.current_turn)

    input_idx_sent = [tokenizer.cls_token_id] + tokenizer.encode(dialogue_context, add_special_tokens=False, truncation=True, max_length=max_tokenizing_length) + [tokenizer.sep_token_id]
    segment_idx = [0] * len(input_idx_sent) # 여기서는 sep로 나뉜 문장에 대한 별도의 구분이 들어가지 않는다. (0으로 통일)
    # BERT 처럼 구분을 준다면 어떨까?
    target_idx_group = [] # Slot 값 리스트
    gating_idx_list = []  # Gate 값 리스트

    state = _convert_state_dict(input_segment.label)

    # 전체 slot에 대해 리스트를 만들고 값을 추가한다.
    # 데이터에는 일부 slot에 대한 정보만 있다.
    # slot value 값은 sep 토큰으로 구분 (TRADE 신경망에서는 string 배열을 인지하지 못하므로)
    for slot in slot_meta:
        value = state.get(slot, "none") # state에 없는 slot의 값은 none으로 설정
        target_idx_group.append(tokenizer.encode(value, add_special_tokens=False) + [tokenizer.sep_token_id])
        gating_idx_list.append(gating2id.get(value, gating2id["ptr"]))    # none 또는 dontcare가 아니면 ptr로 변환
    target_idx_group = _pad_ids(target_idx_group, tokenizer.pad_token_id)   # Pad 토큰 추가
    # 결과 예시) 
    # [
    #   [21832, 11764, 3],  no + ne + [sep] 아마도...
    #   [21832, 11764, 3],
    #   [8732, 3, 0]
    #   ...
    # ]

    return OpenVocabDSTFeature(
        input_idx_sent=input_idx_sent,
        segment_idx=segment_idx,
        target=target_idx_group,
        gating=gating_idx_list,
        target_origin=input_segment.label
    )


def _pad_ids(
        arrays: List[List[int]], 
        pad_idx: int, 
        max_length: int = -1
    ) -> List[List[int]]:
    """리스트에 Pad 토큰을 추가하여 Shape를 맞춰주는 함수

    Args:
        arrays (List[List[int]]): 원본 리스트
        pad_idx (int): Pad 토큰 값
        max_length (int, optional): 강제로 길이를 맞출지 여부. Defaults to -1 (리스트 내 최대 값).

    Returns:
        List[List[int]]: Pad 토큰이 추가된 리스트
    """
    if max_length < 0:
        # max_length를 따로 설정하지 않으면 리스트 내 최대 길이값으로 설정
        max_length = max(list(map(len, arrays)))

    arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
    return arrays

def _pad_id_of_matrix(arrays, padding, max_length=-1):
    if max_length < 0:
        max_length = max([array.size(-1) for array in arrays])

    new_arrays = []
    for array in arrays:
        n, l = array.size()
        pad = torch.zeros(n, (max_length - l))
        pad[:,:,] = padding
        pad = pad.long()
        m = torch.cat([array, pad], -1)
        new_arrays.append(m.unsqueeze(0))

    return torch.cat(new_arrays, 0)


def _convert_state_dict(state: List) -> Dict:
    """리스트 형태의 state를 dict로 변환

    Ex)

    ['관광-종류-박물관', '관광-지역-서울 중앙'] ==> {'관광-종류': '박물관', '관광-지역': '서울 중앙'}


    Args:
        state (List): state 리스트

    Returns:
        Dict: 변환 결과
    """
    dic = {}
    for slot in state:
        s, v = _split_slot(slot, get_domain_slot=True)
        dic[s] = v
    return dic

def _split_slot(dom_slot_value: str, get_domain_slot: bool = False):
    """Slot을 분할한다

    Ex) get_domian_slot이 True인 경우

    관광-종류-박물관 ==> 관광-종류, 박물관

    Args:
        dom_slot_value (str): state에 있는 slot 원래 텍스트
        get_domain_slot (bool, optional): Domain정보가 이미 있는지 여부. Defaults to False.
    """
    try:
        dom, slot, value = dom_slot_value.split("-")
    except ValueError:
        tempo = dom_slot_value.split("-")
        if len(tempo) < 2:
            return dom_slot_value, dom_slot_value, dom_slot_value
        dom, slot = tempo[0], tempo[1]
        value = dom_slot_value.replace(f"{dom}-{slot}-", "").strip()

    if get_domain_slot:
        return f"{dom}-{slot}", value
    return dom, slot, value


# ----기타


def _convert_data_from_raw(
        path: str,
        dev_split_k: int, 
        seed: int = None
    ):
    # DST에서는 validation을 development라고 하는 것 같다...왜?
    with open(path, 'r') as fr:
        dial_data: List = json.load(fr)
    
    train_data_folds, dev_data_folds = _split_data(dial_data, dev_split_k, seed)

    # DSTInputSegment 리스트로 변환
    train_input_folds, dev_input_folds = [], []
    if len(dev_data_folds) != 0:
        for train_fold, dev_fold in zip(train_data_folds, dev_data_folds):
            train_input_folds.append(_generate_dst_input(train_fold))
            dev_input_folds.append(_generate_dst_input(dev_fold))
        
        return train_input_folds, dev_input_folds
    else:
        for train_data in train_data_folds:
            train_input_folds.append(_generate_dst_input(train_data))
        
        return train_input_folds, None


def _generate_dst_input(data: List) -> List:
    """DST 공통 입력 데이터로 변환

    Args:
        data (List): 원본 데이터

    Returns:
        List: 변환 된 데이터
    """
    dst_input = []

    for dialogue in data:
        history = []
        sys_utter = ""
        for turn in dialogue["dialogue"]:
            if turn["role"] == "sys":
                sys_utter = turn["text"]
                continue
            else:
                user_utter = turn["text"]
                state = turn["state"]   # 무조건 리스트로 들어옴
                # if type(state) != list:
                #     print(state)
                #     print(f"{dialogue['dialogue_idx']} is empty state")
                context = deepcopy(history)
                current_turn = [sys_utter, user_utter]

                dst_input.append(
                    DSTInputExample(
                        context_turns=context,
                        current_turn=current_turn,
                        label=state
                    )
                )

                history.append(sys_utter)
                history.append(user_utter)
    
    return dst_input


def _split_data(data: List, split_k: int, seed: int) -> Union[List, List]:
    """Domain 조합 별을 고려하여 Stratified K Fold로 데이터를 나누어 주는 함수

    Args:
        data (List): 원래 데이터
        split_k (int): 나눌 크기
        seed (int): Shuffle에 사용될 시드 값, None일 경우 Shuffle 없음

    Returns:
        Union[List, List]: 나뉘어진 데이터 K개 묶음
    """
    print("Splitting...")
    if split_k <= 1:
        print(f"Failed to split by K(={split_k})")
        return data, None
    
    # Baseline에서는 도메인 갯수를 기반으로 잘랐지만
    # 여기서는 도메인 조합을 기반으로 자름
    domain_comb_list = []
    domain_comb_set = set()

    for dial in data:
        domain_comb = " ".join(sorted(dial["domains"]))
        domain_comb_list.append(domain_comb)
        domain_comb_set.add(domain_comb)

    k_fold_splitter = StratifiedKFold(n_splits=split_k, shuffle=(seed != None), random_state=seed)
    k_fold_data_indexes = k_fold_splitter.split(X=data, y=domain_comb_list)

    train_data_folds, dev_data_folds = [], []
    for index, (train_idxes, dev_idxes) in enumerate(k_fold_data_indexes):
        train_data = [data[idx] for idx in train_idxes]
        train_data_folds.append(train_data)

        dev_data = [data[idx] for idx in dev_idxes]
        dev_data_folds.append(dev_data)

        # 아래는 도메인 조합 중 빠진 것이 있는지 여부 확인 코드 (없어도 크게 문제는 없음)
        # 현재는 콘솔로 경고 메시지만 보여주지만 추후 필요한 경우 대응 가능
        train_domain_comb_list = []
        for dial in train_data:
            train_domain_comb = " ".join(sorted(dial["domains"]))
            train_domain_comb_list.append(train_domain_comb)
        
        dev_domain_comb_list = []
        for dial in dev_data:
            dev_domain_comb = " ".join(sorted(dial["domains"]))
            dev_domain_comb_list.append(dev_domain_comb)
        
        for domain_comb in domain_comb_set:
            if domain_comb not in train_domain_comb_list:
                print(f"{domain_comb} is not in {index} training set fold")
            if domain_comb not in dev_domain_comb_list:
                print(f"{domain_comb} is not in {index} development set fold")
    
    print()
    print(f"Split dataset to {split_k} fold with seed {seed}")
    return train_data_folds, dev_data_folds
        

if __name__ == "__main__":
    tokenizer_model_name = "monologg/koelectra-base-v3-discriminator"
    print(f"Loading [{tokenizer_model_name}] tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_name)
    print("Loading tokenizer finished")

    load_TRADE_dataset_from_raw(
        "/opt/ml/input/data/",
        "/opt/ml/input/data/train_dataset/train_dials.json",
        "/opt/ml/input/data/train_dataset/slot_meta.json",
        tokenizer,
        seed=327459)
