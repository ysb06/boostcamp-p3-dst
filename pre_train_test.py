import os
import time
import random
import pickle
import numpy as np
from tqdm import tqdm
from filelock import FileLock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import BertTokenizer, BertConfig, BertForPreTraining
from transformers import DataCollatorForLanguageModeling
from transformers.utils import logging

from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback # transformers 4.5.1에서 가능

logger = logging.get_logger(__name__)

# set seed 
# reference : https://hoya012.github.io/blog/reproducible_pytorch/
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
set_seed(42) 

train_data_file = "/opt/ml/input/data/train_dataset/train_dials.json"
dev_data_file = "/opt/ml/input/data/eval_dataset/eval_dials.json"


class TextDatasetForNextSentencePrediction(Dataset):
    def __init__(
        self,
        tokenizer,
        file_path,
        block_size,
        overwrite_cache=False,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_nsp_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        self.tokenizer = tokenizer

        lock_path = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.
        
        
        # ✅ 캐시 형태로 파일을 저장합니다 
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]",
                    time.time() - start,
                )
            else:
                print(f"Creating features from dataset file at {directory}")
                logger.info(f"Creating features from dataset file at {directory}")
                # Make dataset
                self.documents = [[]]

                # ✅ 기존 코드엔 progress bar가 없어서 추가하였습니다 : 공식코드는 TQDM이 없음 -> pbar로 걸어주자
                cnt = 0
                count_data = len(open(file_path, "r", errors="ignore").readlines())

                pbar = tqdm(total=count_data)
                with open(file_path, encoding="utf-8") as f:
                    while True:  
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)
                        pbar.update(1)
                pbar.close()

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []

                for doc_index, document in enumerate(tqdm(self.documents)):
                    self.create_examples_from_document(document, doc_index)  

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]",
                    cached_features_file,
                    time.time() - start,
                )

    def create_examples_from_document(self, document, doc_index):
        """Creates examples for a single document."""
        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(
            pair=True
        )

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.

        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0
        
        # ✅ NSP를 위한 Preprocessing : NSP가 필요없다면, 이 부분을 제외해주세요 
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])
                    tokens_b = []
                    if (
                        len(current_chunk) == 1
                        or random.random() < self.nsp_probability
                    ):
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = random.randint(
                                0, len(self.documents) - 1
                            )
                            if random_document_index != doc_index:
                                break
                        random_document = self.documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                        """Truncates a pair of sequences to a maximum sequence length."""
                        while True:
                            total_length = len(tokens_a) + len(tokens_b)
                            if total_length <= max_num_tokens:
                                break
                            trunc_tokens = (
                                tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                            )
                            assert len(trunc_tokens) >= 1
                            # We want to sometimes truncate from the front and sometimes from the
                            # back to add more randomness and avoid biases.
                            if random.random() < 0.5:
                                del trunc_tokens[0]
                            else:
                                trunc_tokens.pop()

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    # add special tokens
                    input_ids = self.tokenizer.build_inputs_with_special_tokens(
                        tokens_a, tokens_b
                    )
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = (
                        self.tokenizer.create_token_type_ids_from_sequences(
                            tokens_a, tokens_b
                        )
                    )
                    
                    # ✅ 데이터가 저장되는 형태     
                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(
                            token_type_ids, dtype=torch.long
                        ),
                        "next_sentence_label": torch.tensor(
                            1 if is_random_next else 0, dtype=torch.long
                        ),
                    }

                    # ✅ 주의 : 이렇게 append 하는 방식은 대용량 corpus에서 메모리 이슈를 불러올 수 있습니다 ! 
                    self.examples.append(example)

                current_chunk = []
                current_length = 0

            i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


tokenizer = BertTokenizer.from_pretrained('dsksd/bert-ko-small-minimal')

# for dataset
train_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path=train_data_file,
    block_size=256,
    overwrite_cache=False,
    short_seq_probability=0.1,
    nsp_probability=0.5,
)

dev_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path=dev_data_file,
    block_size=256,
    overwrite_cache=False,
    short_seq_probability=0.1,
    nsp_probability=0.5,
)


print(len(train_dataset))
print(len(dev_dataset))


config = BertConfig.from_pretrained('dsksd/bert-ko-small-minimal')
config.model_name_or_path = 'dsksd/bert-ko-small-minimal'
# config.n_gate = len(processor.gating2id)
config.proj_dim = None

model = BertForPreTraining('dsksd/bert-ko-small-minimal', config=config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ [MASK] 과정은 Huggingface collator 따름
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.5
)

n_epochs = 10

training_args = TrainingArguments(
    output_dir='./checkpoints',
    learning_rate=4e-4,
    overwrite_output_dir=True,
    num_train_epochs=n_epochs,
    per_gpu_train_batch_size=30, # 서버에 맞게 설정
    save_steps=2000,
    save_total_limit=10, # 메모리 생각해서 알아서 조절 !
    logging_steps=2000,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",  # `epoch`: Evaluate every end of epoch. / mlm평가는 loss로 하는게 편함 
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=20, early_stopping_threshold=0.0001
)
trainer = Trainer(
    callbacks=[early_stopping], # callback사용을 위해 필요한 argument
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

# trainer.train()

# trainer.save_model("./checkpoints/final_bert_model")