import torch
from tqdm import tqdm
from transformers.models.bert.tokenization_bert import BertTokenizer

def postprocess_state(state):
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
    return state


def inference(model, eval_loader, slot_meta, device, tokenizer):
    model.eval()
    predictions = []
    labels = []
    for batch in tqdm(eval_loader):
        input_ids, segment_ids, input_masks, gating_ids, target_ids, label_texts = batch
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        input_masks = input_masks.to(device)
        gating_ids = gating_ids.to(device)
        target_ids = target_ids.to(device)

        with torch.no_grad():
            o, g = model(input_ids, segment_ids, input_masks, 9)
            # o = slot 값?
            # g = gate

            _, generated_ids = o.max(-1)
            _, gated_ids = g.max(-1)

        for gate, gen in zip(gated_ids.tolist(), generated_ids.tolist()):
            prediction = recover_state(gate, gen, slot_meta, tokenizer)
            prediction = postprocess_state(prediction)
            predictions.append(prediction)
        
        labels.extend(label_texts)

    return predictions, labels


def recover_state(gate_list, gen_list, slot_meta, tokenizer: BertTokenizer):
    # 원래 레이블을 되찾아 주는 함수
    # gate가 ptr 이면 텍스트(gen)으로부터 텍스트 복원
    # gate가 ptr과 none이외 (여기서는 dontcare경우만)이면 gate값을 value로 복원
    # none인 경우 리스트에 포함시키지 않음
    assert len(gate_list) == len(slot_meta)
    assert len(gen_list) == len(slot_meta)

    id2gating = { 0: "none", 1: "dontcare", 2: "ptr" }

    recovered = []
    for slot, gate, value in zip(slot_meta, gate_list, gen_list):
        if id2gating[gate] == "none":
            continue

        if id2gating[gate] == "dontcare":
            recovered.append(f"{slot}-dontcare")
            continue

        token_id_list = []
        for id_ in value:
            if id_ in tokenizer.all_special_ids:
                break

            token_id_list.append(id_)
        value = tokenizer.decode(token_id_list, skip_special_tokens=True)

        if value == "none":
            continue

        recovered.append(f"{slot}-{value}")
    return recovered