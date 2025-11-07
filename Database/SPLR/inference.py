import torch
from collections import  defaultdict
def clean_entities(entity_list):
    contained = set()
    for i, e1 in enumerate(entity_list):
        for j, e2 in enumerate(entity_list):
            if i != j and e1 in e2:
                contained.add(e1)
    if contained:
        return sorted(contained, key=entity_list.index)
    else:
        return list(sorted(set(entity_list), key=entity_list.index))
def get_entities(pred,offset_mapping,text,index_2_rel):
    point=[]
    entities=[]
    l_t = pred[0].T
    pred_mx_list = [mx.T for mx in pred[2:10]]
    delta_list = list(range(8))
    thresholds = [0.4] + [0.7] * 7

    for pred_mx, delta, thresh in zip(pred_mx_list, delta_list, thresholds):
        for j in range(pred_mx.size(0)):
            head_ids = torch.where(pred_mx[j] > thresh)[0]
            tail_ids = torch.where(l_t[0] > thresh)[0]
            for head_id in head_ids:
                tail_id = head_id + delta
                if tail_id in tail_ids:
                    if head_id < 0 or tail_id > len(offset_mapping[0]):
                        print(
                            f"Skipping index: obj_head_id={head_id}, obj_tail_id={tail_id}, length={len(offset_mapping)}")
                        continue
                    try:
                        head_pos_id = offset_mapping[0][head_id][0]
                        tail_pos_id = offset_mapping[0][tail_id][1]
                        point.append([head_pos_id, tail_pos_id, j, delta + 1])
                    except IndexError:
                        print(f"Index out of range: obj_head_id={head_id}, obj_tail_id={tail_id}")
                        continue
    entity = [tuple(h) for h in point]
    if not entity:
        return list((entities))
    else:

        for index, _ in enumerate(entity):
            pos_id = entity[index]
            head_id = pos_id[0]
            tail_id = pos_id[1]
            length = pos_id[-1]
            type = pos_id[2]
            object_text = text[head_id:tail_id]
            entities.append((object_text, index_2_rel[type], length))

    label_map = {
        "PER": "人物", "HONOR": "功名", "REG": "户籍地", "DEGREE": "甲第等级",
        "EXAM": "科举考试", "LOC": "地点", "AGE": "年龄", "ORD": "排行", "RANK": "排名",
        "OFF": "职官", "DATE": "时间", "FIELD": "学术专长", "STYLE": "字",
        "BOOK": "书", "STU": "学籍", "REGTYP": "户籍类型"
    }
    class_dict = defaultdict(list)
    for text, tag, _ in entities:
        zh_tag = label_map.get(tag)
        if zh_tag:  #
            class_dict[zh_tag].append(text)
        else:
            print(f'【WARNING】标签 {tag} 未定义映射，自动跳过！')
    result = {k: clean_entities(v) for k, v in class_dict.items()}
    return result

def ner_inference(text,tokenizer,device,model,index_2_type):
    input_data = tokenizer(text,  return_tensors="pt",  return_offsets_mapping=True)
    input_ids = input_data["input_ids"].to(device)
    attention_mask = input_data["attention_mask"].to(device)
    offset_mapping = input_data['offset_mapping'].to(device)
    with torch.no_grad():
        encoded_text,pred =  model.forward(input_ids,attention_mask)
        entities = get_entities(pred, offset_mapping,text, index_2_type)
    return entities
