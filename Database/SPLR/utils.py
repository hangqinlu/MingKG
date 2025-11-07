def build_type_index(path):
    with open(path, encoding='utf-8') as f:
        index_2_type = f.read().split("\n")
    type_2_index = {r: i for i, r in enumerate(index_2_type)}

    return type_2_index, index_2_type