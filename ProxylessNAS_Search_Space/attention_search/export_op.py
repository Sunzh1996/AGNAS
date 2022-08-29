import copy
import pickle
import numpy as np
from config import config

if __name__ == '__main__':
    with open('attention_weights.pickle', 'rb') as f:
        attention_weights = pickle.load(f)
    print(attention_weights)

    path_id = []  # to comply with the same rule defined in RLNAS for retraining
    for i in range(config.layers):
        inp = config.backbone_info[i+1][0]
        oup = config.backbone_info[i+1][1]
        stride = config.backbone_info[i+1][-1]
        candidate_ops = copy.deepcopy(config.blocks_keys)
        if inp == oup and stride == 1:
            candidate_ops.append('skip')
        _attention_weights = attention_weights[i].tolist()
        assert len(candidate_ops) == len(_attention_weights)

        choice_idx = _attention_weights.index(max(_attention_weights))
        choice_op = candidate_ops[choice_idx]
        print(i, choice_op)

        if choice_op == 'skip':
            path_id.append(-1)
        else:
            path_id.append(config.blocks_keys.index(choice_op))
    print(path_id)
