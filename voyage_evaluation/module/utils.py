from typing import Dict, List

import numpy as np
import torch.distributed as dist


def merge_prediction(prediction: Dict, new_prediction: Dict):
    """ Merge predictions with dictionary format (qid, cid) -> score """
    for qid, qvalues in new_prediction.items():
        if qid not in prediction:
            prediction[qid] = qvalues
        else:
            for cid, score in qvalues.items():
                if cid not in prediction[qid]:
                    prediction[qid][cid] = score
                else:
                    # When the docs are splitted, we take a max pooling
                    prediction[qid][cid] = max(prediction[qid][cid], score)
    return prediction


def gather_list(data : List, num_devices: int):
    """ Gather list data and merge them into a list """
    if num_devices == 1:
        return data
    gathered = [None] * num_devices
    dist.all_gather_object(gathered, data)
    gathered = sum(gathered, [])
    return gathered


def convert_embd_type(embd: List[float], embd_type: str, max_value: float = 0.16):
    # embd: list of float
    # embd_type: float, int8, binary
    # return: list of float
    if embd_type == "float":
        return embd
    embd = np.array(embd)
    if embd_type == "int8":
        # [-max_value, max_value] => [-128, 127]
        embd = ((embd / max_value).clip(-1, 1) * 127).astype(np.int8)
        return embd.tolist()
    if embd_type == "uint8":
        # [-max_value, max_value] => [0, 255]
        embd = (((embd / max_value).clip(-1, 1) + 1) / 2 * 255).astype(np.uint8)
        return embd.tolist()
    if embd_type == "binary":
        embd = (np.packbits(embd > 0) - 128).astype(np.int8)
        return embd.tolist()
    raise NotImplementedError(f"embd_type {embd_type} is invalid.")