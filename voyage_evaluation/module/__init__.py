from typing import Optional

from .encoder import Encoder
from .retriever import Retriever


def get_encoder(
    model_type: str,
    model_name: str,
    save_embds: bool = False,
    load_embds: bool = False,
    embd_type: str = "float",
    **kwargs,
):
    
    return Encoder(
        model_type,
        model_name,
        save_embds=save_embds,
        load_embds=load_embds,
        embd_type=embd_type,
        **kwargs,
    )


def get_retriever(
    model_type: str,
    topk: int = 100,
    similarity: str = "cosine",
    save_prediction: bool = False,
    embd_type: str = "float",
    data_type: str = "eval",
    **kwargs,
):
    return Retriever(
        topk=topk,
        similarity=similarity,
        save_prediction=save_prediction,
    )
