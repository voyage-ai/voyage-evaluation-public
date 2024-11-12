from typing import Optional


def get_embedding_model(
    model_type: str, 
    model_name: str,
    api_key: Optional[str] = None,
):
    from .embed import SentenceTransformersEmbeddingModel, OpenAIEmbeddingModel, VoyageAPIEmbeddingModel
    if model_type == "sentence-transformers":
        return SentenceTransformersEmbeddingModel(model_name)
    elif model_type == "openai":
        return OpenAIEmbeddingModel(model_name, api_key=api_key)
    elif model_type == "voyage_api":
        return VoyageAPIEmbeddingModel(model_name, api_key=api_key)
    else:
        raise NotImplementedError(f"model_type {model_type} is not valid.")
