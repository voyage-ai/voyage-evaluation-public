import json
import time
import logging
from typing import List, Dict

import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):

    def forward(self, batch: Dict) -> List[List[float]]:
        raise NotImplementedError


class SentenceTransformersEmbeddingModel(EmbeddingModel):

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer

        super().__init__()
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def forward(self, batch):
        return self.model.encode(batch["text"])


class OpenAIEmbeddingModel(EmbeddingModel):

    def __init__(self, model_name, api_key):
        import openai
        import tiktoken
        
        super().__init__()
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_length = 8191

    def forward(self, batch):
        import openai

        texts = batch["text"]
        tokens = [self.tokenizer.encode(text, disallowed_special=()) for text in texts]
        if self.max_length:
            tokens = [t[:self.max_length] for t in tokens]
        while True:
            try:
                result = self.client.embeddings.create(
                    input=tokens,
                    model=self.model_name,
                )
                embeddings = [d.embedding for d in result.data]
                return embeddings
            except openai.RateLimitError as e:
                logging.error(e)
                time.sleep(30)
            except openai.InternalServerError as e:
                logging.error(e)
                time.sleep(60)


class VoyageAPIEmbeddingModel(EmbeddingModel):

    def __init__(self, model_name, api_key):
        import voyageai

        super().__init__()
        self.client = voyageai.Client(
            api_key=api_key,
            max_retries=3,
        )
        self.model_name = model_name
        self.max_retries = 5

    def forward(self, batch):
        import voyageai

        texts = batch["text"]
        num_retries = 0
        while num_retries < self.max_retries:
            try:
                result = self.client.embed(
                    texts, model=self.model_name, truncation=True)
                return result.embeddings
            except voyageai.error.RateLimitError as e:
                logging.error(e)
                time.sleep(30)
            except voyageai.error.ServiceUnavailableError as e:
                logging.error(e)
                time.sleep(60)
            num_retries += 1
        raise RuntimeError(
            f"Voyage embedding API did not respond in {num_retries} retries."
        )