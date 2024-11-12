import torch
import io
from PIL import Image


class EmbeddingDataCollator:

    def __call__(self, examples):
        assert len(examples) > 0
        batch = {
            key: [example[key] for example in examples]
            for key in examples[0].keys()
        }
        batch["embd"] = torch.tensor(batch["embd"])
        return batch


class RetrieveDataCollator:

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self._early_truncate = True

    def __call__(self, examples):
        assert len(examples) > 0
        batch = {}
        batch["id"] = [ex["id"] for ex in examples]
        batch["text"] = [ex["text"] for ex in examples]

        if self.tokenizer:
            texts = [s.strip() for s in batch["text"]]

            if self._early_truncate:
                max_str_len = self.tokenizer.model_max_length * 6
                texts = [s[:max_str_len] for s in texts]
 
            batch["input"] = self.tokenizer(
                texts,
                padding=True, 
                truncation=True, 
                return_tensors="pt",
            )

        return batch