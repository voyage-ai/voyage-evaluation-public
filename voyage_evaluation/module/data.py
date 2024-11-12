import torch
from typing import Optional, Any
from pytorch_lightning import LightningDataModule

from voyage_evaluation.dataset import get_retrieve_dataset
from voyage_evaluation.dataset.collator import get_retrieve_data_collator
from voyage_evaluation.dataset.collator.retrieve import EmbeddingDataCollator, RetrieveDataCollator
from voyage_evaluation.dataset.utils import EmptyDataset, JSONLDataset


    
class RetrieveDataModule(LightningDataModule):

    def __init__(
        self, 
        task_name: str,
        data_type: str = "eval",
        data_path: Optional[str] = None,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: int = 32, 
        embd_batch_size: int = 1024, 
        num_workers: int = 4,
        dataset_kwargs: Optional[dict] = None,
        collator_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.task_name = task_name
        self.batch_size = batch_size
        self.embd_batch_size = embd_batch_size
        self.num_workers = num_workers
        self.dataset = get_retrieve_dataset(
            self.task_name,
            data_path=data_path,
            data_type=data_type,
            **dataset_kwargs,
        )
        self.query_collator = get_retrieve_data_collator(
            model_type=model_type,
            model_name=model_name,
            is_query=True,
            **collator_kwargs,
        )
        self.corpus_collator = get_retrieve_data_collator(
            model_type=model_type,
            model_name=model_name,
            is_query=False,
            **collator_kwargs,
        )

    def prepare_data(self):
        self.dataset.prepare_data()

    def queries_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset.queries, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn=self.query_collator,
        )

    def corpus_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset.corpus, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn=self.corpus_collator,
        )

    def set_queries_embds(self, queries_embds=None, queries_embds_files=None):
        if queries_embds:
            self.queries_embds = queries_embds
            self.queries_embd_ds = EmptyDataset(queries_embds)
        else:
            self.queries_embd_ds = JSONLDataset(queries_embds_files)
        assert len(self.queries_embd_ds) == len(self.dataset.queries)

    def set_corpus_embds(self, corpus_embds=None, corpus_embds_files=None):
        if corpus_embds:
            self.corpus_embds = corpus_embds
            self.corpus_embd_ds = EmptyDataset(corpus_embds)
        else:
            self.corpus_embd_ds = JSONLDataset(corpus_embds_files)
        # TODO: check this assertion later, removed for chunk model
        # assert len(self.corpus_embd_ds) == len(self.dataset.corpus)

    def queries_embd_dataloader(self):
        return torch.utils.data.DataLoader(
            self.queries_embd_ds,
            batch_size=self.embd_batch_size,
            num_workers=self.num_workers,
            collate_fn=EmbeddingDataCollator(),
        )

    def corpus_embd_dataloader(self):
        return torch.utils.data.DataLoader(
            self.corpus_embd_ds,
            batch_size=self.embd_batch_size,
            num_workers=self.num_workers,
            collate_fn=EmbeddingDataCollator(),
        )