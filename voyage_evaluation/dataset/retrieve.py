import os
import json
import random
from abc import ABC
from functools import cache
from typing import Dict, Optional
import json
import torch
from torch.utils.data import Dataset
from voyage_evaluation.dataset.utils import JSONLDataset

QUERY_INSTRUCT = "Represent the query for retrieving supporting documents: "
CORPUS_INSTRUCT = "Represent the document for retrieval: "


def add_instruct(dataset: Dataset, instruct: str):
    
    def _add_instruct(example):
        example["text"] = instruct + example["text"]
        return example

    for item in dataset.data:
        item = _add_instruct(item)

    return dataset


class RetrieveDataset(ABC):

    def __init__(self, data_path: str, task_name: str, add_instruct: bool = False):
        self.data_path = data_path
        self.task_name = task_name
        self.task_path = os.path.join(self.data_path, self.task_name)
        self.add_instruct = add_instruct

    @property
    @cache
    def corpus(self) -> Dataset:
        # Dataset of dicts with fields {"id", "text"}
        corpus = self._corpus()
        if self.add_instruct:
            corpus = add_instruct(corpus, CORPUS_INSTRUCT)
        return corpus

    def _corpus(self) -> Dataset:
        raise NotImplementedError

    @property
    @cache
    def queries(self) -> Dataset:
        # Dataset of dicts with fields {"id", "text"}
        queries = self._queries()
        if self.add_instruct:
            queries = add_instruct(queries, QUERY_INSTRUCT)
        return queries

    def _queries(self) -> Dataset:
        raise NotImplementedError

    @property
    @cache
    def relevance(self) -> Dict:
        # Dict of dict: relevance[query_id][corpus_id] = score
        pass

    def prepare_data(self):
        _ = self.corpus
        _ = self.queries
        _ = self.relevance


class RetrieveDatasetEvalType(RetrieveDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert os.path.isdir(self.task_path), f"{self.task_path} is not a directory."

    @property
    def corpus_file(self) -> str:
        for name in ["corpus.jsonl", "corpus.arrow"]:
            file = os.path.join(self.task_path, name)
            if os.path.exists(file):
                return file
        raise FileNotFoundError(
            f"Corpus file (corpus.{{jsonl/arrow}}) does not exist under {self.task_path}."
        )

    @cache
    def _corpus(self) -> Dataset:
        return JSONLDataset(self.corpus_file)

    @property
    def queries_file(self) -> str:
        for name in ["queries.jsonl", "queries.arrow"]:
            file = os.path.join(self.task_path, name)
            if os.path.exists(file):
                return file
        raise FileNotFoundError(
            f"Queries file (queries.{{jsonl/arrow}}) does not exist under {self.task_path}."
        )

    @cache
    def _queries(self) -> Dataset:
        return JSONLDataset(self.queries_file)

    @property
    def relevance_file(self) -> str:
        for name in ["relevance.json", "relevance.jsonl"]:
            file = os.path.join(self.task_path, name)
            if os.path.exists(file):
                return file
        raise FileNotFoundError(
            f"Relevance file (relevance.{{json/jsonl}}) does not exist under {self.task_path}."
        )

    @property
    @cache
    def relevance(self) -> Dict:
        relevant_docs = {}
        try:
            with open(self.relevance_file) as f:
                for line in f:
                    data = json.loads(line)
                    for key, value in data.items():
                        if key not in relevant_docs:
                            relevant_docs[key] = value
                        else:
                            relevant_docs[key].update(value)
        except FileNotFoundError:
            return {}
        return relevant_docs
