import os
import json
import logging
import argparse

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from voyage_evaluation.task import run_retrieve_task
from voyage_evaluation.dataset import get_task_list
from voyage_evaluation.module import get_encoder, get_retriever


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus", type=int, default=None, help="Number of gpus used for encoding.")
    parser.add_argument(
        "--cpus", type=int, default=1, help="Number of cpus used for computation (this is only for models that are not using gpus).")
    parser.add_argument(
        "--bf16", action="store_true", help="Use bf16 precision.")
    parser.add_argument(
        "--do_retrieve", action="store_true", help="Run retrieval.")
    # Model
    parser.add_argument(
        "--model_type", type=str, default="openai", help="Model type options: `openai`, `voyage_api`.")
    parser.add_argument(
        "--model_name", type=str, default=None, help="Model name or path.")
    parser.add_argument(
        "--add_instruct", action="store_true", help="Whether to add instruction to query and document.")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument(
        "--embd_batch_size", type=int, default=1024, help="Batch size for computing similarity of embeddings.")
    parser.add_argument(
        "--embd_in_memory_threshold", type=int, default=200000,
        help="Embeddings will be stored in memory if the amount is below this threshold.")
    parser.add_argument(
        "--api_key", type=str, default=None, help="API key.")
    parser.add_argument(
        "--embd_type", type=str, default="float", help="Embedding type. Options: float, int8, uint8, binary, ubinary.")
    parser.add_argument(
        "--similarity", type=str, default="cosine", help="Similarity function. Options: cosine, dot-product.")
    parser.add_argument(
        "--rescore", action="store_true", help="In the case of binary embedding, whether to rescore with float embeddings.")
    parser.add_argument(
        "--bm25_tokenizer", type=str, default="basic", help="Tokenizer used by the bm25 retriever.")
    parser.add_argument(
        "--prompt_type", type=str, default="v1", help="Type of prompt.")
    # Data
    parser.add_argument(
        "--data_path", type=str, default=None, required=True, help="Path of the dataset, must be specified for custom tasks.")
    parser.add_argument(
        "--task_name", type=str, default=None, help="Name of the task. Can be multiple tasks splitted by `,`.")
    parser.add_argument(
        "--data_type", default="eval", choices=["eval", "train", "chunk", "merge"], help="Dataset type.")
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Cache directory of Huggingface dataset.")
    parser.add_argument(
        "--candidate_file", type=str, default=None, help="Canditate file for reranking.")
    parser.add_argument(
        "--candidate_path", type=str, default=None, help="Path to the rerank candidate files.")
    parser.add_argument(
        "--max_length", type=int, default=None, help="Maximum length of model input. If None, use the value in tokenizer config.")
    parser.add_argument(
        "--max_query_length", type=int, default=None, help="Maximum length of query tokens.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader.")
    parser.add_argument(
        "--group_by_query", action="store_true", help="Whether to group the candidates by query in a batch.")
    parser.add_argument(
        "--split_candidate", action="store_true", help="Whether to split the candidate docs by max_length.")
    # Output
    parser.add_argument(
        "--save_path", type=str, default=None, required=True, help="Path to save the output.")
    parser.add_argument(
        "--save_embds", action="store_true", help="Whether to save the embeddings.")
    parser.add_argument(
        "--load_embds", action="store_true", help="Whether to load the computed embeddings.")
    parser.add_argument(
        "--save_intermediate", action="store_true", help="Whether to save intermediate predictions.")
    parser.add_argument(
        "--load_intermediate", action="store_true", help="Whether to load intermediate predictions.")
    parser.add_argument(
        "--save_prediction", action="store_true", help="Whether to save the predictions.")
    parser.add_argument(
        "--topk", type=int, default=100, help="Number of top documents per query.")
    parser.add_argument(
        "--no-eval", dest="eval", action="store_false", default=True, help="Whether to run evaluation.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to overwrite the results.")
    
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    if args.gpus:
        trainer = pl.Trainer(
            strategy=DDPStrategy(find_unused_parameters=False),
            accelerator="gpu",
            devices=args.gpus,
            precision="bf16" if args.bf16 else "32",
        )
    else:
        trainer = pl.Trainer(
            strategy=DDPStrategy(),
            accelerator="cpu",
            devices=args.cpus,
        )

    if not trainer.is_global_zero:
        logging.basicConfig(level=logging.ERROR)

    task_list = get_task_list(args.task_name, args.data_path, args.data_type)
    trainer.print("Tasks:", json.dumps(task_list, indent=4))
    trainer.print("Model type:", args.model_type)
    trainer.print("Model name:", args.model_name)

    if args.do_retrieve:
        encoder = get_encoder(
            args.model_type,
            args.model_name,
            save_embds=args.save_embds,
            load_embds=args.load_embds,
            embd_type=args.embd_type,
            api_key=args.api_key,
        )
        retriever = get_retriever(
            args.model_type,
            topk=args.topk,
            embd_type=args.embd_type,
            data_type=args.data_type,
            similarity=args.similarity,
            save_prediction=args.save_prediction,
        )
        eval_results = {
            task: run_retrieve_task(task, trainer, encoder, retriever, args)
            for task in task_list
        }
        metric = "ndcg_at_10"

    if args.eval and trainer.is_global_zero:
        trainer.print("=" * 40)
        trainer.print(args.save_path)
        trainer.print("=" * 40)
        for task in task_list:
            if metric in eval_results[task]:
                trainer.print(f"{task:<32}{eval_results[task][metric]:.4f}")


if __name__ == "__main__":
    main()