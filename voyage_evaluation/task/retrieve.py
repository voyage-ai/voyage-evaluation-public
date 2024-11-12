import os
import json
from termcolor import colored

from beir.retrieval.evaluation import EvaluateRetrieval

from voyage_evaluation.module.data import RetrieveDataModule
import voyage_evaluation.constants as constants


def run_retrieve_evaluation(relevance, prediction):
    if len(relevance) != len(prediction):
        raise RuntimeError("Prediction and ground truth have different sizes.")
    
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        relevance, prediction, k_values=[1,3,5,10,20,50,100], ignore_identical_ids=False
    )
    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }
    return scores


def run_retrieve_task(task_name, trainer, encoder, retriever, args):
    task_save_path = os.path.join(args.save_path, task_name)
    os.makedirs(task_save_path, exist_ok=True)

    if not args.overwrite:
        eval_file = os.path.join(task_save_path, constants.RETRIEVE_EVAL_FILENAME)
        pred_file = os.path.join(task_save_path, constants.RETRIEVE_PRED_FILENAME)
        if args.eval:
            if os.path.exists(eval_file):
                with open(eval_file) as f:
                    scores = json.load(f)
                return scores
        else:
            if os.path.exists(pred_file):
                return

    # DataModule manages the datasets
    dataset_kwargs = {"add_instruct": args.add_instruct}
    collator_kwargs = {}

    dm = RetrieveDataModule(
        task_name=task_name,
        data_path=args.data_path,
        data_type=args.data_type,
        model_type=args.model_type,
        model_name=args.model_name,
        batch_size=args.batch_size,
        embd_batch_size=args.embd_batch_size,
        num_workers=args.num_workers,
        dataset_kwargs=dataset_kwargs,
        collator_kwargs=collator_kwargs,
    )
    if trainer.is_global_zero:
        dm.prepare_data()
        trainer.print(colored(dm.task_name, "red"))
        trainer.print("Queries size:", len(dm.dataset.queries))
        trainer.print("Corpus size:", len(dm.dataset.corpus))
        if args.data_type in ["chunk", "merge"]:
            trainer.print("Raw corpus size:", len(dm.dataset.raw_corpus))
    
    trainer.strategy.barrier()

    if len(dm.dataset.queries) < trainer.num_devices or len(dm.dataset.corpus) < trainer.num_devices:
        trainer.print(colored("Skipping the task due to too few queries / documents.", "red"))
        return {}

    if len(dm.dataset.queries) >= 1e6:
        trainer.print(colored("Skipping the task due to too many queries.", "red"))
        return {}

    if args.model_type == "bm25":
        # Build the index from corpus
        retriever.build_index(dm.dataset.corpus)
        # Compute the scores for queries
        retriever.save_file = os.path.join(task_save_path, constants.RETRIEVE_PRED_FILENAME)
        trainer.predict(model=retriever, dataloaders=dm.queries_dataloader())
    
    else:
        # Compute the query embeddings
        trainer.print(colored("Encode queries", "yellow"))
        encoder.is_query = True
        encoder.in_memory = (len(dm.dataset.queries) < args.embd_in_memory_threshold)
        encoder.save_file = os.path.join(task_save_path, constants.QUERIES_EMBD_FILENAME)
        if args.load_embds and encoder.embd_files_exist(trainer.num_devices):
            queries_embds_files = encoder.get_embd_files(trainer.num_devices)
            trainer.print(f"Embedding files exist: {queries_embds_files}")
            dm.set_queries_embds(queries_embds_files=queries_embds_files)
        else:
            trainer.print(f"in_memory = {encoder.in_memory}")
            trainer.print(f"save_file = {encoder.save_file}")
            trainer.predict(model=encoder, dataloaders=dm.queries_dataloader())
            # Set the query embeddings
            queries_embds_files = encoder.get_embd_files()
            dm.set_queries_embds(queries_embds=encoder.embds, queries_embds_files=queries_embds_files)
        
        # Compute the corpus embeddings
        trainer.print(colored("Encode corpus", "yellow"))
        encoder.is_query = False
        encoder.save_file = os.path.join(task_save_path, constants.CORPUS_EMBD_FILENAME)
        encoder.in_memory = (len(dm.dataset.corpus) < args.embd_in_memory_threshold)
        if args.load_embds and encoder.embd_files_exist(trainer.num_devices):
            corpus_embds_files = encoder.get_embd_files(trainer.num_devices)
            trainer.print(f"Embedding files exist: {corpus_embds_files}")
            dm.set_corpus_embds(corpus_embds_files=corpus_embds_files)
        else:
            trainer.print(f"in_memory = {encoder.in_memory}")
            trainer.print(f"save_file = {encoder.save_file}")
            trainer.predict(model=encoder, dataloaders=dm.corpus_dataloader())
            # Set the corpus embeddings
            corpus_embds_files = encoder.get_embd_files()
            dm.set_corpus_embds(corpus_embds=encoder.embds, corpus_embds_files=corpus_embds_files)

        # Run retriever
        trainer.print(colored("Retrieve", "yellow"))
        retriever.corpus_embd_dataloader = dm.corpus_embd_dataloader()
        retriever.in_memory = (len(dm.dataset.queries) < args.embd_in_memory_threshold)
        retriever.save_file = os.path.join(task_save_path, constants.RETRIEVE_PRED_FILENAME)
        trainer.predict(model=retriever, dataloaders=dm.queries_embd_dataloader())

        # Remove the embeddings
        if not args.save_embds and not args.load_embds and trainer.is_global_zero:
            for file in queries_embds_files + corpus_embds_files:
                if os.path.exists(file):
                    os.remove(file)

    # Run evaluation
    if args.eval and trainer.is_global_zero:
        scores = run_retrieve_evaluation(dm.dataset.relevance, retriever.prediction)
        trainer.print("-" * 40)
        trainer.print("Model:", colored(f"{args.model_name}", "red"))
        trainer.print("Task:", colored(f"{dm.task_name}", "magenta"))
        trainer.print("Save path:", colored(task_save_path, "yellow"))
        trainer.print("Retrieval evaluation:")
        trainer.print(scores)
        with open(os.path.join(task_save_path, constants.RETRIEVE_EVAL_FILENAME), "w") as f:
            json.dump(scores, f)
        trainer.print(os.path.join(task_save_path, constants.RETRIEVE_EVAL_FILENAME))
        return scores

    return
