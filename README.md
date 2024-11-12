# Voyage AI Evaluation

Voyage AI retrieval evaluation codebase.


## Features
- API models.
- Save / load of intermediate embeddings / reranker predictions.

## Environment

Create a new conda environment
```
conda create -n voyage-eval python=3.10
conda activate voyage-eval
```

Install the dependencies
```
pip install -r requirements.txt
```

While the codebase should work with recent versions of its dependencies, the following are the key package versions that it has been tested on.
- `torch==2.3.1+cu121`
- `lightning==2.2.5`
- `sentence_transformers==3.0.0`


## Usage

Example command
```bash
python main.py \
    --bf16 \
    --do_retrieve \
    --model_type voyage_api \
    --batch_size 16 \
    --model_name voyage-3 \
    --api_key <your_api_key> \
    --task_name TEST_TASKS \
    --data_path data/ \
    --save_path output/retrieve/voyage-3 \
    --save_prediction \
    --add_instruct
```

```bash
python main.py \
    --bf16 \
    --do_retrieve \
    --model_type openai \
    --batch_size 16 \
    --model_name text-embedding-3-large \
    --api_key <your_api_key> \
    --task_name TEST_TASKS \
    --data_path data/ \
    --save_path output/retrieve/text-embedding-3-large \
    --save_prediction
```

Explanation of the arguments:
- `--data_path` base path where the datasets are stored.
- `--task_name` can be a list of tasks or task groups, seperated by `,`; task groups are defined [here](voyage_evaluation/dataset/tasks.py);
if not specified, load all directories/files under `data_path` (but not recursively).
- `--overwrite` when this flag is set, exising results in the output path will be ignored and overwritten; 
otherwise, the code will load exising results and will not run the model again.
- `--save_embds` save the embeddings to files, useful when evaluating APIs.
- `--load_embds` load the computed embeddings from files and resume computation.
- `--no-eval` disable evaluation, set this flag when you only want to run inference.

Please check [main.py](main.py) for full definition of arguments.

