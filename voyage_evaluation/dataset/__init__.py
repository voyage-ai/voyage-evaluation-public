import os
from typing import Optional
from .retrieve import RetrieveDatasetEvalType
from .tasks import TASK_GROUPS


def get_retrieve_dataset(
    task_name: str,
    data_path: Optional[str] = None,
    data_type: str = "eval",
    **kwargs,
):
    if data_path is None:
        raise ValueError(f"data_path must be specified for task {task_name}.")
    if data_type == "eval":
        return RetrieveDatasetEvalType(task_name=task_name, data_path=data_path, **kwargs)
    elif data_type == "train":
        return RetrieveDatasetTrainType(task_name=task_name, data_path=data_path, **kwargs)
    else:
        raise ValueError(f"data_type {data_type} is not valid.")


def get_task_list(task_name: Optional[str], data_path: Optional[str] = None, data_type: str = "eval"):
    """
    Get the list of tasks.
    If task_name is not None, return the list of tasks specified by task_name.
    If task_name is None, return the list of tasks in data_path.
    """
    if task_name is not None:
        task_list = []
        for name in task_name.split(","):
            if name in TASK_GROUPS:
                task_list += TASK_GROUPS[name]
                if data_type == "train":
                    raise ValueError(f"Task group {name} is not compatible with `data_type=train`.")
            else:
                task_list.append(name)
        if data_path is not None:
            for task in task_list:
                if not os.path.exists(os.path.join(data_path, task)):
                    raise FileNotFoundError(f"Task {task} does not exist in {data_path}.")
    else:
        task_list = sorted(os.listdir(data_path))
        if data_type == "eval":
            task_list = [task for task in task_list if os.path.isdir(os.path.join(data_path, task))]
        elif data_type == "train":
            task_list = [task for task in task_list if os.path.isfile(os.path.join(data_path, task))]
    return task_list
