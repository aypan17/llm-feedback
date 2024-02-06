import json
import os
import os.path as osp
import datetime

import pandas as pd

from .my_typing import *


def create_folder(file_path: str) -> bool:
    """Create folders for the file_path if the folders do not exist.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the folder is created, False if the folder already exists.
    """
    folder_path = osp.dirname(file_path)
    if not osp.exists(file_path):
        os.makedirs(folder_path, exist_ok=True)
        return True
    return False


def remove_file(file_path: str) -> None:
    """Remove a file."""
    if osp.exists(file_path):
        os.remove(file_path)


def append_file(
    file_path: str,
    text: str,
    mode: str = "a",
    seperator: str = "\n\n########################\n",
) -> None:
    """Append text to a file."""
    if create_folder(file_path):
        mode = "w"
    with open(file_path, mode=mode) as f:
        f.write(seperator)
        f.write(text)


def append_jsonl(
    file_path: str,
    content: Any,
    mode: str = "a",
) -> None:
    """Append a json object to a jsonl file."""
    if create_folder(file_path):
        mode = "w"
    with open(file_path, mode=mode) as f:
        f.write(json.dumps(content) + "\n")


def parse_and_write(results_json, args):
    case_idx = results_json["case_idx"]
    agent = results_json["agent"]
    results_str = f"Case: {case_idx}\nToolkits: {results_json['case']['Toolkits']}\nPrompt: {results_json['input']}\n"
    results_str += f"Agent: {agent}\n"
    results_str += f"Num errors: {args.max_errors}\n"
    results_str += f"p_error: {args.p_error}\n\n"
    try:
        sim_log = ""
        for i, step in enumerate(results_json["intermediate_steps"]):
            if i == len(results_json["intermediate_steps"]) - 1:
                results_str += f"Thought: {step[0][-1]}"
            else:
                results_str += f"Thought: {step[0][-1]}\n"
                results_str += f"Observation: {step[1][0]}\n"
                sim_log += step[1][-1].split("\n")[-3] + "\n"

        results_str += f"\n\n=============\n\n{sim_log}\n\n"
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # format case_idx to have 3 digits
        case_idx = str(case_idx).zfill(3)
        # make sure the folder exists
        if not osp.exists(f"./results_{agent}"):
            os.makedirs(f"./cases_{agent}")
        with open(f"./results_{agent}/case-{case_idx}_{date}.txt", "w") as f:
            f.write(results_str)
    except:
        print("Error writing results")


def write_json(content: Any, file_path: str, indent=4) -> None:
    """Write a json object to a json file."""
    create_folder(file_path)
    with open(file_path, "w") as f:
        json.dump(content, f, indent=indent)


def read_json(file_path: str) -> Any:
    with open(file_path, "r") as f:
        return json.load(f)


def read_jsonl(file_path: str) -> List[Any]:
    all_data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                all_data.append(data)
    return all_data


def read_file(file_path: str) -> Any:
    suffix = file_path.split(".")[-1]
    if suffix == "json":
        return read_json(file_path)
    elif suffix == "jsonl":
        return read_jsonl(file_path)
    elif suffix in ["txt", "log", "md"]:
        with open(file_path, "r") as f:
            return f.readlines()
    elif suffix in ["csv"]:
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unknown file type: {suffix}")


def remove_file_extension(file_path: str) -> str:
    """Remove the file extension suffix of a file path."""
    return ".".join(file_path.split(".")[:-1])


def replace_file_extension(file_path: str, new_extension: str) -> str:
    """Replace the file extension suffix of a file path."""
    return remove_file_extension(file_path) + f".{new_extension}"
