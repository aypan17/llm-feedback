"""
Evaluate the agent based on a set of trajectories.

Usage: python scripts/evaluate.py -inp <input_trajectories_file> -ev <evaluator_type>
"""

import argparse
import random

import openai

from dotenv import load_dotenv
from toolemu.dataloader import DataLoader
from toolemu.evaluators import EVALUATORS
from toolemu.executors import FuncExecutor
from toolemu.utils import (
    llm_register_args,
    load_openai_llm_with_args,
    print_intermediate_result_and_stop,
)
from toolemu.utils.my_typing import *

load_dotenv()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--eval-type",
    "-ev",
    type=str,
    default="agent_safe",
    choices=EVALUATORS.keys(),
)
parser.add_argument("--split_by_errors", action="store_true")
args_, _ = parser.parse_known_args()

evaluator_class = EVALUATORS[args_.eval_type]
llm_register_args(parser, prefix="evaluator")
evaluator_class.register_args(parser)
DataLoader.register_args(parser)
FuncExecutor.register_args(
    parser, default_num_retries=5, default_batch_size=5, default_timeout=600
)
parser.add_argument("--output-file-prefix", "-out", type=str, default=None)
parser.add_argument("--eval-results-out-suffix", "-es", type=str, default="")
parser.add_argument("--random-seed", "-seed", type=int, default=42)

args = parser.parse_args()
random.seed(args.random_seed)

def split_by_errors(traj):
    trajs_split_by_errors = []
    split_points = []
    for i, step in enumerate(traj["intermediate_steps"]):
        if i == len(traj["intermediate_steps"]) - 1:
            continue
        simulator_thought = step[1][-1].lower()
        if (
            "enumerate all errors" in simulator_thought
            or "select the error" in simulator_thought
            or "simulate the error" in simulator_thought
        ):  
            split_points.append(i)
    for i in split_points:
        split_traj = traj.copy()
        split_traj["intermediate_steps"] = traj["intermediate_steps"][: i+1]
        trajs_split_by_errors.append(split_traj)
    trajs_split_by_errors.append(traj)
    return trajs_split_by_errors, len(split_points)


def main():
    trajs = DataLoader.from_args(args, return_mode="with_idx", item_name="trajectory")
    output_file_prefix = args.output_file_prefix or trajs.base_path
    output_path = f"{output_file_prefix}_eval{args.eval_results_out_suffix}_{args.eval_type}.jsonl"
    if args.split_by_errors:
        output_path = output_path.replace(".jsonl", "_split-by-error.jsonl")

    if args.critique_rounds > 0:
        raise ValueError("Evaluation does not support critique rounds yet.")
    evaluator_llm = load_openai_llm_with_args(args, prefix="evaluator")
    evaluator = evaluator_class.from_args(args, evaluator_llm)

    if evaluator._stop_at in ["preprocess", "prompt"]:
        result = evaluator({"trajectory": trajs[0]["item"]})
        print_intermediate_result_and_stop([result], evaluator._stop_at)

    def evaluate_trajec(traj_with_idx):
        traj_idx, traj = traj_with_idx["idx"], traj_with_idx["item"]
        if args.split_by_errors:
            trajs_split_by_errors, _ = split_by_errors(traj)
            result = []
            try:
                for i, split_traj in enumerate(trajs_split_by_errors):
                    #results = evaluator({"trajectory": split_traj})
                    results = [{}]
                    traj_result = results[0]
                    traj_result["eval_id"] = traj_idx
                    traj_result["num_errors"] = i
                    result.append(traj_result)
            except Exception as e:
                result = {"error": str(e)}
                return traj_with_idx, result
        else:
            try:
                results = evaluator({"trajectory": traj})
                result = results[0]
                result["eval_id"] = traj_idx
            except Exception as e:
                result = {"error": str(e)}
                return traj_with_idx, result
        return None, result

    runner = FuncExecutor.from_args(args)
    runner.run(evaluate_trajec, trajs, output_path, evaluate=True)
    print(
        "You may want to use scripts to convert the result jsonl file "
        f"{output_path} to json for easier reading."
    )


if __name__ == "__main__":
    main()
