"""
Emulate the agent's tool-use trajectories.

Usage: python scripts/emulate.py -inp <input_cases_file> -atp <agent_type> -stp <simulator_type> -ctp <case_type>
"""

import argparse
import random

import anthropic
from anthropic import Anthropic
import openai.error
from dotenv import load_dotenv
from toolemu.agent_executor_builder import build_agent_executor
from toolemu.agents import AGENT_TYPES, SIMULATORS
from toolemu.dataloader import DataLoader
from toolemu.executors import FuncExecutor
from toolemu.tools import *
from toolemu.utils import (
    case_to_input_dict,
    filter_keys,
    get_toolkit_names,
    llm_register_args,
    load_openai_llm_with_args,
    replace_agent_action_with_list,
)
from toolemu.utils.my_typing import *

load_dotenv()
ROLES = ["agent", "simulator"]

parser = argparse.ArgumentParser()
for role in ROLES:
    llm_register_args(parser, prefix=role)
DataLoader.register_args(parser)
FuncExecutor.register_args(
    parser, default_num_retries=5, default_batch_size=1, default_timeout=1800
)
parser.add_argument(
    "--output-file-prefix",
    "-outpre",
    type=str,
    default="./dumps/traj",
)
parser.add_argument(
    "--output-file-suffix",
    "-outsuf",
    type=str,
    default="",
)
parser.add_argument(
    "--agent-type",
    "-atp",
    type=str,
    default="naive",
    choices=AGENT_TYPES,
)
parser.add_argument(
    "--simulator-type",
    "-stp",
    type=str,
    default="std_thought",
    choices=list(SIMULATORS.keys()),
)
parser.add_argument("--max-iterations", "-mi", type=int, default=10)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--random-seed", "-seed", type=int, default=42)
parser.add_argument("--num_critique_steps", "-ncs", type=int, default=0)
parser.add_argument("--p_error", "-pe", type=float, default=0.0)
parser.add_argument("--max_errors", "-me", type=int, default=0)
parser.add_argument("--allow_all_tools", "-aat", action="store_true")
parser.add_argument("--cases", type=str, default="")

args = parser.parse_args()
random.seed(args.random_seed)

def main():
    if args.agent_model_name.startswith("gpt-3.5-turbo-instruct"):
        args.agent_max_tokens=1024
    llms = {role: load_openai_llm_with_args(args, prefix=role) for role in ROLES}
    if args.cases == "":
        cases = DataLoader.from_args(args, return_mode="with_idx", item_name="case")
    else:
        idx = [int(i) for i in args.cases.split(",")]
        cases = DataLoader.from_args(args, return_mode="with_idx", item_name="case", selected_indexes=idx)
    runner = FuncExecutor.from_args(args)
    output_path = (
        f"{args.output_file_prefix}_sim_{args.simulator_type}_error_{args.p_error}"
        f"_agent_{args.agent_model_name}_{args.agent_type}{args.output_file_suffix}.jsonl"
    )
    def generate_trajectory(case_with_idx):
        case_idx, case = case_with_idx["idx"], case_with_idx["item"]
        agent_executer = build_agent_executor(
            get_toolkit_names(case),
            llms["agent"],
            llms["simulator"],
            agent_type=args.agent_type,
            simulator_type=args.simulator_type,
            verbose=args.verbose,
            max_iterations=args.max_iterations,
            num_critique_steps=args.num_critique_steps,
            p_error=args.p_error,
            max_errors=args.max_errors,
            allow_all_tools=args.allow_all_tools,
            case_idx=case_idx,
        )
        inputs = filter_keys(case_to_input_dict(case), agent_executer.input_keys)
        try:
            outputs = agent_executer(inputs)
            failed_item = None
        except (openai.error.InvalidRequestError) as e:
            print(f"{case_idx}: {str(e)}")
            outputs = {"error": str(e)}
            failed_item = case_with_idx
        outputs = replace_agent_action_with_list(outputs)  # ad-hoc fix
        outputs["case"] = case
        outputs["case_idx"] = case_idx
        if args.agent_model_name.startswith("claude"):
            outputs["agent"] = f"{llms['agent'].model}_{args.agent_type}"
        else:
            outputs["agent"] = f"{llms['agent'].model_name}_{args.agent_type}"
        return failed_item, outputs

    runner.run(generate_trajectory, cases, output_path, args=args)
    print(
        "You may want to use scripts to convert the result jsonl file "
        f"{output_path} to json for easier reading."
    )


if __name__ == "__main__":
    main()
