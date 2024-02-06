import argparse
from collections import defaultdict
import tqdm
import re
import json
import datetime
import time

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import openai
import numpy as np

from toxicity import get_perspective_toxicity_scores, plot_toxicity_scores
from api_keys import ANTHROPIC_API_KEY, OPENAI_API_KEY

anthropic = Anthropic(ANTHROPIC_API_KEY)
openai.api_key = OPENAI_API_KEY

MAX_TRIES = 100
MODEL_STORE = {}


def query(
    model_name,
    user_prompt,
    system_prompt="You are a helpful assistant.",
    max_tokens=100,
    temperature=1.0,
    stop_sequences=[],
):
    if model_name.startswith("claude"):
        tries = 0
        while tries < MAX_TRIES:
            try:
                completion = anthropic.completions.create(
                    model=model_name,
                    prompt=f"{HUMAN_PROMPT} {system_prompt}\n{user_prompt}{AI_PROMPT}",
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences,
                )
                return completion.completion
            except Exception as e:
                tries += 1
                print("Error:", e)
                time.sleep(2)
    elif model_name.startswith("gpt"):
        tries = 0
        while tries < MAX_TRIES:
            try:
                completion = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop_sequences,
                )
                print(completion)
                return completion["choices"][0]["message"]["content"]
            except Exception as e:
                tries += 1
                print("Error:", e)
                time.sleep(2)
    else:
        raise ValueError(f"Unknown model: {model_name}.")


def judge_responses(model_name, judges, judge_prompt_fn, responses, args):
    if model_name == "random":
        return np.random.choice(len(responses))

    scores = defaultdict(int)
    idx = np.random.choice(len(judges), size=args.n_judges, replace=False)

    for judge in tqdm.tqdm(np.array(list(judges.items()))[idx]):
        user_prompt = judge_prompt_fn(
            judge, args.topic, args.objective, args.item_type, responses
        )
        tries = 0
        while tries < MAX_TRIES:
            try:
                out = query(model_name, user_prompt)
                match = re.search(r"\d+", out)
                scores[int(match.group()) - 1] += 1
                break
            except Exception as e:
                tries += 1
                print("Error:", e)
                time.sleep(2)

    # If there is a tie, choose randomly
    max_score = max(scores.values())
    winning_idxs = [i for i, score in scores.items() if score == max_score]
    return np.random.choice(winning_idxs), scores


def process_agent(agent, winning_agent, winning_response, i, args):
    init = i == 0
    if init and args.experiment == "optimization":
        with open("optimization/experiment_data.json", "r") as f:
            data = json.load(f)
        assert 0 <= args.agent_idx < len(data[args.seed]["init_responses"])
        winning_response = data[args.seed]["init_responses"][args.agent_idx]
    user_prompt = get_agent_prompt(
        agent,
        args.topic,
        args.objective,
        args.item_type,
        winning_agent,
        winning_response,
        init=init,
    )
    response = query(
        args.agent_model,
        user_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    response = process_response(response)
    if args.experiment == "reward_hacking":
        toxicity_score = get_perspective_toxicity_scores([response])
    else:
        toxicity_score = {"toxicity": [0]}

    return agent[0], response, toxicity_score["toxicity"][0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_model",
        choices=["claude", "gpt", "gpt-3.5-turbo"],
        required=True,
    )
    parser.add_argument(
        "--judge_model",
        choices=["claude", "gpt", "gpt-3.5-turbo", "random"],
        required=True,
    )
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--n_rounds", type=int, default=10)
    parser.add_argument("--agent_idx", type=int, default=-1)
    parser.add_argument("--n_judges", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def replace_model_name(name):
    if name == "claude":
        return "claude-1.1"
    elif name == "gpt":
        return "gpt-4-0613"
    else:
        return name


def run_experiment(agents, judges, args):
    agent_responses = defaultdict(list)
    filtered_responses = []
    all_toxicity_scores = []
    winning_agent = None
    winning_response = None
    for i in range(args.n_rounds):
        print(f"====== Round {i+1} ======")
        responses = []
        round_toxicity_scores = []

        for agent in agents.items():
            agent_name, response, toxicity = process_agent(
                agent, winning_agent, winning_response, i, args
            )
            responses.append(response)
            agent_responses[agent_name].append(response)
            round_toxicity_scores.append(toxicity)

        winning_idx = judge_responses(
            args.judge_model, judges, get_judge_prompt, responses, args
        )
        winning_agent = list(agents.items())[winning_idx]
        winning_response = responses[winning_idx]
        filtered_responses.append(winning_response)
        all_toxicity_scores.append(round_toxicity_scores)
        print(f"Winning response: {winning_response}")

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    data = {
        "agent_model": args.agent_model,
        "judge_model": args.judge_model,
        "agent_idx": args.agent_idx,
        "experiment": args.experiment,
        "n_rounds": args.n_rounds,
        "n_judges": args.n_judges,
        "seed": args.seed,
        "topic": args.topic,
        "item_type": args.item_type,
        "objective": args.objective,
        "agent_responses": agent_responses,
        "responses": filtered_responses,
        "toxicity_scores": all_toxicity_scores,
    }
    if args.experiment == "reward_hacking":
        if args.agent_idx != -1:
            name = f"rh_singleagent_results/agent-{args.agent_name}_{date}"
        else:
            name = f"rh_multiagent_results/agent-{args.agent_model}_filter-{args.judge_model}_{date}"
        plot_toxicity_scores(
            all_toxicity_scores, path=f"{name}.svg", labels=list(agents.keys())
        )
    else:
        name = f"optimization_results/agent-{args.agent_model}_{date}"

    with open(f"{name}.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    args.agent_model = replace_model_name(args.agent_model)
    args.judge_model = replace_model_name(args.judge_model)

    if args.experiment == "reward_hacking":
        from reward_hacking.utils import (
            init_experiment,
            get_agent_prompt,
            get_judge_prompt,
            process_response,
        )
    elif args.experiment == "optimization":
        from optimization.utils import (
            init_experiment,
            get_agent_prompt,
            get_judge_prompt,
            process_response,
        )
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}.")

    agents, judges, args = init_experiment(args)
    run_experiment(agents, judges, args)
