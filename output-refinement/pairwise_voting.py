from dataclasses import dataclass
import json
import argparse
from collections import defaultdict
import itertools
import tqdm

import openai
from anthropic import Anthropic
import asyncio

from filtering import judge_responses
from async_filtering import async_judge_responses
from api_keys import ANTHROPIC_API_KEY, OPENAI_API_KEY

anthropic = Anthropic(ANTHROPIC_API_KEY)
openai.api_key = OPENAI_API_KEY

@dataclass
class Item:
    id: str
    text: str

    def __str__(self):
        return f"{self.id}: {self.text}"


def llm_vote_fn(item1, item2, args):
    _, scores = judge_responses(
        args.judge_model, args.judges, args.judge_prompt, [item1.text, item2.text], args
    )
    votes = []
    for vote, count in scores.items():
        if vote == 0:
            votes.extend([(item1.id, item2.id) for _ in range(count)])
        elif vote == 1:
            votes.extend([(item2.id, item1.id) for _ in range(count)])
    return votes


async def llm_avote_fn(item1, item2, args):
    _, scores = await async_judge_responses(
        args.judge_model, args.judges, args.judge_prompt, [item1.text, item2.text], args
    )
    votes = []
    for vote, count in scores.items():
        if vote == 0:
            votes.extend([(item1.id, item2.id) for _ in range(count)])
        elif vote == 1:
            votes.extend([(item2.id, item1.id) for _ in range(count)])
    return votes


def perform_ranking(items, vote_fn, args):
    votes = []
    for i, j in tqdm.tqdm(itertools.combinations(range(len(items)), 2)):
        if i == j:
            continue
        # Schedule the coroutine for execution and return a task
        votes.extend(vote_fn(items[i], items[j], args))
    return votes


async def async_perform_ranking(items, vote_fn, args):
    votes = []
    tasks = []

    semaphore = asyncio.Semaphore(args.max_concurrent_tasks)

    async def sem_task(item1, item2, args):
        async with semaphore:  # This will block if there are 60 tasks already running
            return await vote_fn(item1, item2, args)

    progress_bar = tqdm.tqdm(total=len(items) * (len(items) - 1) / 2)

    for i, j in itertools.combinations(range(len(items)), 2):
        if i == j:
            continue
        # Schedule the coroutine for execution and return a task
        task = asyncio.create_task(sem_task(items[i], items[j], args))
        task.add_done_callback(lambda _: progress_bar.update(1))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    progress_bar.close()

    for result in results:
        votes.extend(result)

    return votes


def vote(items, args):
    comparisons = {}
    for corp in tqdm.tqdm(items):
        comparisons[corp] = perform_ranking(items[corp], llm_vote_fn, args)
    return comparisons


async def async_vote(items, args):
    comparisons = {}
    for corp in items:
        comparisons[corp] = await async_perform_ranking(items[corp], llm_avote_fn, args)
    return comparisons


def load_items(args):
    items = defaultdict(list)
    with open(f"{args.data_json}.json", "r") as f:
        data = json.load(f)
        for corp in data["agent_responses"]:
            for i, response in enumerate(data["agent_responses"][corp]):
                if i > args.cutoff:
                    continue
                items[corp].append(Item(id=f"{corp}_{i}", text=response))
    args.topic = data["topic"]
    args.experiment = data["experiment"]
    args.objective = data.get("objective", None)
    args.item_type = data.get("item_type", "tweet")
    return items, data


if __name__ == "__main__":
    MAX_TRIES = 10
    args = argparse.ArgumentParser()
    args.add_argument("--data_json", type=str, required=True)
    args.add_argument("--n_judges", type=int, default=3)
    args.add_argument("--judge_model", type=str, default="gpt-3.5-turbo")
    args.add_argument("--cutoff", type=int, default=11)
    args.add_argument("--max_concurrent_tasks", type=int, default=100)
    args.add_argument("--no_async", action="store_true")
    args = args.parse_args()

    items, data = load_items(args)
    if args.experiment == "reward_hacking" or args.experiment == "tweet":
        from reward_hacking.utils import get_judge_prompt

        args.judges = {i: None for i in range(args.n_judges)}
        args.judge_prompt = get_judge_prompt
    elif args.experiment == "optimization":
        from optimization.utils import get_judge_prompt

        args.judges = {i: None for i in range(args.n_judges)}
        args.judge_prompt = get_judge_prompt
    if args.no_async:
        comparisons = vote(items, args)
    else:
        comparisons = asyncio.run(async_vote(items, args))

    data["items"] = [
        [str(item) for item in corp_items] for corp_items in items.values()
    ]
    data[f"comparisons_{args.judge_model}"] = comparisons
    with open(f"{args.data_json}.json", "w") as f:
        json.dump(data, f, indent=2)
