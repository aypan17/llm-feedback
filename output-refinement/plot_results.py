import os
import json
from collections import defaultdict
import argparse

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import choix


def agg_mva_data(data, agg_method, k, split_by_agent):
    if split_by_agent:
        for agent in data:
            moving_avgs = []
            for i in range(len(data[agent]) - k + 1):
                window = data[agent][i : i + k]
                moving_avgs.append(np.mean(window))
            data[agent] = np.array(moving_avgs)
        return data
    else:
        if len(data) == 0:
            return []
        # Aggregate over actors for each round based on agg_method
        if agg_method == "max":
            aggregated = np.max(data, axis=1)
        elif agg_method == "min":
            aggregated = np.min(data, axis=1)
        elif agg_method == "median":
            aggregated = np.median(data, axis=1)
        elif agg_method == "mean":
            aggregated = np.mean(data, axis=1)
        else:
            raise NotImplementedError

        moving_avgs = []
        # Calculate moving averages on aggregated data
        for i in range(len(aggregated) - k + 1):
            window = aggregated[i : i + k]
            moving_avgs.append(np.mean(window))
        return np.array(moving_avgs)


def update_scores(data, args):
    for topic, scores in data.items():
        scores_ = agg_mva_data(scores, args.agg, args.k, args.split_by_agent)
        data[topic] = scores_


def update_data(data, judge_model, args):
    def agent_to_name(agent):
        if agent == "O":
            return "OccupyDemocrats"
        elif agent == "F":
            return "FoxNews"
        elif agent == "B":
            return "Breitbart"
        elif agent == "M":
            return "MSNBC"
        elif agent == "gpt-3.5-turbo":
            return "GPT-3.5-Turbo"
        elif agent == "gpt-4-0613":
            return "GPT-4"
        elif agent == "claude-2":
            return "Claude-2"
        else:
            return agent

    for filename in tqdm(os.listdir(args.directory)):
        if filename.endswith(".json"):
            with open(os.path.join(args.directory, filename), "r") as f:
                content = json.load(f)
            if not content.get("judge_model").startswith(judge_model):
                continue
            topic = content.get("seed", "")
            if args.toxicity:
                for i, agent_ in enumerate(content.get("agent_responses")):
                    agent = agent_to_name(agent_)
                    toxicity_scores = content.get("toxicity_scores", [])[: args.cutoff]
                    if args.split_by_agent:
                        data[topic][agent] = [t[i] for t in toxicity_scores]
                    else:
                        data[topic].append([t[i] for t in toxicity_scores])
            else:
                comparisons = {}
                for key in content:
                    if key == "comparisons":
                        for agent_ in content[key]:
                            comparisons[agent_] = content[key][agent_]
                    elif key.startswith("comparisons"):
                        d_ = content[key]
                        comparisons[list(d_.keys())[0]] = list(d_.values())[0]

                for agent_ in comparisons:
                    pairs = []
                    redone = set()
                    for p in comparisons[agent_]:
                        p0 = int(p[0].split("_")[1])
                        p1 = int(p[1].split("_")[1])
                        if p0 < args.cutoff and p1 < args.cutoff:
                            if (p1, p0) not in redone:
                                pairs.append((p0, p1))
                                redone.add((p0, p1))
                    params = choix.opt_pairwise(args.cutoff, pairs)
                    if args.split_by_agent:
                        data[topic][agent_to_name(agent_)] = params
                    else:
                        data[topic].append(params)
    for topic in data:
        if args.split_by_agent:
            for agent in data[topic]:
                data[topic][agent] = np.transpose(np.array(data[topic][agent]))
        else:
            data[topic] = np.transpose(np.array(data[topic]))


def plot(ax, raw_data, args, label=""):
    if args.split_by_agent:
        agents_to_data = defaultdict(list)
        for topic in raw_data:
            for agent in raw_data[topic]:
                agents_to_data[agent].append(raw_data[topic][agent])
        for agent in ["OccupyDemocrats", "FoxNews", "Breitbart", "MSNBC"]:
            data = np.array(agents_to_data[agent])
            y = np.mean(data, axis=0)
            yerr = np.std(data, axis=0) / np.sqrt(len(data))

            x_avgs = np.arange(len(data[0]))
            ax.plot(x_avgs, y, label=agent)
            ax.fill_between(x_avgs, y - yerr, y + yerr, alpha=0.2)
    else:
        data = []
        for topic in raw_data:
            data.append(raw_data[topic])
        data = np.array(data)
        y = np.mean(data, axis=0)
        yerr = np.std(data, axis=0) / np.sqrt(len(data))

        x_avgs = np.arange(len(data[0]))
        ax.plot(x_avgs, y, label=label)
        ax.fill_between(x_avgs, y - yerr, y + yerr, alpha=0.2)

    plt.xticks(np.arange(0, len(x_avgs) + 1, 5), fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Dialogue Turn", fontsize=20)
    plt.ylabel(
        f"{'Perspective API Score' if args.toxicity else 'Engagement Score'}",
        fontsize=20,
    )
    #plt.ylim(0, 1)
    plt.xlim(-0.2,10.2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="results/")
    parser.add_argument(
        "--agg", type=str, choices=["max", "min", "median", "mean"], default="mean"
    )
    parser.add_argument("--toxicity", action="store_true")
    parser.add_argument("--split_by_agent", action="store_true")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--cutoff", type=int, default=11)
    args = parser.parse_args()

    # Initialize the data dictionary
    if args.split_by_agent:
        random_data = defaultdict(lambda: {})
        gpt_data = defaultdict(lambda: {})
    else:
        random_data = defaultdict(lambda: [])
        gpt_data = defaultdict(lambda: [])

    # Define the directory where the json files are located
    update_data(random_data, "random", args)
    update_data(gpt_data, "gpt", args)
    update_scores(random_data, args)
    update_scores(gpt_data, args)

    fig, ax = plt.subplots(figsize=(8, 4))
    # if len(gpt_data) > 0:
    #     plot(ax, random_data, args, label="Random")
    #     plot(ax, gpt_data, args, label="GPT-3.5")
    # else:
    print(random_data)
    print(gpt_data)
    plot(ax, gpt_data, args)

    plt.legend(
        loc="lower right",
        bbox_to_anchor=(0.95, 0),
        ncol=1,
        fontsize=10,
        title_fontsize=10,
        title="Agent Persona",
    )

    title = f"{'Toxicity' if args.toxicity else 'Engagement'} averaging over 100 topics"
    if not args.split_by_agent:
        title += f" and {len(random_data[0])} agents"
    plt.title(title, fontsize=20)
    plt.savefig(
        f"{'toxicity' if args.toxicity else 'engagement'}_scores.pdf",
        dpi=1200,
        format="pdf",
        bbox_inches="tight",
    )
