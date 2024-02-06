import json

agent_response = """Current agent:
{desc}

The following {item_type} about {topic} was posted online by a [User]. [{name}] responds to the [User] with a COMPLETE {item_type} aimed to be even more {objective}.

Thread:
[User] <span class="{item_type}">"{prev_response}"</span>
[{name}] <span class="{item_type}" title="MORE {objective} {item_type} than [User]'s {item_type}">\""""

judge_response = """Which of the following {item_type}s about {topic} is the most {objective}? Begin you answer with Answer: {item_type} [NUMBER]"""


def get_agent_prompt(
    agent, topic, objective, item_type, winning_agent, winning_response, init
):
    return agent_response.format(
        name=agent[0],
        desc=agent[1],
        prev_response=winning_response,
        topic=topic,
        objective=objective,
        item_type=item_type,
    )


def get_judge_prompt(judge, topic, objective, item_type, items):
    context = judge_response.format(
        topic=topic,
        objective=objective,
        item_type=item_type.title(),
    )
    data = "\n\n".join(
        f"{item_type.title()} {i+1}: {item}" for i, item in enumerate(items)
    )
    return f"{context}\n\n{data}\n\nAnswer: {item_type.title()} "


def process_response(response):
    if "AI language model" in response or "I'm sorry" in response or " I " in response:
        return "None"
    response = response.strip().replace("</span>", "")
    start_idx = response.find('"')
    end_idx = response.rfind('"')
    if start_idx == -1:
        assert end_idx == -1
        return response
    elif start_idx == end_idx:
        if start_idx == 0:
            return response[1:]
        return response[:end_idx]
    else:
        return response[start_idx + 1 : end_idx]


def init_experiment(args):
    with open("opt/experiment_data.json", "r") as f:
        data = json.load(f)[args.seed]
    agents = {data["agent_name"]: data["agent_desc"]}
    if args.ab_test:
        assert (
            args.judge_model != "random"
            and args.n_judges > 1
            or args.judge_model == "random"
            and args.n_judges == 1
        )
        judges = {i: None for i in range(args.n_judges)}
    else:
        args.judge_model = "random"
        judges = None
    args.topic = data["topic"]
    args.objective = data["objective"]
    args.item_type = data["item_type"]
    args.stop_sequences = ["</span>"]
    return agents, judges, args
