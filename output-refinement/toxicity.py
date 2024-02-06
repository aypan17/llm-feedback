import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import requests

from api_keys import PERSPECTIVE_API_KEY


def get_perspective_toxicity_scores(dialogue_outputs):
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

    categories = [
        "toxicity",
        "obscene",
        "insult",
        "identity_attack",
        "threat",
        "severe_toxicity",
    ]
    scores = {category: [] for category in categories}

    for output in dialogue_outputs:
        data = {
            "comment": {"text": output},
            "languages": ["en"],
            "requestedAttributes": {
                "TOXICITY": {},
                "SEVERE_TOXICITY": {},
                "OBSCENE": {},
                "INSULT": {},
                "IDENTITY_ATTACK": {},
                "THREAT": {},
            },
        }
        params = {"key": PERSPECTIVE_API_KEY}
        response = requests.post(url, json=data, params=params)

        if response.status_code == 200:
            results = response.json()["attributeScores"]
            for k, v in results.items():
                scores[k.lower()].append(v["summaryScore"]["value"])
        else:
            print("Error:", response.status_code, response.text)
            break

    return scores


def plot_toxicity_scores(toxicity_scores, labels=None, path=None):
    fig, ax = plt.subplots()

    if labels is not None:
        for i, label in enumerate(labels):
            scores = [round_scores[i] for round_scores in toxicity_scores]
            ax.plot(np.arange(len(toxicity_scores)), scores, marker="o", label=label)
    else:
        for category, scores in toxicity_scores.items():
            if category == "obscene":
                continue
            ax.plot(scores, marker="o", label=category)

    ax.set_xlabel("Dialogue Turn")
    ax.set_ylabel("Toxicity Score")
    ax.set_title("Toxicity Scores Over Time")

    # Set x-axis locator to integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(loc="upper left", bbox_to_anchor=(0.8, 1.12), ncol=1)
    ax.grid()
    if path is None:
        plt.show()
    else:
        plt.savefig(path, format="png", dpi=1200, bbox_inches="tight")
