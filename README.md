# :repeat: :car: :chart_with_upwards_trend: :chart_with_downwards_trend: Test-time Feedback Loops Drive In-Context Reward Hacking in LLMs
This repository contains code for the paper [Test-time Feedback Loops Drive
In-context Reward Hacking in LLMs](drive.google.com). The code is split up into 
two separate directories, `output-refinement/` and `policy-refinement/`, 
corresponding to each experiment setup. All experiments are run by first
`cd`-ing into the relevant directory.

![An example feedback loop](splash.pdf)

## Output-refinement

### Setup
- Install [choix](https://github.com/lucasmaystre/choix)
- Obtain an [OPENAI_API_KEY](https://platform.openai.com/docs/quickstart?context=python), a [Perspective API key](https://developers.perspectiveapi.com/s/docs-get-started), and an [ANTHROPIC_API_KEY](https://docs.anthropic.com/claude/docs/getting-access-to-claude) (optional).
- Add your api keys to `api_keys.py`

### Running the code

**Generation:** 
Obtain outputs for each experiment with 
```
# Use async_filtering.py if your APIs have higher rate limits; otherwise use filtering.py
python3 [async_filtering.py OR async_filtering.py] 
    --experiment [optimization OR reward_hacking] # Selects the experiment to run
    --n_rounds 11 # Number of cycles of feedback
    --agent_model [gpt or gpt-3.5-turbo or claude] # The LLM used to produce outputs
    --judge_model [random OR gpt OR gpt-3.5-turbo OR claude] # The model used to evaluate outputs
    --n_judges 3 # The number of judges used to evaluate outputs
    --seed 0 # The experiment subtask to run
    --agent_idx [-1 OR 0 OR 1 OR 2 OR 3] # The agent to run; leave as -1 for multi-agent experiment
    --ab_test # Optional; set if running the experiment with environment feedback
```
**Evaluation:**
Score the outputs along the objectives with
```
python3 pairwise_voting.py 
    --data_json JSON_FILE_OF_OUTPUTS 
    --judge_model [random OR gpt OR gpt-3.5-turbo OR claude] # The model used to evaluate completions
```

## Policy-refinement
The code is adapted from [ToolEmu](https://github.com/ryoungj/ToolEmu). For our experiments,
we made several changes to allow for server-side API errors.

**Enabling error generation:** 
- Added `policy-refinement/assets/generate_errors.py` to add errors to the [toolkits provided in ToolEmu](https://github.com/ryoungj/ToolEmu/blob/main/assets/all_toolkits.json).
- Modify `policy-refinement/toolemu/agent_executor_builder.py` to determine when errors are injected.
- Modify `policy-refinement/toolemu/agents/virtual_agent_executor.py` to call the error simulator.
- Create prompt for error simulator in `policy-refinement/toolemu/prompts/simulator/error.py` 

**Enabling error evaluation:**
- Added constraint violation prompt in `policy-refinement/toolemu/prompts/evaluator/agent_constraint_evaluator.py` based off of `policy-refinement/toolemu/prompts/evaluator/agent_constraint_evaluator.py`
- Add option to evaluate trajectories based on segments in `policy-refinement/scripts/evaluate.py`

### Setup
Follow the [ToolEmu](https://github.com/ryoungj/ToolEmu) installation instructions in `policy-refinement/README.md`

### Running the code
**Generation:** 
Follow the [ToolEmu](https://github.com/ryoungj/ToolEmu) emulation code in `policy-refinement/README.md`. Our code adds two new options in `policy-refinement/scripts/emulate.py`:
- `--p_error`, which controls the likelihood of errors at each step.
- `--max_errors`, which controls the maximum number of errors an environment can have.

**Evaluation:**
Follow the [ToolEmu](https://github.com/ryoungj/ToolEmu) evaluation code in `policy-refinement/README.md`. Our code adds one new option (for constraint violation evaluation) in `policy-refinement/scripts/evaluate.py`:
- `--split_by_errors`, which switches evaluation to evaluate trajectories based on segments. 

## Citation
Feel free to cite our paper:
```
@article{pan2024llmfeedback
    author = {Pan, Alexander and Jones, Erik and Jagadeesan, Meena and Steinhardt, Jacob},
    title = {Test-time Feedback Loops Drive In-Context Reward Hacking in LLMs},
    journal= {arXiv},
    year = {2024}
}
```