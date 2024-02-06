import json

import numpy as np
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chat_models.base import BaseChatModel
from toolemu.agents import (
    SIMULATORS,
    AgentExecutorWithToolkit,
    ZeroShotAgentWithToolkit,
)
from toolemu.tools import get_toolkits_by_names, get_all_possible_toolkits
from toolemu.utils.my_typing import *


def build_agent_executor(
    toolkits: List[str],
    agent_llm: BaseLanguageModel,
    simulator_llm: BaseLanguageModel,
    critiquer_llm: Optional[BaseLanguageModel] = None,
    num_critique_steps: int = 0,
    max_allowed_steps: int = 3,
    p_error: float = 0.0,
    max_errors: int = 0,
    allow_all_tools: bool = False,
    agent_type: str = "naive",
    simulator_type: str = "std_thought",
    verbose: bool = True,
    return_intermediate_steps: bool = True,
    max_iterations: int = 10,
    callback_manager: Optional[BaseCallbackManager] = None,
    case_idx: Optional[int] = None,
) -> Type[AgentExecutorWithToolkit]:
    toolkits = get_toolkits_by_names(toolkits)
    if allow_all_tools:
        all_toolkits = get_all_possible_toolkits()
    else: 
        all_toolkits = None
    agent = ZeroShotAgentWithToolkit.from_llm_and_toolkits(
        toolkits=toolkits,
        llm=agent_llm,
        agent_type=agent_type,
        use_chat_format=isinstance(agent_llm, BaseChatModel),
    )

    if max_errors > 0:
        assert p_error == 0
        if agent_type == 'forceful':
            first_error_step = 1
        else:
            with open("assets/num_steps.json", "r") as f:
                num_steps = json.load(f)
                num_steps = num_steps[f"{agent_llm.model_name}-{agent_type}"][str(case_idx)]
            first_error_step = 5 if num_steps > 3 else 1
        #first_error_step = np.random.choice(max(2, num_steps-2))+1
        #first_error_step = np.random.choice(max(1, num_steps))
        #first_error_step = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.25, 0.05])
    else:
        first_error_step = -1

    executor_class = SIMULATORS[simulator_type]
    if simulator_type != "normal":
        agent_executer = executor_class.from_agent_and_toolkits(
            toolkits=toolkits,
            agent=agent,
            verbose=verbose,
            llm_simulator=simulator_llm,
            llm_critiquer=critiquer_llm,
            p_error=p_error,
            max_errors=max_errors,
            first_error_step=first_error_step,
            num_critique_steps=num_critique_steps,
            max_allowed_steps=max_allowed_steps,
            all_toolkits=all_toolkits,
            use_chat_format=isinstance(simulator_llm, BaseChatModel),
            return_intermediate_steps=return_intermediate_steps,
            max_iterations=max_iterations,
            callback_manager=callback_manager,
        )
    else:
        agent_executer = executor_class.from_agent_and_toolkits(
            toolkits=toolkits,
            agent=agent,
            verbose=verbose,
            max_iterations=max_iterations,
            callback_manager=callback_manager,
        )
    return agent_executer
