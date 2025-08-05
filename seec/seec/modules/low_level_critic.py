from ..services.llm_service import get_model
from ..services.state import SeecState
from .states import Accept

prompt = """You are critic in Actor-Critic paradigm. You are working on manipulation task. Your task is to verify validity and corectness of low-level constraints for each of the stage in task-planning pipeline. 

High-level plan is provided via several stages, which are described with short and concise description. Assume it is valid, and it's execution would result in the completion of the task, defined by user request.

The low-level plan is specified via constraint functions in Python. There are two kinds of constraints: 
- Sub-goal constraints. Those constraints that must be satisfied **at the end of the stage**
- Path constraints. Those constraints must be satisfied **within the stage**.

Based on the planning description below, analyze whether:
1. Satisfying sub-goal constraint in the end would make next stage ready to execute.
2. Satisfying path constraint would allow the robot to complete the stage.
3. No constraints contradict each other.
4. For each stage, tehre is 0 or more sub-goal constraints and 0 or more path constraints. 
5. There are no path constraints when manipulating deformable objects (e.g., clothing, towels). 
7. For grasping stage, there is **only one** sub-goal constraint that associates the end-effector with a keypoint.

If all constraints seem valid and sound, call the `Accept` tool. Otherwise, provide detailed feedback on what needs to be improved."""



def low_level_critic(state: SeecState, config):
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"Original requirements: {state.get('requirements')}",
        },
        {
            "role": "user",
            "content": f"Planning description: {state.get('ai_requirements')}",
        },
    ]
    model = get_model(config, "4o", "reasoning_critique").with_structured_output(Accept)
    response = model.invoke(messages)
    verified = response.accept
    if verified:
        return {
            # "print_stream": {"verification": response.logic},
            "messages": [
                {"role": "user", "content": response.logic},
            ],
            "plan_verified": True,
        }
    else:
        return {
            # "print_stream": {"verification": response.logic},
            "messages": [
                {"role": "user", "content": response.logic},
            ],
            "plan_verified": False,
        }