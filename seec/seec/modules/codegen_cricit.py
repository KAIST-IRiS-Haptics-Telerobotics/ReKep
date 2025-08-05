from ..services.llm_service import get_model
from ..services.state import SeecState
from .states import Accept

prompt = """You are critic in Actor-Critic paradigm. You are working on manipulation task. Your task is to verify validity and corectness of generated code for low-level constraints for each of the stage in task-planning pipeline. 

High-level plan is provided via several stages, which are described with short and concise description. Assume it is valid, and it's execution would result in the completion of the task, defined by user request.

The low-level plan is specified via constraint functions in Python. There are two kinds of constraints: 
- Sub-goal constraints. Those constraints that must be satisfied **at the end of the stage**
- Path constraints. Those constraints must be satisfied **within the stage**.
Assume that the plan is valid, and it should be flawlessly executed by the code.


Based on the provided code, analyze whether:
1. Output must not containt any import statements.
2. Each function precisely implements the constraints described in the prompt.
3. Execution of the code would not result in any runtime errors.
4. Each constraint takes a dummy end-effector point and a set of keypoints as input and returns a numerical cost, where the constraint is satisfied if the cost is smaller than or equal to zero. 
5. There are no "if" statements in constraints. 
6. Inputs to the constraints are as follows: 
    1 `end_effector`: np.array of shape `(3,)` representing the end-effector position. 
    2 `keypoints`: np.array of shape `(K, 3)` representing the keypoint positions. 
7. Inside of each function, only native Python functions and NumPy functions. 
8. For grasping stage, sub-goal constraint associates the end-effector with a keypoint.
9. For non-grasping stage, the end-effector position is never referenced. 
10. Output must contain numer of stages as python integer variable, and grasp keypoint indices and release keypoint indices as list of integers. 

If all constraints seem valid and sound, call the `Accept` tool. Otherwise, provide detailed feedback on what needs to be improved."""



def codegen_critic(state: SeecState, config):
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