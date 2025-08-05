from ..services.llm_service import get_model
from ..services.state import SeecState, reasoning_models

prompt = """You are actor in Actor-Critic paradigm. You are working on manipulation task, controlling end-effector. Given scene description, user request, a valid plan consisting of stages, and list of constraints to complete each stage, your task is to write a code that implements low-level constraints in practice. 

High-level plan is provided via several stages, which are described with short and concise description. Assume it is valid, and it's execution would result in the completion of the task, defined by user request.

The low-level plan is specified via constraint functions in Python. There are two kinds of constraints: 
- Sub-goal constraints. Those constraints that must be satisfied **at the end of the stage**
- Path constraints. Those constraints must be satisfied **within the stage**.
You are given a correctly formatted constraints, which you must implement in Python code precisely.

### Requirements
- Output must not containt any import statements.
- Each constraint takes a dummy end-effector point and a set of keypoints as input and returns a numerical cost, where the constraint is satisfied if the cost is smaller than or equal to zero. 
- Avoid using "if" statements in your constraints. 
- Inputs to the constraints are as follows: 
    - `end_effector`: np.array of shape `(3,)` representing the end-effector position. 
    - `keypoints`: np.array of shape `(K, 3)` representing the keypoint positions. 
- For any path constraint that requires the robot to be still grasping a keypoint `i`, you may use the provided function `get_grasping_cost_by_keypoint_idx` by calling `return get_grasping_cost_by_keypoint_idx(i)` where `i` is the index of the keypoint. 
- Inside of each function, you may use native Python functions and NumPy functions, and the provided `get_grasping_cost_by_keypoint_idx` function. 
- Grasping stage sub-goal constraint should accosiate the end-effector with a keypoint.
- Non-grasping stages should not refer to the end-effector position. 
- In order to move a keypoint, its associated object must be grasped in one of the previous stages. 
- You may use two keypoints to form a vector, which can be used to specify a rotation (by specifying the angle between the vector and a fixed axis). 
- You may use multiple keypoints to specify a surface or volume. 
- You may also use the center of multiple keypoints to specify a position. 

### Input
- The user has requested a task.
- The image of the environment, overlayed with keypoints marked with their indices
- The list of stages, each stage is represented via short description, less than 10 words.
- The list of sub-goal constraints and path constraints for each stage.

### Scene Description
{scene_description_dict}

### User Request
"{user_request}"

### Stages
"{stages}"

### Constraints
"{constraints}"

### Output (Example Format)
```python  
# Your explanation of how many stages are involved in the task and what each stage is about. 
# ...  

num_stages = ?  

### stage 1 sub-goal constraints (if any) 
def stage1_subgoal_constraint1(end_effector, keypoints): 
    "Put your explanation here." 
    ... 
    return cost 

# Other sub-goal constraints...

### stage 1 path constraints (if any) 
def stage1_path_constraint1(end_effector, keypoints): 
    "Put your explanation here." 
    ... 
    return cost 

# Other path constraints...    


# Summarize keypoints to be grasped in all grasping stages.
# The length of the list should be equal to the number of stages.
# For grapsing stage, write the keypoint index. For non-grasping stage, write -1.
grasp_keypoints = [?, ..., ?]

# Summarize at **the end of which stage** the robot should release the keypoints.
# The keypoint indices must appear in an earlier stage as defined in `grasp_keypoints` (i.e., a keypoint can only be released only if it has been grasped previously).
# Only release object when it's necessary to complete the task, e.g., drop bouquet in the vase.
# The length of the list should be equal to the number of stages.
# If a keypoint is to be released at the end of a stage, write the keypoint index at the corresponding location. Otherwise, write -1.
release_keypoints = [?, ..., ?]
```

### Your Turn:
"""


def codegen_actor(state: SeecState, config):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "user", "content": state.get("requirements")},
    ] + state["messages"]
    model = get_model(config, "o4-mini", "reasoning_model")

    # messages = [
    #     {"role": "system", "content": prompt},
    #     {"role": "user", "content": state.get("requirements")},
    # ] + state["messages"]
    # model = get_model(config, "4o", "reasoning_model")

    response = model.invoke(messages)
    planning_reasoning = {}
    planning_reasoning["plan"] = response.content
    return {
        "ai_requirements": response.content,
        "messages": [response],
    }
