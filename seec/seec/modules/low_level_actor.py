from ..services.llm_service import get_model
from ..services.state import SeecState, reasoning_models

prompt = """You are actor in Actor-Critic paradigm. You are working on manipulation task, controlling end-effector. Given scene description, task description, and sequence of stages required to execute the plan, your task is to generate low-level plan for executing it. 

The low-level plan is specified via constraint functions. There are two kinds of constraints: 
- Sub-goal constraints. Those constraints that must be satisfied **at the end of the stage**
- Path constraints. Those constraints must be satisfied **within the stage**.
Your task it **not to write code**, but to generate list of constraints and their description.

### Requirements
- For each stage, you may write 0 or more sub-goal constraints and 0 or more path constraints. 
- Avoid using path constraints when manipulating deformable objects (e.g., clothing, towels). 
- For grasping stage, you should write **only one** sub-goal constraint that associates the end-effector with a keypoint. No path constraints and any other subgoal constraints are needed. 

### Input
- The user has requested a task.
- The image of the environment, overlayed with keypoints marked with their indices
- The list of stages, each stage is represented via short description, less than 10 words.

### Scene Description
{scene_description_dict}

### User Request
"{user_request}"

### Stages
"{stages}"

### Output (Example Format)
- "grasp teapot" stage: 
    - sub-goal constraints: "align the end-effector with the teapot handle" 
    - path constraints: None 
- "align teapot with cup opening" stage: 
    - sub-goal constraints: "the teapot spout needs to be 10cm above the cup opening" 
    - path constraints: "robot is grasping the teapot", "the teapot must stay upright to avoid spilling" 
- "pour liquid" stage: 
    - sub-goal constraints: "the teapot spout needs to be 5cm above the cup opening", "the teapot spout must be tilted to pour liquid" 
    - path constraints: "the teapot spout is directly above the cup opening"

### Your Turn:
"""


def low_level_actor(state: SeecState, config):
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
