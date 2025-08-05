from ..services.llm_service import get_model
from ..services.state import SeecState, reasoning_models

prompt = """You are actor in Actor-Critic paradigm. You are working on manipulation task, controlling end-effector. Your task it to generate high-level task plan in a robotic framework. Your task is to interpret a user request and input image into high-level stages which are required to complete the task.  

### Input
- The user has requested a task.
- The image of the environment, overlayed with keypoints marked with their indices

### Requirements
- Determine how many stages are involved in the task. 
- Each stage is represented via short description, less than 10 words.
- Grasping must be an independent stage.
- A single folding action should consist of two stages: one grasp and one place.
- The robot can only grasp one object at a time. 
- During one stage, robot mostly moves in a straight line, without complex movements.
- During one stage, robot could not avoid collisions, so if robot needs to move around an object, it should be done in a separate stage.

### Clarification
If unsure, ask for clarification or report limitations.

### User Request
"{user_request}"

### Output (Example Format)
3 stages: "grasp teapot", "align teapot with cup opening", "pour liquid"

### Your Turn:
"""

def high_level_actor(state: SeecState, config):
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
        # "print_stream": planning_reasoning,
        "ai_requirements": response.content,
        "messages": [response],
    }