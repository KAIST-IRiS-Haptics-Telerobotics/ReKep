from ..services.llm_service import get_model
from ..services.state import SeecState
from .states import Accept

verifier_prompt = """You are critic in Actor-Critic paradigm. You are working on manipulation task. Your task is to verify soundness and correctness of a high-level robotic planning solution. High-level plan is provided via several stages, which are described with short and concise description.

Based on the planning description below, analyze whether:
1. Completing all stages in sequence would result in the completin of the task, defined by user request
2. There are no missing stages in the plan: after completing one stage, the next stage should be ready to execute
3. All robot grasps are represented via separate stage
4. Each stage is atomic and could be executed via low-level controller
5. Each stage should not contain complex movements, such as avoiding obstacles or moving around objects. If such movements are required, they should be represented via separate stage.

If the plan seems valid and sound, call the `Accept` tool. Otherwise, provide detailed feedback on what needs to be improved.

Focus on the logical consistency and feasibility of the planning approach, not implementation details."""



def high_level_critic(state: SeecState, config):
    messages = [
        {"role": "system", "content": verifier_prompt},
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