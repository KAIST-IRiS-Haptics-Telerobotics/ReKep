from langgraph.graph import MessagesState
from typing import TypedDict, Literal
from typing import Final, Tuple


class SeecState(MessagesState):
    requirements: str
    ai_requirements: str
    code: str
    accepted: bool
    plan_verified: bool
    safety_messages: list
    safety_accepted: bool


class OutputState(TypedDict):
    code: str


class GraphConfig(TypedDict):
    high_level_actor: Literal["4o", "4o-mini", "o1", "o1-mini", "o4-mini", "o3"]
    high_level_critic: Literal["4o", "4o-mini", "o1", "o1-mini", "o4-mini", "o3"]
    low_level_actor: Literal["4o", "4o-mini", "o1", "o1-mini", "o4-mini", "o3"]
    cypher_query: Literal["4o", "4o-mini", "o1", "o1-mini", "o4-mini", "o3"]
    low_level_critic: Literal["4o", "4o-mini", "o1", "o1-mini", "o4-mini", "o3"]
    codegen_actor: Literal["4o", "4o-mini", "o1", "o1-mini", "o4-mini", "o3"]
    codegen_critic: Literal["4o", "4o-mini", "o1", "o1-mini", "o4-mini", "o3"]
    thread_id: str


reasoning_models: tuple[str, ...] = ("o1-mini", "o1", "o4-mini", "o3")
