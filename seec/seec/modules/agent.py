from typing import Literal

from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from .high_level_actor import high_level_actor
from .high_level_critic import high_level_critic
from .low_level_actor import low_level_actor
from .low_level_critic import low_level_critic
from .codegen_actor import codegen_actor
from .codegen_static_check import codegen_static_check
from .codegen_cricit import codegen_critic
from ..services.state import SeecState, OutputState, GraphConfig

"""
The overall architecture of the SEEC planner is presented below:

  ------------------------------  High-level planning  -----------------------------
  |                                                                                |
  |   +--------------------+    decline    +--------------------+                  |
  |   |  High-level plan   |<-------------\|  High-level critic |                  |
  |   +--------------------+-------------->|                    |                  |
  |            |  no req.                  +--------------------+                  |
  |            v                                     |                             | 
  |           (X)                                    |                             |
  ---------------------------------------------------|------------------------------
                                                     |
                            accept                   /
                /-----------------------------------/
                |
  --------------|----------------  Low-level planning  -----------------------------
  |             |                                                                  |                                 
  |             ⌄                                                                  |
  |   +--------------------+    decline    +--------------------+                  |
  |   | Subgoals &         |<-------------\|  Low-level critic  |                  |
  |   | patterns           |-------------->|                    |                  |
  |   +--------------------+               +--------------------+                  |
  |                                                  |                             |
  ---------------------------------------------------|------------------------------
                                                     |
                            accept                   /
             /--------------------------------------/
             |
  -----------|------------------------  Write Code  --------------------------------
  |          ⌄                                                                     |
  |   +-------------+    decline    +--------------+                               |
  |   |  Code Gen   |<------------\ |  Static crit |---- accept ----\              |
  |   +-------------+-------------> +--------------+                 \             |
  |         ^                                                         ⌄            |
  |         | decline                                            +-----------+     |
  |         \----------------------------------------------------| Code Crit |     |
  |                                                              +-----------+     |
  |                                                                   |            |
  |                                                                   v            |
  |                                                                  (X)           |
  ----------------------------------------------------------------------------------
"""


def static_check_cond(state: SeecState) -> Literal["codegen_critic", "codegen_actor"]:
    if isinstance(state["messages"][-1], AIMessage):
        return "codegen_critic"
    else:
        return "codegen_actor"


def cricit_cond(state: SeecState, on_success: str, on_fail: str) -> str:
    if state.get("plan_verified", False):
        return on_success
    else:
        return on_fail


def get_graph() -> StateGraph:
    # Define a new graph
    memory = MemorySaver()
    workflow = StateGraph(SeecState, input=MessagesState, output=OutputState, config_schema=GraphConfig)

    # Add nodes
    workflow.add_node(high_level_actor)
    workflow.add_node(high_level_critic)
    workflow.add_node(low_level_actor)
    workflow.add_node(low_level_critic)
    workflow.add_node(codegen_actor)
    workflow.add_node(codegen_static_check)
    workflow.add_node(codegen_critic)

    # Set entry point
    workflow.set_entry_point("high_level_actor")

    # Main planning flow
    # --- High-level planning ---
    workflow.add_edge("high_level_actor", "high_level_critic")
    workflow.add_conditional_edges(
        "high_level_critic",
        lambda state: cricit_cond(state, "low_level_actor", "high_level_actor"),
    )
    # --- Low-level planning ---
    workflow.add_edge("low_level_actor", "low_level_critic")
    workflow.add_conditional_edges(
        "low_level_critic",
        lambda state: cricit_cond(state, "codegen_actor", "low_level_actor"),
    )
    # --- Code generation ---
    workflow.add_edge("codegen_actor", "codegen_static_check")
    workflow.add_conditional_edges("codegen_static_check", static_check_cond)
    workflow.add_conditional_edges(
        "codegen_critic",
        lambda state: cricit_cond(state, END, "codegen_actor"),
    )
    graph = workflow.compile(checkpointer=memory)
    
    return graph
