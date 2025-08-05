from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph


def get_model(config, default, key):
    model = config["configurable"].get(key, default)
    if model == "4o":
        return ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model == "4o-mini":
        return ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    elif model == "o1":
        return ChatOpenAI(temperature=1, model_name="o1-preview")
    elif model == "o1-mini":
        return ChatOpenAI(temperature=1, model_name="o1-mini")
    elif model == "o4-mini":
        return ChatOpenAI(temperature=1, model_name="o4-mini")
    elif model == "o3":
        return ChatOpenAI(temperature=1, model_name="o3")
    else:
        raise ValueError
