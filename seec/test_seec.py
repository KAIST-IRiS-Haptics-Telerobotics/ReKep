# from modules.user_interface import launch_interface
import os

from langchain_core.messages import HumanMessage
from seec.services.state import GraphConfig
from seec.modules.agent import get_graph
import base64


graph_config = GraphConfig(
    high_level_actor="o4-mini",
    high_level_critic="o4-mini",
    low_level_actor="o4-mini",
    cypher_query="o4-mini",
    low_level_critic="o4-mini",
    codegen_actor="o4-mini",
    codegen_critic="o4-mini",
    thread_id="1",
)

config = {"configurable": graph_config}


def main():
    graph = get_graph()
    img_path = os.path.join(os.path.dirname(__file__), "..", "vlm_query", "pen", "query_img.png")
    # load image as bytes
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    image_data = base64.b64encode(img_bytes).decode("utf-8")

    from matplotlib import pyplot as plt
    plt.imshow(plt.imread(img_path))
    plt.show()

    # attach the image to the human message
    # attach the image to the human message with explicit types
    human_msg = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Put the pen in the cup",
            },
            {
                "type": "image",
                "source_type": "base64",
                "data": image_data,
                "mime_type": "image/jpeg",
            },
        ],
    }

    stream = graph.stream(
        {"messages": [human_msg]},
        config=config,
        stream_mode="updates",
    )
    for chunk in stream:
        print("=========================================================")
        print(chunk)
        print("=========================================================")

    final_state = {
        k: v for k, v in graph.get_state(config).values.items() if k != "messages"
    }
    print(final_state)


if __name__ == "__main__":
    main()
