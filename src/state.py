from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class BlogState(TypedDict):
    """
    Rappresenta lo stato globale che viene passato tra i nodi del grafo.
    """
    # Lista dei messaggi (essenziale per il loop ReAct e per i Tool)
    messages: Annotated[list, add_messages]
    original_prompt: str
    intent: str
    macro_domain: str
    specific_topic: str