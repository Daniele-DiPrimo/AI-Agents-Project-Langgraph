from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
import operator

class BlogState(TypedDict):
    """
    Rappresenta lo stato globale che viene passato tra i nodi del grafo.
    """
    # Lista dei messaggi (essenziale per il loop ReAct e per i Tool)
    messages: Annotated[list, add_messages]
    original_prompt: str
    intent: str
    subject: str
    specific_topic: str
    prompt_to_reasoner: str
    suggestions: list[dict]
    plan_justification: str
    current_suggestion_index: int
    research_material: str
    final_article: str
    graph_results: dict

class ReasonerState(TypedDict):
    """Stato del reasoner subgraph"""

    # --- Ereditati da BlogState (stessi nomi → LangGraph li copia automaticamente)
    intent: str
    subject: str
    specific_topic: str
    article_type: str
    prompt_to_reasoner: str

    # --- Piano strutturato (prodotto dal planner)
    tool_plan: list[dict]      
    raw_results: dict
    graph_results: dict
    approved_sources: Annotated[list[dict], operator.add]
    not_approved_sources: Annotated[list[dict], operator.add]
    research_material: str

    # --- Flag di controllo flusso
    sources_evaluated: bool
    is_complete: bool
    missing_info: str
    iterations: int