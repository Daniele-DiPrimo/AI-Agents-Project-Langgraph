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
    research_material: str
    graph_results: str
    final_article: str

class ReasonerState(TypedDict):
    """Stato del reasoner subgraph"""

    # --- Ereditati da BlogState (stessi nomi → LangGraph li copia automaticamente)
    intent: str
    subject: str
    specific_topic: str
    article_type: str
    prompt_to_reasoner: str

    # --- Piano strutturato (prodotto dal planner)
    tool_plan: list[str]           # ["tavily", "semantic_scholar"]
    raw_results: list[dict]        # output grezzo dei tool
    graph_results: str             # risultato della ricerca nel Knowledge Graph
    approved_sources: Annotated[list[dict], operator.add]   # fonti approvate dal source_evaluator
    research_material: str         # sintesi finale pulita → passa al BlogState
    visited_urls: Annotated[list[str], operator.add]

    # --- Flag di controllo flusso
    sources_evaluated: bool
    is_complete: bool
    missing_info: str
    iterations: int                # contatore cicli per evitare loop infiniti