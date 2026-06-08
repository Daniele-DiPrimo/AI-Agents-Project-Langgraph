from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import BaseMessage, add_messages

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

class ReasonerState(TypedDict):
    """Stato del reasoner subgraph"""

    # --- Ereditati da BlogState (stessi nomi → LangGraph li copia automaticamente)
    intent: str
    macro_domain: str
    specific_topic: str
    article_type: str

    # --- Conversazione con i tool (solo per il loop ReAct interno)
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Piano strutturato (prodotto dal planner)
    tool_plan: list[str]           # ["tavily", "semantic_scholar"]
    search_queries: dict[str, str] # {"tavily": "...", "semantic_scholar": "..."}
    current_step: int              # indice corrente nel tool_plan

    # --- Dati reali che circolano tra i nodi
    raw_results: list[dict]        # output grezzo dei tool
    approved_sources: list[dict]   # fonti approvate dal source_evaluator
    research_material: str         # sintesi finale pulita → passa al BlogState

    # --- Flag di controllo flusso
    sources_evaluated: bool
    is_complete: bool
    iterations: int                # contatore cicli per evitare loop infiniti