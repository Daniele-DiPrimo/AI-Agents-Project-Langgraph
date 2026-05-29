from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from src.state import BlogState
from src.agents import classifier_node, reasoner_node, writer_node, exercises_writer_node
from src.tools import blog_tools
from langgraph.checkpoint.memory import MemorySaver

def route_after_reasoner(state: BlogState):
    """
    Router personalizzato che intercetta il tool 'done' per terminare il grafo.
    """
    last_message = state["messages"][-1]
    intent = state.get("intent", "").lower()
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            # Se l'agente ha deciso di chiamare 'done', usciamo dal loop
            if tool_call["name"] == "done":
                if intent == "esercizio":
                    return "exercises_writer"
                return "writer"
                
        # Se ha chiamato qualsiasi altro tool (es. tavily, arxiv), andiamo al nodo tools
        return "tools"
 
    # --- FALLBACK DI SICUREZZA ---
    # Se per caso l'LLM smette di chiamare tool e scrive testo libero
    # smistiamo comunque al nodo corretto per non far bloccare il grafo.
    if intent == "esercizio":
        return "exercises_writer"
    
    return "writer"

builder = StateGraph(BlogState)

builder.add_node("classifier", classifier_node)
builder.add_node("reasoner", reasoner_node)
builder.add_node("writer", writer_node)
builder.add_node("exercises_writer", exercises_writer_node)

# Al ToolNode passiamo TUTTI i tool TRANNE 'done', 
# perché 'done' non deve mai essere eseguito fisicamente
executable_tools = [t for t in blog_tools if t.name != "done"]
builder.add_node("tools", ToolNode(executable_tools))


builder.add_edge(START, "classifier")
builder.add_edge("classifier", "reasoner")

# 3. IL NODO CRITICO: Il Router Condizionale con mappatura esplicita
# Passiamo un dizionario come terzo argomento per aiutare LangGraph Studio
# a capire dove portano le stringhe restituite dalla funzione route_after_reasoner.
builder.add_conditional_edges(
    "reasoner", 
    route_after_reasoner,
    {
        "tools": "tools",  # Se la funzione restituisce "tools", vai al nodo "tools"
        "writer": "writer" ,
        "exercises_writer": "exercises_writer"          # Se restituisce END (es. tramite il tool 'done'), vai alla fine
    }
)

# 4. Dal tool si torna SEMPRE al ragionatore (chiusura del loop ReAct)
builder.add_edge("tools", "reasoner")
builder.add_edge("writer", END)
builder.add_edge("exercises_writer", END)


blog_system = builder.compile(
    interrupt_after=["writer", "exercises_writer"]  # Diciamo al grafo di fermarsi SUBITO DOPO aver scritto l'articolo
)