from langgraph.graph import StateGraph, START, END
from src.state import ReasonerState
from src.agents_subgraph_reasoner import (
    reasoner_node,
    source_evaluator_node,
    completeness_evaluator_node,
    tool_executor_node
)
from src.tools import blog_tools

def route_after_tool(state: ReasonerState):
    """
    Controlla quale tool è stato appena eseguito dal tool_executor.
    Se è il K-RAG locale, torna direttamente al reasoner.
    Se è un tool di ricerca Web, passa la mano agli elementi di valutazione.
    """
    tool_plan = state.get("tool_plan", [])

    if tool_plan:
        last_tool_call = tool_plan[-1]
        
        # Verifichiamo se l'ultimo messaggio proviene dal tool K-RAG unificato
        # Nota: 'ricerca_krag_unificata' deve corrispondere al @tool della tua libreria
        if last_tool_call.get("name") == "ricerca_krag_unificata":
            return "reasoner"
            
    # Di default, se viene usato Tavily/Google Search o altri tool esterni
    return "source_evaluator"

def route_after_completeness(state: ReasonerState):

    if(state.get("iterations") == 5): 
        return END

    if(state.get("is_complete", False)):
        return END
    
    return "reasoner"

reasoner_subgraph_builder = StateGraph(ReasonerState)

executable_tools = [t for t in blog_tools]

reasoner_subgraph_builder.add_node("reasoner", reasoner_node)
reasoner_subgraph_builder.add_node("source_evaluator", source_evaluator_node)
reasoner_subgraph_builder.add_node("completeness_evaluator", completeness_evaluator_node)
reasoner_subgraph_builder.add_node("tool_executor", tool_executor_node)

reasoner_subgraph_builder.add_edge(START, "reasoner")
reasoner_subgraph_builder.add_edge("reasoner", "tool_executor")

reasoner_subgraph_builder.add_conditional_edges(
    "tool_executor",
    route_after_tool, 
    {
        "reasoner": "reasoner",
        "source_evaluator": "source_evaluator"
    }
)

reasoner_subgraph_builder.add_edge("source_evaluator", "completeness_evaluator")

reasoner_subgraph_builder.add_conditional_edges(
    "completeness_evaluator",
    route_after_completeness, {
        "reasoner": "reasoner",
        END: END
    }
)

reasoner_subgraph = reasoner_subgraph_builder.compile()