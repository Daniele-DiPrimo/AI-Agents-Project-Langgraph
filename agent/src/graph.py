from langgraph.graph import StateGraph, START, END

from src.state import BlogState
from src.agents import classifier_node, planner_node, hitl_planner_node, writer_node, human_review_node, save_article_node, information_gathering_node
from src.reasoner_graph import reasoner_subgraph

def route_after_classifier(state: BlogState):
    """Legge l'intent dallo stato e restituisce il nome del nodo successivo."""
    intent = state.get("intent")
    
    if intent == "Suggerimento":
        return "planner"
    elif intent == "ArticoloTeorico" or intent == "TechNews" or intent == "Eserciziario":
        return "information_gathering"
    else:
        # Fallback nel caso in cui l'intent non sia riconosciuto
        print("Error: Classifier didn't recognize the intent.")
        return END

builder = StateGraph(BlogState)

builder.add_node("classifier", classifier_node)
builder.add_node("planner", planner_node)
builder.add_node("hitl_planner", hitl_planner_node)
builder.add_node("writer", writer_node)
builder.add_node("human_review", human_review_node)
builder.add_node("save_article", save_article_node)
builder.add_node("reasoner_subgraph", reasoner_subgraph)
builder.add_node("information_gathering", information_gathering_node)


builder.add_edge(START, "classifier")

builder.add_conditional_edges(
    "classifier",
    route_after_classifier,
    {      
        "planner": "planner",        
        "information_gathering": "information_gathering",
        END: END
    }
)

builder.add_edge("planner", "hitl_planner")
#builder.add_edge("hitl_planner", "information_gathering") 
builder.add_edge("information_gathering", "reasoner_subgraph")
builder.add_edge("reasoner_subgraph", "writer")
builder.add_edge("writer", "human_review")
builder.add_edge("human_review", "save_article") # human_review_node decide dinamicamente tramite Command() se andare a "writer", "save_article" o END
builder.add_edge("save_article", END) #save article decide dinamicamente tramite Command() se andare a "hitl_planner" o END


blog_system = builder.compile()