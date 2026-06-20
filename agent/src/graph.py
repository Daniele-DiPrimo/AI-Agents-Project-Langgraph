from langgraph.graph import StateGraph, START, END
from src.state import BlogState

from src.agents import classifier_node, writer_node, human_review_node, save_article_node
from src.reasoner_graph import reasoner_subgraph


builder = StateGraph(BlogState)

builder.add_node("classifier", classifier_node)
builder.add_node("writer", writer_node)
builder.add_node("human_review", human_review_node)
builder.add_node("save_article", save_article_node)
builder.add_node("reasoner_subgraph", reasoner_subgraph)


builder.add_edge(START, "classifier")
builder.add_edge("classifier", "reasoner_subgraph")
builder.add_edge("reasoner_subgraph", "writer")
builder.add_edge("writer", "human_review")
builder.add_edge("human_review", "save_article")
# human_review_node decide dinamicamente tramite Command() se andare a "writer", "save_article" o END
builder.add_edge("save_article", END) # Il salvataggio porta sempre alla fine del grafo


blog_system = builder.compile()