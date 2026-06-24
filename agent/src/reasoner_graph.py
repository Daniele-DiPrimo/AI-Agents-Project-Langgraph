from langgraph.graph import StateGraph, START, END
from src.state import ReasonerState
from src.agents_subgraph_reasoner import (
    reasoner_node,
    source_evaluator_node,
    completeness_evaluator_node,
    tool_executor_node
)

def route_after_completeness(state: ReasonerState):

    if(state.get("iterations") == 5): 
        return END

    if(state.get("is_complete", False)):
        return END
    
    return "reasoner"

reasoner_subgraph_builder = StateGraph(ReasonerState)

reasoner_subgraph_builder.add_node("reasoner", reasoner_node)
reasoner_subgraph_builder.add_node("source_evaluator", source_evaluator_node)
reasoner_subgraph_builder.add_node("completeness_evaluator", completeness_evaluator_node)
reasoner_subgraph_builder.add_node("tool_executor", tool_executor_node)

reasoner_subgraph_builder.add_edge(START, "reasoner")
reasoner_subgraph_builder.add_edge("reasoner", "tool_executor")

reasoner_subgraph_builder.add_edge("tool_executor", "source_evaluator")
    

reasoner_subgraph_builder.add_edge("source_evaluator", "completeness_evaluator")

reasoner_subgraph_builder.add_conditional_edges(
    "completeness_evaluator",
    route_after_completeness, {
        "reasoner": "reasoner",
        END: END
    }
)

reasoner_subgraph = reasoner_subgraph_builder.compile()