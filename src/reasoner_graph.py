from langgraph.graph import StateGraph, START, END
from src.state import ReasonerState
from src.agents_subgraph_reasoner import (
    planner_node,
    source_evaluator_node,
    completeness_evaluator_node,
    tool_executor_node
)
from src.tools import blog_tools


def route_after_completeness(state: ReasonerState):

    if(state.get("iterations") == 3): 
        return END

    if(state.get("is_complete", False)):
        return END
    return "planner"

reasoner_subgraph_builder = StateGraph(ReasonerState)

executable_tools = [t for t in blog_tools]

reasoner_subgraph_builder.add_node("planner", planner_node)
reasoner_subgraph_builder.add_node("source_evaluator", source_evaluator_node)
reasoner_subgraph_builder.add_node("completeness_evaluator", completeness_evaluator_node)
reasoner_subgraph_builder.add_node("tool_executor", tool_executor_node)

reasoner_subgraph_builder.add_edge(START, "planner")
reasoner_subgraph_builder.add_edge("planner", "tool_executor")
reasoner_subgraph_builder.add_edge("tool_executor", "source_evaluator")
reasoner_subgraph_builder.add_edge("source_evaluator", "completeness_evaluator")
reasoner_subgraph_builder.add_conditional_edges(
    "completeness_evaluator",
    route_after_completeness, {
        "planner": "planner",
        END: END
    }
)

reasoner_subgraph = reasoner_subgraph_builder.compile()