from langgraph.graph import StateGraph, START, END
from src.state import ReasonerState
from src.agents import (
    planner_node,
    source_evaluator_node,
    completeness_evaluator_node
)
from src.tools import blog_tools
from langgraph.prebuilt import ToolNode

def route_after_planner(state: ReasonerState):
    """Router per il sottografo reasoner"""

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools_ex"

    return "completeness_evaluator"

def route_after_completeness(state: ReasonerState):

    if(state.get("iterations") == 3): 
        return END

    if(state.get("is_complete", False)):
        return END
    return "planner"

reasoner_subgraph_builder = StateGraph(ReasonerState)

executable_tools = [t for t in blog_tools if t.name != "done"]

reasoner_subgraph_builder.add_node("planner", planner_node)
reasoner_subgraph_builder.add_node("source_evaluator", source_evaluator_node)
reasoner_subgraph_builder.add_node("completeness_evaluator", completeness_evaluator_node)
reasoner_subgraph_builder.add_node("tools_ex", ToolNode(executable_tools))

reasoner_subgraph_builder.add_edge(START, "planner")
reasoner_subgraph_builder.add_conditional_edges(
    "planner", 
    route_after_planner,{
        "tools_ex": "tools_ex",
        "completeness_evaluator": "completeness_evaluator"
    }
)

reasoner_subgraph_builder.add_edge("tools_ex", "source_evaluator")
reasoner_subgraph_builder.add_edge("source_evaluator", "completeness_evaluator")
reasoner_subgraph_builder.add_conditional_edges(
    "completeness_evaluator",
    route_after_completeness, {
        "planner": "planner",
        END: END
    }
)

reasoner_subgraph = reasoner_subgraph_builder.compile()