from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import NegotiationState
from nodes import ai_node, setup_node, mediator_node, evaluation_node
from langgraph.prebuilt import ToolNode, tools_condition
from tools.rag_tools import policy_search_tool


def create_graph():
    def should_continue(state: NegotiationState):
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        if "합의" in last_message or "종료" in last_message or len(state["messages"]) > 30:
            return "evaluate"
        return "human_input"

    workflow = StateGraph(NegotiationState)

    workflow.add_node("ai_agent", ai_node)
    workflow.add_node("setup_node", setup_node)
    workflow.add_node("mediator", mediator_node)
    workflow.add_node("evaluate", evaluation_node)
    workflow.add_node("human_input", lambda state: state)
    workflow.add_node("tools", ToolNode([policy_search_tool]))

    workflow.set_entry_point("setup_node")
    workflow.add_edge("setup_node", "ai_agent")

    workflow.add_conditional_edges(
        "ai_agent",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "mediator"
        }
    )
    workflow.add_edge("tools", "ai_agent")
    workflow.add_conditional_edges(
        "mediator",
        should_continue,
        {
            "evaluate": "evaluate",      # 협상 종료 시 평가 노드로
            "human_input": "human_input"  # 계속 진행 시 사람 입력 대기로
        }
    )
    workflow.add_edge("human_input", "ai_agent")
    workflow.add_edge("evaluate", END)

    return workflow.compile(checkpointer=MemorySaver(), interrupt_before=["human_input"])