from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from tools.rag_tools import policy_search_tool

from core.state import NegotiationState
from core.nodes import (
    setup_node, 
    negotiator_node, 
    evaluation_node, 
    reflection_node,
    logging_node
)
def route_to_setup_or_negotiator(state:NegotiationState):
    """
    START에서 실행될 라우터 로직
    """
    messages = state.get("messages", [])
    is_finished = state.get("is_finished", False)

    if not messages or is_finished:
        return "setup"
    
    return "negotiator"

def route_after_negotiation(state: NegotiationState):
    """협상 후 분기 로직 (공통)"""
    last_msg = state["messages"][-1]
    
    # 툴 호출 확인
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    
    # 종료 조건 확인 (턴 수 or is_finished 플래그)
    MAX_TURNS = 10
    current_turns = len([m for m in state["messages"] if m.type == "human"])
    
    if state.get("is_finished") or current_turns >= MAX_TURNS:
        return "evaluator"
    
    return END # 대화 종료 (사용자 입력을 기다림)

def route_after_evaluation(state: NegotiationState):
    """평가 후 분기 로직 (Reflexion 전용)"""
    current_reflections = state.get("reflections", [])
    max_retries = state.get("max_retries", 3) 
    
    if len(current_reflections) >= max_retries:
        return "logger"
        
    return "reflector"

def build_graph(mode: str, use_checkpoint: bool = True):
    """모드에 따라 적절한 그래프를 생성하여 반환하는 통합 팩토리 함수.

    현재 UI에서 사용하는 모드 값:
    - "baseline":      BASELINE 프롬프트 사용
    - "cot_previous":  COT_PREVIOUS 프롬프트 사용
    - "cot_upgrade":   COT_NEGOTIATOR (개선 버전) 프롬프트 사용

    이들 모드는 모두 단일 패스 그래프(Reflexion 없음)를 사용합니다.
    구(舊) "Reflexion" 모드만 반성 루프를 포함한 그래프를 유지합니다.
    """
    workflow = StateGraph(NegotiationState)
    tools = [policy_search_tool]

    workflow.add_node("setup", setup_node)
    workflow.add_node("negotiator", negotiator_node)
    workflow.add_node("evaluator", evaluation_node) # 학습용 보상 계산
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("logger", logging_node) # 최종 CSV 저장

    # Start -> Setup -> Negotiator
    workflow.add_conditional_edges(
        START,
        route_to_setup_or_negotiator,
        {
            "setup": "setup",
            "negotiator": "negotiator"
        }
    )
    # [Setup] -> [Negotiator]
    workflow.add_edge("setup", "negotiator")
    
    # Tools <-> Negotiator
    workflow.add_edge("tools", "negotiator")

    # --- 모드별 분기 ---
    # 새 모드(baseline, cot_previous, cot_upgrade)는 모두 단일 패스 사용
    if mode in ("baseline", "cot_previous", "cot_upgrade") or "CoT" in mode or "Baseline" in mode:
        # [CoT 모드]: Negotiator -> Evaluator -> Logger -> END
        
        workflow.add_conditional_edges(
            "negotiator",
            route_after_negotiation,
            {
                "tools": "tools",
                "evaluator": "logger", # 대화가 끝나면 평가
                END: END                # 사용자 입력 대기
            }
        )
        
        # Logger -> END 
        workflow.add_edge("logger", END)

    else:
        workflow.add_node("reflector", reflection_node)

        # Negotiator -> (Tools / Evaluator / END)
        workflow.add_conditional_edges(
            "negotiator",
            route_after_negotiation,
            {
                "tools": "tools",
                "evaluator": "evaluator",
                END: END
            }
        )

        # Evaluator -> (Reflector / Logger)
        workflow.add_conditional_edges(
            "evaluator",
            route_after_evaluation,
            {
                "reflector": "reflector", # 실패: 반성하러 이동
                "logger": "logger"        # 성공: 저장하고 종료
            }
        )
        
        # Reflector -> Setup (재시도)
        # *중요: 반성 내용을 안고 Setup으로 가서 다시 시작 (Setup에서 reflections는 유지해야 함)
        workflow.add_edge("reflector", "setup")
        
        # Logger -> END
        workflow.add_edge("logger", END)
    
    if use_checkpoint:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    else:
        return workflow.compile()

# for langgraph-cli testing
graph = build_graph("baseline", use_checkpoint=False)