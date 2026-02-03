from core.state import NegotiationState
from langchain_core.messages import AIMessage
from langchain_core.messages import RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from tools.rag_tools import policy_search_tool
import uuid
from datetime import datetime
import os
import copy
import pandas as pd
from core.prompts import (
    COT_NEGOTIATOR_SYSTEM,
    COT_NEGOTIATOR_HUMAN, 
    EVALUATOR_SYSTEM,
    EVALUATOR_HUMAN,
    REFELXION_NEGOTIATOR_SYSTEM,
    REFELXION_NEGOTIATOR_HUMAN,
    REFELXION_REFLECTION_SYSTEM,
    REFELXION_REFLECTION_HUMAN
)
from core.scenarios import (
    PRIORITIES,
    SCENARIOS
)
from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)

PROMPT_REGISTRY = {
    "reflexion": {
        "system": REFELXION_NEGOTIATOR_SYSTEM,
        "human": REFELXION_NEGOTIATOR_HUMAN
    },
    "cot": {
        "system": COT_NEGOTIATOR_SYSTEM,
        "human": COT_NEGOTIATOR_HUMAN
    }
}

def setup_node(state: NegotiationState):
    u_role = state.get("user_role", "구매자") 
    a_role = "판매자" if u_role == "구매자" else "구매자"
    model = state.get("model", "gpt-4o")
    mode = state.get("mode", "reflexion")

    max_retries = state.get("max_retries", 3)
    past_reflections = state.get("reflections", [])

    delete_messages = [RemoveMessage(id=m.id) for m in state.get("messages", [])]

    user_goals = copy.deepcopy(PRIORITIES.get(u_role, {}))
    if state.get("user_priority_inputs"):
        user_goals = state["user_priority_inputs"]
    user_priority_str = "\n".join([f"- {k} ({v}점)" for k, v in user_goals.items()])
    
    ai_goals = copy.deepcopy(PRIORITIES.get(a_role, {}))
    if state.get("ai_priority_inputs"):
        ai_goals = state["ai_priority_inputs"]
    ai_priority_str = "\n".join([f"- {k} ({v}점)" for k, v in ai_goals.items()])


    # 초기 State 구성
    initial_state = {
        "messages": delete_messages, 
        "user_role": u_role, 
        "ai_role": a_role, 
        "model": model,
        "is_finished": False,
        "mode": mode,

        "user_scenario": SCENARIOS[u_role],
        "ai_scenario": SCENARIOS[a_role],
        "user_priority": user_priority_str,
        "ai_priority": ai_priority_str,
        "user_goals": user_goals,
        "ai_goals": ai_goals,
        
        "summary": "",  # 요약 필드 초기화
        
        "buyer_score": 0.0,
        "seller_score": 0.0,
        "buyer_reward": 0.0,
        "seller_reward": 0.0,
        "final_result": "",
        
        "reflections": past_reflections, 
        "max_retries": max_retries
    }
    return initial_state

def negotiator_node(state: NegotiationState):
    mode = state.get("prompt_mode", "reflexion")
    templates = PROMPT_REGISTRY.get(mode, PROMPT_REGISTRY["reflexion"])

    tools = [policy_search_tool]
    llm = _create_llm(state, temperature=0.9).bind_tools(tools)
    
    recent_msgs = state["messages"][-4:] if state["messages"] else []
    recent_summary = "\n".join([f"{type(m).__name__}: {m.content}" for m in recent_msgs])

    weighted_priority_context = _get_weighted_priority(state)
    
    reflections_str = _parse_reflections(state.get("reflections", []))

    last_message = (
        state["messages"][-1].content 
        if state["messages"] 
        else f"이제 협상을 시작합니다. 당신은 {state['ai_role']}로서 상대방에게 첫 마디를 건네세요."
    )

    system_message = SystemMessagePromptTemplate.from_template(templates["system"])
    human_message = HumanMessagePromptTemplate.from_template(templates["human"])
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    chain = prompt | llm

    response = chain.invoke({
        "role": state["ai_role"],
        "opponent": state["user_role"],
        "scenario": state["ai_scenario"],
        "priority": weighted_priority_context,
        "recent_summary": recent_summary,
        "reflections": reflections_str,
        "last_message": last_message
    })

    if response.tool_calls:
        return {"messages": [response]}

    content = response.content
    if "최종 발화:" in content:
        clean_response = content.split("최종 발화:")[-1].strip()
    else:
        clean_response = content.strip()

    return {"messages": [AIMessage(content=clean_response)]}

def reflection_node(state: NegotiationState):
    llm = _create_llm(state, temperature=0.5)

    weighted_priority = _get_weighted_priority(state, include_instruction=False)

    trajectory = "\n".join([f"[{m.type}] {m.content}" for m in state["messages"]])
    reflections = "\n".join(state.get("reflections", []))

    if state["ai_role"] == "구매자":
        my_reward = state.get("buyer_reward", 0.0)
    else:
        my_reward = state.get("seller_reward", 0.0)
    
    scores = f"이번 협상에서 획득한 나의 점수: {my_reward}점"

    system_message = SystemMessagePromptTemplate.from_template(
        template=REFELXION_REFLECTION_SYSTEM
    )

    human_message = HumanMessagePromptTemplate.from_template(
        template=REFELXION_REFLECTION_HUMAN,
        input_variables=["role", "scenario", "priority", "scores", "past_reflections", "trajectory"]
    )
    
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    chain = prompt | llm | StrOutputParser()

    content = chain.invoke({
        "role": state["ai_role"],
        "scenario": state["ai_scenario"],
        "priority": weighted_priority,
        "scores": scores,
        "past_reflections": reflections,
        "trajectory": trajectory
    })

    if "최종 결과" in content:
        result_text = content.split("최종 결과:")[-1].strip()
    else:
        result_text = content.strip()

    return {"reflections": [result_text]}

def logging_node(state: NegotiationState):
    """
    협상 종료 후, 연구 분석용 데이터를 생성하고 저장하는 노드
    """
    
    llm = _create_llm(state, temperature=0.5)
    system_message = SystemMessagePromptTemplate.from_template(
            template=EVALUATOR_SYSTEM
        )

    human_message = HumanMessagePromptTemplate.from_template(
        template=EVALUATOR_HUMAN
    )
    
    dialogue = "\n".join([f"[{m.type}] {m.content}" for m in state["messages"]])

    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    chain = prompt | llm | StrOutputParser()
    
    result_text = chain.invoke({
        "dialogue": dialogue,
    }).split("최종 결과:")[-1].strip()

    buyer_points, seller_points = _calculate_points(state, result_text)

    _save_result_to_csv(state, dialogue, buyer_points, seller_points)

    # 7. 최종 상태 업데이트
    return {
        "final_result": result_text,
        "buyer_score": buyer_points,
        "seller_score": seller_points,
        "is_finished": True
    }

def evaluation_node(state: NegotiationState):
    """
    Refexion 에이전트 학습용 보상 계산 노드
    """
    
    llm = _create_llm(state, temperature=0.5)
    system_message = SystemMessagePromptTemplate.from_template(
            template=EVALUATOR_SYSTEM
        )

    human_message = HumanMessagePromptTemplate.from_template(
        template=EVALUATOR_HUMAN
    )
    
    dialogue = "\n".join([f"[{m.type}] {m.content}" for m in state["messages"]])

    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    chain = prompt | llm | StrOutputParser()
    
    result_text = chain.invoke({
        "dialogue": dialogue,
    }).split("최종 결과:")[-1].strip()

    buyer_reward, seller_reward = _calculate_rewards(state, result_text)

    return {
        "buyer_reward": buyer_reward,
        "seller_reward": seller_reward
    }

def _calculate_points(state, result_text):
    """
    [OTC 정의]
    refund otc : 1 (full)/ .5 (half) / 0 (none)
    reviews otc: 1 (retract) / 0 (keep)
    apology otc: 1 (didn't) / 0 (did)

    [점수 공식]
    buyer points = (Refund_otc * Rank_1) + (1-MyReview_otc ) * Rank_2 + (OtherReview_otc * Rank_3) + (Apology_otc * Rank_4)
    seller_points = (1 - Refund_otc) * Rank_1 + (1 - MyReview_otc) * Rank_2 + (OtherReview_otc * Rank_3) + (Apology_otc * Rank_4) 
    """
    buyer_goals = state["user_goals"] if state["user_role"] == "구매자" else state["ai_goals"]
    seller_goals = state["ai_goals"] if state["user_role"] == "구매자" else state["user_goals"]

    b_rank1 = buyer_goals.get("환불 받기", 0)
    b_rank2 = buyer_goals.get("판매자에 대한 부정적인 리뷰 유지하기", 0)
    b_rank3 = buyer_goals.get("판매자가 나에 대한 부정적인 리뷰 철회하기", 0)
    b_rank4 = buyer_goals.get("상대로부터 공식적인 사과받기", 0)

    s_rank1 = seller_goals.get("환불 방어", 0)
    s_rank2 = seller_goals.get("구매자에 대한 부정적인 리뷰 유지하기", 0)
    s_rank3 = seller_goals.get("구매자가 나에 대한 부정적인 리뷰 철회하기", 0)
    s_rank4 = seller_goals.get("상대로부터 공식적인 사과받기", 0)

    refund_otc = 0.0
    if "환불: 전체" in result_text:
        refund_otc = 1.0
    elif "환불: 부분" in result_text:
        refund_otc = 0.5
    else:
        refund_otc = 0.0

    buyer_review_otc = 1.0 if "구매자 리뷰: 철회" in result_text else 0.0
    seller_review_otc = 1.0 if "판매자 리뷰: 철회" in result_text else 0.0

    seller_apology_otc = 0.0 if "판매자 사과: 있음" in result_text else 1.0
    buyer_apology_otc = 0.0 if "구매자 사과: 있음" in result_text else 1.0

    buyer_points = (
        (refund_otc * b_rank1) + 
        ((1 - buyer_review_otc) * b_rank2) + 
        (seller_review_otc * b_rank3) + 
        (seller_apology_otc * b_rank4)
    )

    seller_points = (
        ((1 - refund_otc) * s_rank1) + 
        ((1 - seller_review_otc) * s_rank2) + 
        (buyer_review_otc * s_rank3) + 
        (buyer_apology_otc * s_rank4)
    )

    return buyer_points, seller_points

def _calculate_rewards(state, result_text):
    """
    [OTC 정의]
    refund otc : 1 (full)/ .5 (half) / 0 (none)
    reviews otc: 1 (retract) / 0 (keep)
    apology otc: 0 (didn't) / 1 (did) -> 차이점

    [점수 공식]
    buyer points = (Refund_otc * Rank_1) + (1-MyReview_otc ) * Rank_2 + (OtherReview_otc * Rank_3) + (Apology_otc * Rank_4)
    seller_points = (1 - Refund_otc) * Rank_1 + (1 - MyReview_otc) * Rank_2 + (OtherReview_otc * Rank_3) + (Apology_otc * Rank_4) 

    기존 연구: 사과를 하지 않은 경우 (부정적 결과) 1로 코딩
    agent reward: 에이전트가 목표 달성을 긍정적 신호로 학습하도록 apology를 positive action으로 구현함
    """
    buyer_goals = state["user_goals"] if state["user_role"] == "구매자" else state["ai_goals"]
    seller_goals = state["ai_goals"] if state["user_role"] == "구매자" else state["user_goals"]

    b_rank1 = buyer_goals.get("환불 받기", 0)
    b_rank2 = buyer_goals.get("판매자에 대한 부정적인 리뷰 유지하기", 0)
    b_rank3 = buyer_goals.get("판매자가 나에 대한 부정적인 리뷰 철회하기", 0)
    b_rank4 = buyer_goals.get("상대로부터 공식적인 사과받기", 0)

    s_rank1 = seller_goals.get("환불 방어", 0)
    s_rank2 = seller_goals.get("구매자에 대한 부정적인 리뷰 유지하기", 0)
    s_rank3 = seller_goals.get("구매자가 나에 대한 부정적인 리뷰 철회하기", 0)
    s_rank4 = seller_goals.get("상대로부터 공식적인 사과받기", 0)

    refund_otc = 0.0
    if "환불: 전체" in result_text:
        refund_otc = 1.0
    elif "환불: 부분" in result_text:
        refund_otc = 0.5
    else:
        refund_otc = 0.0

    buyer_review_otc = 1.0 if "구매자 리뷰: 철회" in result_text else 0.0
    seller_review_otc = 1.0 if "판매자 리뷰: 철회" in result_text else 0.0

    seller_apology_otc = 1.0 if "판매자 사과: 있음" in result_text else 0.0
    buyer_apology_otc = 1.0 if "구매자 사과: 있음" in result_text else 0.0

    buyer_reward = (
        (refund_otc * b_rank1) + 
        ((1 - buyer_review_otc) * b_rank2) + 
        (seller_review_otc * b_rank3) + 
        (seller_apology_otc * b_rank4)
    )

    seller_reward = (
        ((1 - refund_otc) * s_rank1) + 
        ((1 - seller_review_otc) * s_rank2) + 
        (buyer_review_otc * s_rank3) + 
        (buyer_apology_otc * s_rank4)
    )

    return buyer_reward, seller_reward

def _save_result_to_csv(state, result_text, buyer_points, seller_points):
    save_dir = "conversations"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    unique_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_name = f"Reflexion_Result_{timestamp}_{unique_id}.csv"
    file_path = os.path.join(save_dir, file_name)

    formatted_history = []
    for m in state["messages"]:
        speaker = state["user_role"] if m.type == "human" else state["ai_role"]
        content = m.content.strip()
        formatted_history.append([speaker, content])

    buyer_goals = state["user_goals"] if state["user_role"] == "구매자" else state["ai_goals"]
    seller_goals = state["ai_goals"] if state["user_role"] == "구매자" else state["user_goals"]

    df = pd.DataFrame(formatted_history, columns=["화자", "발화"])
    df["session_id"] = f"{timestamp}_{unique_id}"
    df["Human Role"] = state["user_role"]
    df["AI Role"] = state["ai_role"]
    
    df["구매자 목표"] = str(buyer_goals)
    df["판매자 목표"] = str(seller_goals)
    
    df["구매자 점수"] = buyer_points
    df["판매자 점수"] = seller_points
    df["평가 상세"] = result_text

    df.to_csv(file_path, index=False, encoding="utf-8-sig")

def _get_weighted_priority(state: NegotiationState, include_instruction: bool = True) -> str:
    goals = state.get("ai_goals", {})
    if not goals:
        return state.get("ai_priority", "")

    sorted_goals = sorted(goals.items(), key=lambda x: x[1], reverse=True)
    
    priority_lines = []
    total_score = 0
    
    for rank, (goal, score) in enumerate(sorted_goals, 1):
        total_score += score
        
        if score >= 70:
            tag = "[절대 사수/타협 불가]"
        elif score >= 30:
            tag = "[중요/부분 타협 가능]"
        else:
            tag = "[협상 카드/양보 가능]"
            
        priority_lines.append(f"{rank}순위. {goal} (배점: {score}점) {tag}")

    priority_content = "\n".join(priority_lines)

    if include_instruction:
        strategy_instruction = (
            f"당신의 목표는 위 항목들의 달성 여부에 따른 '총 획득 점수'를 최대화하는 것입니다.\n"
            f"- 배점이 높은 항목은 반드시 지켜야 합니다.\n"
            f"- 배점이 낮은 항목은 배점이 높은 항목을 얻기 위한 Trade-off로 적극 활용하세요.\n"
            f"- 상대가 배점이 높은 항목을 위협하면 강하게 방어하고, 배점이 낮은 항목을 요구하면 쿨하게 양보하여 신뢰를 얻으세요."
        )
        return f"{priority_content}\n{strategy_instruction}"
    else:
        return priority_content

def _parse_reflections(reflections) -> str:
    """Reflection 객체 안전 변환"""
    safe_reflections = []
    for r in reflections:
        if isinstance(r, str):
            safe_reflections.append(r)
        elif hasattr(r, "content"):
            safe_reflections.append(r.content)
        else:
            safe_reflections.append(str(r))
    return "\n".join(safe_reflections)

def _create_llm(state, temperature):
    """
    provider/model 형식의 문자열을 파싱하여 init_chat_model을 안전하게 호출하는 헬퍼 함수
    """
    full_model_name = state.get("model", "gpt-4o").strip()

    if "claude" in full_model_name.lower():
        if "/" in full_model_name:
            model_name = full_model_name.split("/", 1)[1]
        else:
            model_name = full_model_name
            
        model_name = model_name.strip()
        
        return ChatAnthropic(
            model=model_name,
            temperature=temperature
        )

    if full_model_name == "gpt-4o" or "gpt" in full_model_name.lower():
         return ChatOpenAI(
            model="gpt-4o",
            temperature=temperature
        )

    return init_chat_model(model=full_model_name, temperature=temperature)