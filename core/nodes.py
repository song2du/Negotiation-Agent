from core.state import NegotiationState
import itertools
import matplotlib.pyplot as plt
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
import json
import pandas as pd
from core.prompts import (
    COT_NEGOTIATOR_SYSTEM,
    COT_NEGOTIATOR_HUMAN, 
    EVALUATOR_SYSTEM,
    EVALUATOR_HUMAN,
    REFLEXION_NEGOTIATOR_SYSTEM,
    REFLEXION_NEGOTIATOR_HUMAN,
    REFLEXION_REFLECTION_SYSTEM,
    REFLEXION_REFLECTION_HUMAN,
    BASELINE_SYSTEM,
    BASELINE_HUMAN 
)
from core.scenarios import (
    PRIORITIES,
    SCENARIOS,
    FEWSHOTS
)
from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)

PROMPT_REGISTRY = {
    "reflexion": {
        "system": REFLEXION_NEGOTIATOR_SYSTEM,
        "human": REFLEXION_NEGOTIATOR_HUMAN
    },
    "cot": {
        "system": COT_NEGOTIATOR_SYSTEM,
        "human": COT_NEGOTIATOR_HUMAN
    },
    "baseline": {
        "system": BASELINE_SYSTEM,
        "human": BASELINE_HUMAN
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
    mode_val = state.get("mode", "reflexion")
    
    if "CoT" in mode_val:
        mode = "cot"
    elif "Baseline" in mode_val:
        mode = "baseline"
    elif "Reflexion" in mode_val:
        mode = "reflexion"

    templates = PROMPT_REGISTRY.get(mode, PROMPT_REGISTRY["reflexion"])

    if mode == "baseline":
        llm = _create_llm(state, temperature=0.9)
    else:
        tools = [policy_search_tool]
        llm = _create_llm(state, temperature=0.9).bind_tools(tools)
    
    # 메시지 분류: 현재 진행 중인 Tool Chain과 완료된 대화 기록(History) 분리
    all_msgs = state.get("messages", [])
    chat_history = []
    tool_chain = []
    
    # 1. 툴 체인인지 확인 (마지막 메시지가 툴 메시지이거나, AI의 툴 호출인 경우)
    idx = len(all_msgs)
    for i in range(len(all_msgs) - 1, -1, -1):
        m = all_msgs[i]
        is_tool_msg = (m.type == "tool")
        is_ai_tool_call = (m.type == "ai" and hasattr(m, "tool_calls") and m.tool_calls)
        
        if is_tool_msg or is_ai_tool_call:
            idx = i
        else:
            # 툴과 관련 없는 메시지를 만나면 체인 종료
            break
            
    chat_history = all_msgs[:idx]
    tool_chain = all_msgs[idx:] # 여기에는 ToolMessage나 AI(tool_calls)가 들어감

    # 2. 화자 식별 매핑 함수 (내부 함수)
    def _map_role(m):
        if m.type == "human":
            return state["user_role"]
        elif m.type == "ai":
            return state["ai_role"]
        elif m.type == "tool":
            return "시스템(Tool)"
        return m.type

    # 3. recent_summary 생성 (chat_history만 사용, 툴 체인은 제외)
    recent_summary = "\n".join([f"{_map_role(m)}: {m.content}" for m in chat_history])

    include_instruction = False if mode == "baseline" else True
    weighted_priority_context = _get_weighted_priority(state, include_instruction=include_instruction)
    
    reflections_str = _parse_reflections(state.get("reflections", []))

    # 4. last_message 결정 (chat_history 기준)
    if chat_history:
        last_msg_obj = chat_history[-1]
        last_message = last_msg_obj.content
    else:
        last_message = f"이제 협상을 시작합니다. 당신은 {state['ai_role']}로서 상대방에게 첫 마디를 건네세요."

    # 5. 프롬프트 구성
    system_template = templates["system"]
    human_template = templates["human"]
    
    system_message = SystemMessagePromptTemplate.from_template(system_template)
    
    # 5-1. 시스템 메시지
    prompt_msgs = [system_message]
    
    # 5-2. 휴먼 메시지 & 툴 체인 처리
    # 툴 체인 중이라면 중간 메시지로 삽입하여 문맥 유지
    if tool_chain:
        prompt_msgs.extend(tool_chain)
    else:
        # 일반 대화 턴이면 Human Message Template 추가
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        prompt_msgs.append(human_message)

    prompt = ChatPromptTemplate.from_messages(prompt_msgs)

    chain = prompt | llm

    # 6. invoke 입력 구성
    invoke_input = {
        "role": state["ai_role"],
        "opponent": state["user_role"],
        "scenario": state["ai_scenario"],
        "priority": weighted_priority_context,
        "recent_summary": recent_summary,
        "reflections": reflections_str,
        "fewshot": FEWSHOTS,
        # tool_chain이 없을 때만 사용되지만, 안전을 위해 전달 (템플릿에 없으면 무시됨)
        "last_message": last_message
    }

    response = chain.invoke(invoke_input)

    if hasattr(response, "tool_calls") and response.tool_calls:
        return {"messages": [response]}
    
    parsed_response = _parse_json_content(response.content)

    thought = ""
    clean_response = response.content

    if parsed_response:
        thought = parsed_response.get("thought", "")
        clean_response = parsed_response.get("response", "")

    ai_message = AIMessage(
        content=clean_response, 
        additional_kwargs={"thought": thought} 
    )

    return {"messages": [ai_message]}

def reflection_node(state: NegotiationState):
    llm = _create_llm(state, temperature=0.5)

    weighted_priority = _get_weighted_priority(state, include_instruction=False)

    # 화자 식별을 명확하게 하기 위해 역할 이름 매핑
    def _map_role(m_type):
        if m_type == "human":
            return state["user_role"]
        elif m_type == "ai":
            return state["ai_role"]
        elif m_type == "tool":
            return "시스템(Tool)"
        return m_type

    trajectory = "\n".join([f"[{_map_role(m.type)}] {m.content}" for m in state["messages"]])
    reflections = "\n".join(state.get("reflections", []))

    if state["ai_role"] == "구매자":
        my_reward = state.get("buyer_reward", 0.0)
    else:
        my_reward = state.get("seller_reward", 0.0)
    
    scores = f"이번 협상에서 획득한 나의 점수: {my_reward}점"

    system_message = SystemMessagePromptTemplate.from_template(
        template=REFLEXION_REFLECTION_SYSTEM
    )

    human_message = HumanMessagePromptTemplate.from_template(
        template=REFLEXION_REFLECTION_HUMAN
    )
    
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    chain = prompt | llm 

    response = chain.invoke({
        "role": state["ai_role"],
        "scenario": state["ai_scenario"],
        "priority": weighted_priority,
        "scores": scores,
        "past_reflections": reflections,
        "trajectory": trajectory
    })

    content_text = response.content if hasattr(response, "content") else str(response)
    parsed_data = _parse_json_content(content_text)

    reflection_thought = ""
    reflection = content_text

    if parsed_data:
        reflection_thought = parsed_data.get("thought", "")
        reflection = parsed_data.get("result", "")


    return {
        "reflections": [reflection],
        "reflection_thoughts": [reflection_thought]
    }

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

    chain = prompt | llm
    
    response = chain.invoke({
        "dialogue": dialogue,
    })
    content_text = response.content if hasattr(response, "content") else str(response)

    parsed_data = _parse_json_content(content_text)

    logger_thought = ""
    result_text = content_text

    if parsed_data:
        logger_thought = parsed_data.get("thought", "")
        result_dict = parsed_data.get("result", {})
        
        result_text = (
            f"환불: {result_dict.get('refund', '없음')}\n"
            f"구매자 리뷰: {result_dict.get('buyer_review', '유지')}\n"
            f"판매자 리뷰: {result_dict.get('seller_review', '유지')}\n"
            f"구매자 사과: {result_dict.get('buyer_apology', '아니오')}\n"
            f"판매자 사과: {result_dict.get('seller_apology', '아니오')}"
        )
    
    state["logger_thought"] = logger_thought
    state["final_result"] = result_text

    unique_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{unique_id}_{timestamp}"

    buyer_points, seller_points = _calculate_points(state, result_text)
    all_outcomes, nash_point = _calculate_nash_point(state)

    try:
        _save_result_to_csv(
            state=state, 
            dialogue=dialogue, 
            result_text=result_text,
            buyer_points=buyer_points, 
            seller_points=seller_points, 
            session_id=session_id)
    
    except Exception as e:
        print(f"csv 저장 실패: {e}")
    
    try:
        _draw_pareto_plot(
            all_outcomes=all_outcomes,
            nash_point=nash_point,
            buyer_score=buyer_points,
            seller_score=seller_points,
            session_id=session_id
        )
    
    except Exception as e:
        print(f"그래프 저장 실패: {e}")

    # 7. 최종 상태 업데이트
    return {
        "final_result": result_text,
        "buyer_score": buyer_points,
        "seller_score": seller_points,
        "logger_thought": logger_thought,
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

    chain = prompt | llm 
    
    response = chain.invoke({
        "dialogue": dialogue,
    })
    content_text = response.content if hasattr(response, "content") else str(response)

    parsed_data = _parse_json_content(content_text)

    evaluator_thought = ""
    result_text = content_text

    if parsed_data:
        evaluator_thought = parsed_data.get("thought", "")
        result_dict = parsed_data.get("result", {})
        
        result_text = (
            f"환불: {result_dict.get('refund', '없음')}\n"
            f"구매자 리뷰: {result_dict.get('buyer_review', '유지')}\n"
            f"판매자 리뷰: {result_dict.get('seller_review', '유지')}\n"
            f"구매자 사과: {result_dict.get('buyer_apology', '아니오')}\n"
            f"판매자 사과: {result_dict.get('seller_apology', '아니오')}"
        )

    buyer_reward, seller_reward = _calculate_rewards(state, result_text)

    return {
        "buyer_reward": buyer_reward,
        "seller_reward": seller_reward,
        "evaluator_thought": evaluator_thought
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

def _save_result_to_csv(state, dialogue, result_text, buyer_points, seller_points, session_id):
    save_dir = "conversations"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mode_prefix = state.get("mode", "Negotiation")

    file_name = f"{mode_prefix}_Result_{session_id}.csv"
    file_path = os.path.join(save_dir, file_name)

    formatted_history = []
    
    for m in state["messages"]:
        speaker = state["user_role"] if m.type == "human" else state["ai_role"]
        if m.type == "tool":
            speaker = "Tool"
        content = m.content.strip()
        thought = m.additional_kwargs.get("thought", "")
        tool_calls = ""
        if hasattr(m, "tool_calls") and m.tool_calls:
            tool_calls = str(m.tool_calls)

        formatted_history.append([speaker, content, thought, tool_calls])

    buyer_goals = state["user_goals"] if state["user_role"] == "구매자" else state["ai_goals"]
    seller_goals = state["ai_goals"] if state["user_role"] == "구매자" else state["user_goals"]

    df = pd.DataFrame(formatted_history, columns=["speaker", "utterance", "negotiator_thought", "tool_calls"])
    
    df["session_id"] = f"{session_id}"
    df["human_role"] = state["user_role"]
    df["ai_role"] = state["ai_role"]
    df["full_dialogue"] = dialogue
    
    df["buyer_goals"] = str(buyer_goals)
    df["seller_goals"] = str(seller_goals)
    
    df["buyer_points"] = buyer_points
    df["seller_points"] = seller_points
    df["evaluation_details"] = result_text

    eval_thoughts = state.get("evaluator_thought", [])

    if isinstance(eval_thoughts, list):
        df["evaluator_thoughts_all"] = "\n---\n".join(eval_thoughts)
    else:
        df["evaluator_thoughts_all"] = str(eval_thoughts)
    
    ref_thoughts = state.get("reflection_thoughts", [])
    if isinstance(ref_thoughts, list):
        df["reflector_thoughts_all"] = "\n---\n".join(ref_thoughts)
    else:
        df["reflector_thoughts_all"] = str(ref_thoughts)
    
    df["logger_thoughts"] = state.get("logger_thought", "")

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

def _calculate_nash_point(state):
    """
    가능한 모든 협상 결과를 시뮬레이션하여
    1) 모든 가능한 점수 리스트 (파레토 구름용)
    2) Nash Point (최적 합의점)
    두 가지를 반환합니다.
    """
    buyer_goals = state["user_goals"] if state["user_role"] == "구매자" else state["ai_goals"]
    seller_goals = state["ai_goals"] if state["user_role"] == "구매자" else state["user_goals"]

    # 각 항목의 배점 가져오기
    b_refund = buyer_goals.get("환불 받기", 0)
    b_my_review = buyer_goals.get("판매자에 대한 부정적인 리뷰 유지하기", 0)
    b_your_review = buyer_goals.get("판매자가 나에 대한 부정적인 리뷰 철회하기", 0)
    b_apology = buyer_goals.get("상대로부터 공식적인 사과받기", 0)

    s_refund = seller_goals.get("환불 방어", 0)
    s_my_review = seller_goals.get("구매자에 대한 부정적인 리뷰 유지하기", 0) # 판매자가 쓴 리뷰 유지
    s_your_review = seller_goals.get("구매자가 나에 대한 부정적인 리뷰 철회하기", 0) # 구매자가 쓴 리뷰 철회
    s_apology = seller_goals.get("상대로부터 공식적인 사과받기", 0)

    all_outcomes = []
    max_product = -1
    nash_point = (0, 0)

    # 모든 경우의 수 순회 (3 x 2 x 2 x 2 x 2 = 48가지)
    # 환불 (1.0:전액, 0.5:부분, 0.0:없음)
    refund_opts = [1.0, 0.5, 0.0]
    # 구매자 리뷰 (1.0:철회, 0.0:유지)
    b_review_opts = [1.0, 0.0] 
    # 판매자 리뷰 (1.0:철회, 0.0:유지)
    s_review_opts = [1.0, 0.0]
    # 사과 (1.0: 안함, 0.0 함)
    b_apology_opts = [1.0, 0.0] 
    s_apology_opts = [1.0, 0.0]

    for rf, br, sr, ba, sa in itertools.product(refund_opts, b_review_opts, s_review_opts, b_apology_opts, s_apology_opts):
        b_score = (rf * b_refund) + ((1 - br) * b_my_review) + (sr * b_your_review) + (sa * b_apology)
        s_score = ((1 - rf) * s_refund) + ((1 - sr) * s_my_review) + (br * s_your_review) + (ba * s_apology)

        all_outcomes.append((b_score, s_score))
        # Nash Product (단, 점수가 0보다 작을 수 없다고 가정)
        product = b_score * s_score
        if product > max_product:
            max_product = product
            nash_point = (b_score, s_score)
            
    return all_outcomes, nash_point

def _draw_pareto_plot(all_outcomes, nash_point, buyer_score, seller_score, session_id):
    """
    협상 결과를 시각화하여 저장하는 함수
    - all_outcomes: 회색 구름 (가능한 모든 결과)
    - nash_point: 금색 별 (최적점)
    - buyer/seller_score: 빨간 점 (실제 결과)
    """
    save_dir = "images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    image_filename = f"Pareto_{session_id}.png"
    image_path = os.path.join(save_dir, image_filename)

    plt.switch_backend('Agg') 
    plt.figure(figsize=(7, 7))

    # 가능한 모든 영역
    all_b = [p[0] for p in all_outcomes]
    all_s = [p[1] for p in all_outcomes]
    plt.scatter(all_b, all_s, color='gray', alpha=0.3, s=50, label='Possible Outcomes')

    # 프론티어 라인
    sorted_points = sorted(all_outcomes, key=lambda x: x[0], reverse=True)
    frontier = []
    max_y = -1
    for x, y in sorted_points:
        if y > max_y:
            frontier.append((x, y))
            max_y = y
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    plt.plot(fx, fy, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label='Pareto Frontier')
    
    # nash point
    nx, ny = nash_point
    plt.scatter(nx, ny, color='gold', marker='*', s=300, edgecolors='orange', zorder=10, label='Nash Point (Ideal)')
    plt.text(nx - 10, ny + 5, f"Nash\n({nx:.0f}, {ny:.0f})", fontsize=9, color='orange', fontweight='bold')

    # 현재 협상 결과
    plt.scatter(buyer_score, seller_score, color='red', s=100, zorder=5, label='Agreement Point')
    plt.text(buyer_score + 2, seller_score + 2, f"({buyer_score}, {seller_score})", fontsize=10, color='red')

    # 스타일 설정
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.xlabel("Buyer Score")
    plt.ylabel("Seller Score")
    plt.title("Negotiation Outcome (Pareto Check)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left')
    
    plt.savefig(image_path, dpi=100, bbox_inches='tight')
    plt.close()

def _parse_json_content(content: str):
    """마크다운 코드 블록 제거 및 JSON 파싱"""
    try:
        clean_content = content.strip()
        if clean_content.startswith("```"):
            clean_content = clean_content.replace("```json", "").replace("```", "")
        return json.loads(clean_content)
    except json.JSONDecodeError:
        print(f"JSON Parsing Failed: {content[:100]}...")
        return None