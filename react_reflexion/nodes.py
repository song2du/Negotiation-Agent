from .state import NegotiationState
from langchain_core.messages import AIMessage
from langchain_core.messages import RemoveMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from tools.rag_tools import policy_search_tool
import uuid
import copy
from datetime import datetime
import os
import pandas as pd
from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from .prompts import NEGOTIATOR_SYSTEM_PROMPT, EVALUATOR_SYSTEM_PROMPT, REFLECTION_SYSTEM_PROMPT
from .prompts import NEGOTIATOR_HUMAN_PROMPT, EVALUATOR_HUMAN_PROMPT, REFLECTION_HUMAN_PROMPT
from .scenarios import PRIORITIES, SCENARIOS

def setup_node(state: NegotiationState):
    u_role = state.get("user_role", "구매자") 
    a_role = "판매자" if u_role == "구매자" else "구매자"
    model = state.get("model", "gpt-4o")

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
        "user_scenario": SCENARIOS[u_role],
        "ai_scenario": SCENARIOS[a_role],
        "user_priority": user_priority_str,
        "ai_priority": ai_priority_str,
        "user_goals": user_goals,
        "ai_goals": ai_goals,
        "is_finished": False,
        "model": model
    }
    return initial_state

def negotiator_node(state: NegotiationState):
    tools = [policy_search_tool]
    llm = init_chat_model(model=state["model"], temperature=0.9).bind_tools(tools)
    
    recent_msgs = state["messages"][-4:] if state["messages"] else []

    # priority 가져오기
    goals = state.get("ai_goals", {})
    sorted_goals = sorted(goals.items(), key=lambda x: x[1], reverse=True)
    strategy_hint = ""
    (top_goal, top_score), (sub_goal, sub_score) = sorted_goals[0], sorted_goals[1]
    if top_score >= 70:
        strategy_hint = (
            f"당신에게 '{top_goal}'(배점: {top_score}점)은 절대 타협할 수 없는 최우선 가치입니다."
            f"이를 지키기 위해 '{sub_goal}'은 과감히 양보하거나 협상 카드로 사용해도 좋습니다."
        )
    elif 40 <= top_score <= 60:
        strategy_hint = (
            f"두 목표({top_goal}, {sub_goal})의 중요도가 대등합니다. "
            f"어느 하나를 포기하기보다, 두 가지를 모두 얻어낼 수 있는 창의적인 대안을 모색하세요."
        )
    else:
        strategy_hint = (
                f"'{top_goal}'에 조금 더 비중을 두되, 상황에 따라 유연하게 대처하세요."
            )
    full_priority_context = f"{state['ai_priority']}\n{strategy_hint}"

    # reflection 가져오기
    raw_reflections = state.get("reflections", [])
    safe_reflections = []
    for r in raw_reflections:
        if isinstance(r, str):
            safe_reflections.append(r)
        elif hasattr(r, "content"):
            safe_reflections.append(r.content)
        else:
            safe_reflections.append(str(r))
    
    # 프롬프트 구성
    system_message = SystemMessagePromptTemplate.from_template(
        template=NEGOTIATOR_SYSTEM_PROMPT,
        input_variables=["role", "opponent", "scenario", "priority", "recent_summary", 
                        "reflections"]
    )
    human_message = HumanMessagePromptTemplate.from_template(
        template=NEGOTIATOR_HUMAN_PROMPT,
        input_variables=["opponent", "last_message"]
    )
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    chain = prompt | llm

    response = chain.invoke({
        "role": state["ai_role"],
        "opponent": state["user_role"],
        "scenario": state["ai_scenario"],
        "priority": full_priority_context,
        "recent_summary": "\n".join([f"{type(m).__name__}: {m.content}" for m in recent_msgs]),
        "reflections": "\n".join(safe_reflections),
        "last_message": state["messages"][-1].content if state["messages"] else f"이제 협상을 시작합니다. 당신은 {state['ai_role']}로서 상대방에게 첫 마디를 건네세요."
    })

    if response.tool_calls:
        return {"messages": [response]}

    content = response.content
    if "최종 발화:" in content:
        clean_response = content.split("최종 발화:")[-1].strip()
    else:
        clean_response = content.strip()

    return {"messages": [AIMessage(content=clean_response)]}

def evaluation_node(state: NegotiationState):
    llm = init_chat_model(model=state["model"], temperature=0.5)

    if state["user_role"] == "구매자":
        buyer_goals_dict = state["user_goals"]
        seller_goals_dict = state["ai_goals"]
    else:
        buyer_goals_dict = state["ai_goals"]
        seller_goals_dict = state["user_goals"]

    b_items = sorted(buyer_goals_dict.items(), key=lambda x: x[1], reverse=True)
    s_items = sorted(seller_goals_dict.items(), key=lambda x: x[1], reverse=True)

    b_main_txt = b_items[0][0] if b_items else "목표 없음"
    b_sub_txt = b_items[1][0] if len(b_items) > 1 else "목표 없음"
    
    s_main_txt = s_items[0][0] if s_items else "목표 없음"
    s_sub_txt = s_items[1][0] if len(s_items) > 1 else "목표 없음"

    unique_id = str(uuid.uuid4())[:8] 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_{unique_id}"
    
    dialogue = "\n".join([f"[{m.type}] {m.content}" for m in state["messages"]])

    system_message = SystemMessagePromptTemplate.from_template(
        template=EVALUATOR_SYSTEM_PROMPT,
        input_variables=["buyer_main_goal", "buyer_sub_goal", "seller_main_goal", "seller_sub_goal"] 
    )

    human_message = HumanMessagePromptTemplate.from_template(
        template=EVALUATOR_HUMAN_PROMPT,
        input_variables=["dialogue", "buyer_main_goal", "buyer_sub_goal", "seller_main_goal", "seller_sub_goal"]
    )
    
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    chain = prompt | llm | StrOutputParser()
    
    result_text = chain.invoke({
        "dialogue": dialogue,
        "buyer_main_goal": b_main_txt,
        "buyer_sub_goal": b_sub_txt,
        "seller_main_goal": s_main_txt,
        "seller_sub_goal": s_sub_txt
    }).split("최종 결과:")[-1].strip()

    buyer_score = 0
    seller_score = 0

    if "구매자 제1목표: 완전" in result_text:
        buyer_score += 70
    elif "구매자 제1목표: 부분" in result_text:
        buyer_score += 50
    # 미달성은 0점 (자동)

    if "구매자 제2목표: 달성" in result_text:
        buyer_score += 30

    # (2) 판매자 점수 계산
    if "판매자 제1목표: 완전" in result_text:
        seller_score += 70
    elif "판매자 제1목표: 부분" in result_text:
        seller_score += 50

    if "판매자 제2목표: 달성" in result_text:
        seller_score += 30

    # 결과 저장
    save_dir = "conversations"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = f"Reflexion_Result_{session_id}.csv"
    file_path = os.path.join(save_dir, file_name)

    formatted_history = []
    for m in state["messages"]:
        speaker = state["user_role"] if m.type == "human" else state["ai_role"]
        content = m.content.strip()
        formatted_history.append([speaker, content])

    df = pd.DataFrame(formatted_history, columns=["화자", "발화"])

    df["session_id"] = session_id
    df["Human Role"] = state["user_role"]
    df["AI Role"] = state["ai_role"]

    df["구매자 목표"] = str(buyer_goals_dict)
    df["판매자 목표"] = str(seller_goals_dict)

    df["구매자 점수"] = buyer_score 
    df["판매자 점수"] = seller_score
    df["평가 상세"] = result_text

    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    # 7. 최종 상태 업데이트
    return {
        "final_result": result_text,
        "buyer_score": buyer_score,
        "seller_score": seller_score,
        "is_finished": True
    }

def reflection_node(state: NegotiationState):
    llm = init_chat_model(model=state["model"], temperature=0.5)

    trajectory = "\n".join([f"[{m.type}] {m.content}" for m in state["messages"]])
    reflections = "\n".join(state.get("reflections", []))

    scores = (
        f"최종 결과: {state.get('final_result', 'N/A')}\n"
        f"구매자 점수: {state.get('buyer_score', 0)}\n"
        f"판매자 점수: {state.get('seller_score', 0)}"
    )

    system_message = SystemMessagePromptTemplate.from_template(
        template=REFLECTION_SYSTEM_PROMPT  
    )

    human_message = HumanMessagePromptTemplate.from_template(
        template=REFLECTION_HUMAN_PROMPT,
        input_variables=["role", "scenario", "priority", "scores", "past_reflections", "trajectory"]
    )
    
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    chain = prompt | llm

    response = chain.invoke({
        "role": state["ai_role"],
        "scenario": state["ai_scenario"],
        "priority": state["ai_priority"],
        "scores": scores,
        "past_reflections": reflections,
        "trajectory": trajectory
    })

    content = response.content

    if "최종 결과" in content:
        result_text = content.split("최종 결과:")[-1].strip()
    else:
        result_text = content.strip()

    return {"reflections": [result_text]}