from .state import NegotiationState
from langchain_core.messages import AIMessage
from langchain_core.messages import RemoveMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from tools.rag_tools import policy_search_tool
import uuid
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

    # 초기 State 구성
    initial_state = {
        "messages": delete_messages, 
        "user_role": u_role, 
        "ai_role": a_role, 
        "user_scenario": SCENARIOS[u_role],
        "ai_scenario": SCENARIOS[a_role],
        "user_priority": PRIORITIES[u_role],
        "ai_priority": PRIORITIES[a_role],
        "is_finished": False,
        "model": model
    }
    return initial_state

def negotiator_node(state: NegotiationState):
    tools = [policy_search_tool]
    llm = init_chat_model(model=state["model"], temperature=0.9).bind_tools(tools)
    
    recent_msgs = state["messages"][-4:] if state["messages"] else []

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
        "priority": state["ai_priority"],
        "recent_summary": "\n".join([f"{type(m).__name__}: {m.content}" for m in recent_msgs]),
        "reflections": "\n".join(state.get("reflections", [])),
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

    unique_id = str(uuid.uuid4())[:8] 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_{unique_id}"
    
    dialogue = "\n".join([f"[{m.type}] {m.content}" for m in state["messages"]])

    system_message = SystemMessagePromptTemplate.from_template(
        template=EVALUATOR_SYSTEM_PROMPT  
    )

    human_message = HumanMessagePromptTemplate.from_template(
        template=EVALUATOR_HUMAN_PROMPT,
        input_variables=["dialogue", "buyer_priority", "seller_priority"]
    )
    
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    chain = prompt | llm | StrOutputParser()
    
    result_text = chain.invoke({
        "dialogue": dialogue,
        "buyer_priority": state["ai_priority"] if state['ai_role'] == "판매자" else state["user_priority"], 
        "seller_priority": state["ai_priority"] if state['ai_role'] == "구매자" else state["user_priority"]
    }).split("최종 결과:")[-1].strip()

    buyer_score = 0
    seller_score = 0

    # 환불 점수
    if "환불: 완전" in result_text:
        buyer_score += 70
        seller_score += 0
    elif "환불: 부분" in result_text:
        buyer_score += 50
        seller_score += 20
    elif "환불: 없음" in result_text:
        buyer_score += 0
        seller_score += 70

    # 차선책 점수
    if "구매자 차선책: 달성" in result_text:
        buyer_score += 30
    if "판매자 차선책: 달성" in result_text:
        seller_score += 30

    # 결과 저장
    save_dir = "conversations"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = f"Reflexion_Result{timestamp}_{session_id}.csv"
    file_path = os.path.join(save_dir, file_name)

    formatted_history = []
    for m in state["messages"]:
        speaker = state["user_role"] if m.type == "human" else state["ai_role"]
        content = m.content.strip()
        formatted_history.append([speaker, content])

    df = pd.DataFrame(formatted_history, columns=["화자", "발화"])
    df["회차"] = session_id
    df["구매자 우선순위"] = state["ai_priority"] if state['ai_role'] == "판매자" else state["user_priority"]
    df["판매자 우선순위"] = state["ai_priority"] if state['ai_role'] == "구매자" else state["user_priority"]
    df["구매자 점수"] = buyer_score 
    df["판매자 점수"] = seller_score
    df["평가 결과"] = result_text

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