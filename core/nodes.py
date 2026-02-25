from core.state import NegotiationState
from langchain_core.messages import AIMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from tools.rag_tools import policy_search_tool
import uuid
from datetime import datetime
import copy
from langchain_core.runnables import RunnableParallel

from core.prompts import (
    COT_NEGOTIATOR_SYSTEM,
    COT_NEGOTIATOR_HUMAN, 
    COT_PREVIOUS_SYSTEM,
    COT_PREVIOUS_HUMAN,
    EVALUATOR_SYSTEM,
    EVALUATOR_HUMAN,
    REFLEXION_NEGOTIATOR_SYSTEM,
    REFLEXION_NEGOTIATOR_HUMAN,
    REFLEXION_REFLECTION_SYSTEM,
    REFLEXION_REFLECTION_HUMAN,
    BASELINE_SYSTEM,
    BASELINE_HUMAN,
    IRP_SYSTEM, IRP_HUMAN,
    SVI_SYSTEM, SVI_HUMAN,
    BATNA_SYSTEM, BATNA_HUMAN,
    STRATEGY_SYSTEM, STRATEGY_HUMAN,
    UTTERANCE_SYSTEM, UTTERANCE_HUMAN
)
from core.scenarios import (
    PRIORITIES,
    SCENARIOS,
    FEWSHOTS
)

from core.helpers import (
    create_llm,
    parse_json_content,
    get_weighted_priority,
    parse_reflections,
    calculate_points,
    calculate_rewards,
    calculate_nash_point,
    draw_pareto_plot,
    pareto_to_base64,
    save_result_to_csv,
    save_result_to_firebase,
    pareto_to_base64
)

PROMPT_REGISTRY = {
    "baseline": {
        "system": BASELINE_SYSTEM,
        "human": BASELINE_HUMAN
    },
    "cot_previous": {
        "system": COT_PREVIOUS_SYSTEM,
        "human": COT_PREVIOUS_HUMAN
    },
    "cot_upgrade": {
        "system": COT_NEGOTIATOR_SYSTEM,
        "human": COT_NEGOTIATOR_HUMAN
    },
    # 구(舊) Reflexion 모드 호환용 (UI에서는 사용 안 함)
    "reflexion": {
        "system": REFLEXION_NEGOTIATOR_SYSTEM,
        "human": REFLEXION_NEGOTIATOR_HUMAN
    }
}

def setup_node(state: NegotiationState):
    u_role = state.get("user_role", "구매자") 
    a_role = "판매자" if u_role == "구매자" else "구매자"
    model = state.get("model", "gpt-4o")
    mode = state.get("mode", "baseline")

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
        "max_retries": max_retries,

        # for Psychological Strategic (IRP-SVI)
        "irp_results": [],
        "batna_results": [],
        "instrumental": [],
        "self": [],
        "process": [],
        "relationship": [],
        "strategies": [],
        "irp_thoughts": [],
        "batna_thoughts": [],
        "svi_thoughts": [],
        "strategy_thoughts": [],
        "logger_thought": ""
    }
    return initial_state

def negotiator_node(state: NegotiationState):
    mode_val = state.get("mode", "baseline")

    # 새로운 모드 문자열 우선 사용, 기존 대문자 모드는 하위 호환용
    if mode_val in PROMPT_REGISTRY:
        mode = mode_val
    elif "CoT" in mode_val:
        # 예전 "CoT" 모드는 개선 버전으로 매핑
        mode = "cot_upgrade"
    elif "Baseline" in mode_val:
        mode = "baseline"
    elif "Reflexion" in mode_val:
        mode = "reflexion"
    else:
        mode = "baseline"

    templates = PROMPT_REGISTRY.get(mode, PROMPT_REGISTRY["baseline"])

    if mode == "baseline":
        llm = create_llm(state, temperature=0.9)
    else:
        tools = [policy_search_tool]
        llm = create_llm(state, temperature=0.9).bind_tools(tools)
    
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
    weighted_priority_context = get_weighted_priority(state, include_instruction=include_instruction)
    
    reflections_str = parse_reflections(state.get("reflections", []))

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
    
    parsed_response = parse_json_content(response.content)

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
    llm = create_llm(state, temperature=0.5)

    weighted_priority = get_weighted_priority(state, include_instruction=False)

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
    parsed_data = parse_json_content(content_text)

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
    
    dialogue = "\n".join([f"[{m.type}] {m.content}" for m in state["messages"]])

    human_eval = state.get("human_evaluation", {})
    
    result_text = (
        f"환불: {human_eval.get('refund', '없음')}\n"
        f"구매자 리뷰: {human_eval.get('buyer_review', '유지')}\n"
        f"판매자 리뷰: {human_eval.get('seller_review', '유지')}\n"
        f"구매자 사과: {human_eval.get('buyer_apology', '아니오')}\n"
        f"판매자 사과: {human_eval.get('seller_apology', '아니오')}"
    )
    
    logger_thought = human_eval.get("thought", "")
    
    state["logger_thought"] = logger_thought
    state["final_result"] = result_text

    unique_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{unique_id}_{timestamp}"

    buyer_points, seller_points = calculate_points(state, result_text)
    all_outcomes, nash_point = calculate_nash_point(state)

    # 파레토 그래프를 base64로 변환
    pareto_image_base64 = None
    try:
        pareto_image_base64 = pareto_to_base64(
            all_outcomes=all_outcomes,
            nash_point=nash_point,
            buyer_score=buyer_points,
            seller_score=seller_points,
            session_id=session_id
        )
    except Exception as e:
        print(f"파레토 그래프 base64 변환 실패: {e}")

    try:
        save_ok = save_result_to_firebase(
            state=state,
            dialogue=dialogue,
            result_text=result_text,
            buyer_points=buyer_points,
            seller_points=seller_points,
            session_id=session_id,
            pareto_image_base64=pareto_image_base64
        )
        if not save_ok:
            print("Firebase 저장 실패")
    except Exception as e:
        print(f"Firebase 저장 실패: {e}")
    
    # 로컬에 파레토 저장하는 로직

    # try:
    #     draw_pareto_plot(
    #         all_outcomes=all_outcomes,
    #         nash_point=nash_point,
    #         buyer_score=buyer_points,
    #         seller_score=seller_points,
    #         session_id=session_id
    #     )
    # 
    # except Exception as e:
    #     print(f"그래프 저장 실패: {e}")

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
    
    llm = create_llm(state, temperature=0.5)
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

    parsed_data = parse_json_content(content_text)

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

    buyer_reward, seller_reward = calculate_rewards(state, result_text)

    return {
        "buyer_reward": buyer_reward,
        "seller_reward": seller_reward,
        "evaluator_thought": evaluator_thought
    }

def analysis_node(state: NegotiationState):
    llm = create_llm(state=state, temperature=0.5)

    # IRP Chain
    irp_prompt = ChatPromptTemplate.from_messages([
        ("system", IRP_SYSTEM),
        ("human", IRP_HUMAN)
    ])
    irp_chain = irp_prompt | llm

    # BATNA Chain
    batna_prompt = ChatPromptTemplate.from_messages([
        ("system", BATNA_SYSTEM),
        ("human", BATNA_HUMAN)
    ])
    batna_chain = batna_prompt | llm

    # SVI Chain
    svi_prompt = ChatPromptTemplate.from_messages([
        ("system", SVI_SYSTEM),
        ("human", SVI_HUMAN)
    ])
    svi_chain = svi_prompt | llm
    
    messages = state.get("messages", [])
    
    # 툴 메시지 제외하고 대화 흐름 파악
    history_msgs = [m for m in messages if m.type != "tool"]
    last_message = history_msgs[-1].content if history_msgs else ""
    history = "\n".join([f"[{m.type}] {m.content}" for m in history_msgs])
    
    parallel_chain = RunnableParallel({
        "irp": irp_chain,
        "batna": batna_chain,
        "svi": svi_chain
    })
    
    # 3. 실행
    results = parallel_chain.invoke({
        "last_message": last_message,
        "messages": history
    })

    irp_data = parse_json_content(results["irp"].content) or {}
    batna_data = parse_json_content(results["batna"].content) or {}
    svi_data = parse_json_content(results["svi"].content) or {}

    # 결과 추출 (프롬프트의 JSON 구조 반영)
    irp_res = irp_data.get("irp_category", "Interests")
    batna_res = batna_data.get("batna_strength", "Mid")

    inst_val = float(svi_data.get("instrumental", 0.0))
    self_val = float(svi_data.get("self", 0.0))
    proc_val = float(svi_data.get("process", 0.0))
    rel_val = float(svi_data.get("relationship", 0.0))

    irp_thought = irp_data.get("thought", "")
    batna_thought = batna_data.get("thought", "")
    svi_thought = svi_data.get("thought", "")
    
    # 4. State 업데이트
    return {
        "irp_results": [irp_res],
        "batna_results": [batna_res],
        "instrumental": [inst_val],
        "self": [self_val],
        "process": [proc_val],
        "relationship": [rel_val],
        "irp_thoughts": [irp_thought],
        "batna_thoughts": [batna_thought],
        "svi_thoughts": [svi_thought]
    }