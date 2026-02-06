import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from core.scenarios import PRIORITIES

def render_messages(chat_placeholder):
    with chat_placeholder.container():
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar=msg.get("avatar")):
                st.markdown(msg["content"])

def normalize_text(text):
    """텍스트 비교를 위한 정규화 헬퍼 함수"""
    if not text: return ""
    return "".join(text.split())

def process_graph_stream(user_input):
    """
    사용자 입력을 받아 그래프를 실행하고, 이벤트를 처리하며 UI를 업데이트함.
    리턴값: boolean (reset_triggered 여부 - 리플렉션 등으로 인한 재시작 필요 시 True)
    """
    inputs = {"messages": [HumanMessage(content=user_input)]}
    reset_triggered = False
    
    # 중복 방지를 위한 기존 메시지 정규화 세트 생성
    existing_contents_normalized = set(
        normalize_text(msg["content"]) for msg in st.session_state.messages if msg.get("content")
    )
    
    # 직전 AI 메시지 확인 (연속 중복 방지)
    last_ai_content_normalized = ""
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "assistant":
            last_ai_content_normalized = normalize_text(msg.get("content", ""))
            break

    # 그래프 스트리밍 시작
    for event in st.session_state.graph.stream(inputs, st.session_state.config):
        for node, data in event.items():
            
            # [A] 협상가(AI) 노드 처리
            if node in ["negotiator"]:
                if "messages" in data and data["messages"]:
                    ai_msg = data["messages"][-1]
                    content = ai_msg.content
                    content_norm = normalize_text(content)

                    # 내용이 없거나, 이미 있는 내용이거나, 직전 내용과 같으면 스킵
                    if not content or \
                       content_norm in existing_contents_normalized or \
                       (last_ai_content_normalized and content_norm == last_ai_content_normalized):
                        continue

                    if not reset_triggered:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(content)
                    
                    # 세션에 기록
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": content, 
                        "avatar": "🤖"
                    })
                    # 중복 체크 리스트 업데이트
                    existing_contents_normalized.add(content_norm)

            # [B] 평가자(Evaluator) 노드 처리
            elif node == "evaluator":
                result_text = data.get("final_result", "")
                if not reset_triggered:
                    with st.status("⚖️ 협상 평가 진행 중...", expanded=True) as status:
                        st.write(result_text)
                        score_info = f"구매자 점수: {data.get('buyer_reward')} / 판매자 점수: {data.get('seller_reward')}"
                        st.info(score_info)
                        status.update(label="평가 완료", state="complete")

            # [C] 반성자(Reflector) 노드 처리 (Reflexion 모드)
            elif node == "reflector":
                reflections = data.get("reflections", [])
                if reflections:
                    # 현재 상태 스냅샷 가져오기
                    snapshot = st.session_state.graph.get_state(st.session_state.config)
                    current_reflections = snapshot.values.get("reflections", [])
                    max_retries = snapshot.values.get("max_retries", 3)
                    current_count = len(current_reflections) + 1 # 현재 시점

                    # 시스템 메시지 추가
                    warning_msg = (f"**[Self-Reflection]** ({current_count}/{max_retries}회)\n"
                                   "목표 달성에 실패했습니다. 전략을 수정하여 다시 협상합니다.")
                    
                    st.session_state.messages = [] # 화면 클리어
                    st.session_state.messages.append({
                            "role": "system",
                            "content": warning_msg,
                            "avatar": "🔄"
                    })
                    
                    reset_triggered = True
                    st.toast("전략 수정 중... 대화를 재설정합니다.", icon="🔄")

    return reset_triggered

def render_sidebar():
    """사이드바 정보 및 초기화 버튼 렌더링"""
    with st.sidebar:
        st.subheader("실험 정보")
        st.write(f"**모드:** {st.session_state.mode}")
        st.write(f"**내 역할:** {st.session_state.user_role}")
        st.write(f"**상대방:** {'판매자' if st.session_state.user_role == '구매자' else '구매자'}")
        st.write(f"**모델:** {st.session_state.model_name}")

        st.divider()
        st.subheader("내 우선순위")
        user_priorities = PRIORITIES.get(st.session_state.user_role, {})
        for item, score in user_priorities.items():
            st.write(f"- {item} ({score}점)")
        
        st.divider()
        if st.button("🔄 실험 다시 하기 (초기화)", type="secondary"):
            st.session_state.is_started = False
            st.session_state.messages = []
            st.rerun()

def render_chat_history():
    """저장된 대화 기록 렌더링"""
    # 시스템 메시지 (항상 상단 표시)
    st.chat_message("system", avatar="📝").write(f"**[SYSTEM]** {st.session_state.mode} 모드로 협상을 시작합니다.")

    # 대화 내용
    for msg in st.session_state.messages:
        # 시스템 메시지는 위에서 따로 처리했거나, messages 리스트에 포함되어 있다면 렌더링
        if msg["role"] == "system":
            with st.chat_message("system", avatar="🔄"): # Reflector 시스템 메시지 등
                st.markdown(msg["content"])
        else:
            with st.chat_message(msg["role"], avatar=msg.get("avatar")):
                st.markdown(msg["content"])

def check_negotiation_finished():
    """협상 종료 상태 확인 및 축하 효과"""
    current_state = st.session_state.graph.get_state(st.session_state.config)
    if current_state.values.get("is_finished") and not current_state.next:
         st.success("🎉 협상이 최종 종료되었습니다!")
         st.balloons()

def render_chat_screen():
    """채팅 화면 전체를 구성하는 메인 함수"""
    
    # 1. 사이드바 렌더링
    render_sidebar()

    # 2. 대화 기록 렌더링 (컨테이너 사용 권장)
    chat_container = st.container()
    with chat_container:
        render_chat_history()

    # 3. 사용자 입력 처리
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # (1) 사용자 메시지 즉시 표시
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

        # (2) AI 응답 처리 (스트리밍)
        with st.spinner("상대방이 생각 중입니다..."):
            # 여기서 복잡한 로직 함수 호출
            should_reset = process_graph_stream(prompt)
            
            if should_reset:
                st.rerun()
        
        # (3) 종료 체크
        check_negotiation_finished()