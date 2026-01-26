import streamlit as st
import uuid
import sys
import os
from langchain_core.messages import HumanMessage, AIMessage

from baseline.graph import create_graph as build_baseline_graph
from react_reflexion.graph import build_reflexion_graph



# --------------------------------------------------------------------------
# UI 및 세션 초기화
# --------------------------------------------------------------------------
st.set_page_config(page_title="HCI Negotiation Agent", layout="wide", page_icon="🤝")

if "is_started" not in st.session_state:
    st.session_state.is_started = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
if "negotiation_status" not in st.session_state:
    st.session_state.negotiation_status = "진행 중"

# --------------------------------------------------------------------------
# 화면 분기 (설정 화면 vs 채팅 화면)
# --------------------------------------------------------------------------

if not st.session_state.is_started:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True) # 상단 여백
        st.title("🤖 HCI Negotiation Agent")
        st.markdown("### 협상 AI 에이전트 실험 플랫폼")
        st.info("실험 설정을 완료하고 '협상 시작' 버튼을 눌러주세요.")
        
        with st.container(border=True):
            # 1. 모드 선택
            mode = st.radio(
                "🧪 실험 모드 선택",
                ["Baseline (기본)", "ReAct+Reflexion (자기성찰)"],
                index=1,
                captions=["기본 에이전트", "실패 시 스스로 반성하고 재도전하는 에이전트"]
            )
            
            # 2. 역할 선택
            role = st.selectbox("👤 사용자 역할", ["구매자", "판매자"])
            
            # 3. 모델 선택
            model_name = st.selectbox(
                "🧠 LLM 모델 선택",
                ("gpt-4o", "claude-3-5-sonnet-latest"),
                index=0
            )

            st.markdown("---")
            
            # 시작 버튼
            if st.button("🚀 협상 시작하기", use_container_width=True, type="primary"):
                # 세션 초기화 및 그래프 로드
                st.session_state.mode = "Baseline" if "Baseline" in mode else "Reflexion"
                st.session_state.user_role = role
                st.session_state.model_name = model_name
                st.session_state.config["configurable"]["thread_id"] = str(uuid.uuid4())
                st.session_state.messages = [] # 화면 표시용 메시지 초기화
                
                # 그래프 선택 로드
                if st.session_state.mode == "Baseline":
                    st.session_state.graph = build_baseline_graph()
                else:
                    st.session_state.graph = build_reflexion_graph()
                
                # 초기 실행 (Setup -> 첫 발화 유도)
                # setup_node가 초기 state를 반환하므로 이를 반영해야 함
                init_inputs = {"user_role": role, "model": model_name, "messages": []}
                
                # Setup 단계 실행 (Reflexion의 경우 setup -> negotiator까지 흐름)
                with st.spinner("에이전트를 초기화하고 시나리오를 로드 중입니다..."):
                    try:
                        # stream 모드로 실행하여 첫 메시지를 가져옴
                        for event in st.session_state.graph.stream(init_inputs, st.session_state.config):
                            for node, data in event.items():
                                # 노드별 출력 처리 (디버깅 및 UI 표시)
                                if "messages" in data and data["messages"]:
                                    last_msg = data["messages"][-1]
                                    if isinstance(last_msg, AIMessage):
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": last_msg.content,
                                            "avatar": "🤖"
                                        })
                    except Exception as e:
                        st.error(f"초기화 중 오류 발생: {e}")
                        st.stop()

                st.session_state.is_started = True
                st.rerun()
else:
    # 사이드바: 현재 상태 정보
    with st.sidebar:
        st.title("실험 정보")
        st.write(f"**모드:** {st.session_state.mode}")
        st.write(f"**내 역할:** {st.session_state.user_role}")
        st.write(f"**상대방:** {'판매자' if st.session_state.user_role == '구매자' else '구매자'}")
        st.write(f"**모델:** {st.session_state.model_name}")
        
        st.divider()
        if st.button("🔄 실험 다시 하기 (초기화)", type="secondary"):
            st.session_state.is_started = False
            st.session_state.messages = []
            st.rerun()

    # 메인 채팅 영역
    st.chat_message("system", avatar="📝").write(f"**[SYSTEM]** {st.session_state.mode} 모드로 협상을 시작합니다.")

    # 1. 기존 메시지 렌더링
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=msg.get("avatar")):
            st.markdown(msg["content"])

    # 2. 사용자 입력 처리
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 UI 표시
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # 3. 그래프 실행 및 응답 대기
        with st.spinner("상대방이 생각 중입니다..."):
            # 그래프에 사용자 메시지 주입
            # LangGraph는 state의 'messages' 키에 append 됨
            inputs = {"messages": [HumanMessage(content=prompt)]}
            
            # Reflexion 모드일 경우 loop가 돌 수 있음 (Negotiator -> Evaluator -> Reflector -> Setup -> Negotiator)
            # 따라서 stream을 통해 중간 과정을 지켜봐야 함
            
            response_container = st.empty() # 스트리밍 또는 중간 과정 표시용
            
            for event in st.session_state.graph.stream(inputs, st.session_state.config):
                for node, data in event.items():
                    
                    # A. 협상가 (Negotiator / AI Agent) 노드
                    if node in ["negotiator", "ai_agent"]:
                        if "messages" in data and data["messages"]:
                            ai_msg = data["messages"][-1]
                            content = ai_msg.content

                            if not content: 
                                continue
                            
                            # UI에 추가 및 표시
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": content, 
                                "avatar": "🤖"
                            })
                            with st.chat_message("assistant", avatar="🤖"):
                                st.markdown(content)

                    # B. 평가자 (Evaluator) 노드 - Reflexion 전용
                    elif node == "evaluator":
                        result_text = data.get("final_result", "")
                        with st.status("⚖️ 협상 평가 진행 중...", expanded=True) as status:
                            st.write(result_text)
                            score_info = f"구매자 점수: {data.get('buyer_score')} / 판매자 점수: {data.get('seller_score')}"
                            st.info(score_info)
                            status.update(label="평가 완료", state="complete")

                    # C. 반성자 (Reflector) 노드 - Reflexion 전용
                    elif node == "reflector":
                        reflections = data.get("reflections", [])
                        if reflections:
                            last_reflection = reflections[-1]
                            with st.chat_message("system", avatar="🧠"):
                                st.warning(f"**[Self-Reflection]** 실패를 감지했습니다. 전략을 수정합니다:\n\n{last_reflection}")
                            # 반성 후에는 다시 Negotiator로 넘어가므로, 사용자는 잠시 기다려야 함

            # 4. 종료 상태 확인
            current_state = st.session_state.graph.get_state(st.session_state.config)
            # setup_node 등에서 is_finished를 관리하거나, evaluator가 끝났을 때 판단
            # Reflexion 그래프에서는 'evaluator'가 끝나고 'reflector'로 안 가면 종료임
            
            # (옵션) 그래프의 is_finished 값 확인
            if current_state.values.get("is_finished") and not current_state.next:
                 st.success("🎉 협상이 최종 종료되었습니다!")
                 st.balloons()