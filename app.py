import streamlit as st
import uuid
import sys
import os
from langchain_core.messages import HumanMessage, AIMessage

from core.graphs import build_graph
from core.scenarios import PRIORITIES

def render_priority_editor(role, key_prefix):
    """
    PRIORITIES 딕셔너리에 정의된 목표들을 가져와서
    사용자가 이름과 배점을 수정할 수 있는 입력 폼을 렌더링함.
    """
    # 1. 기본값 가져오기 (없으면 빈 딕셔너리)
    defaults = PRIORITIES.get(role, {})
    
    updated_goals = {}
    total_score = 0
    
    # 2. 각 목표별 입력 필드 생성
    # Streamlit은 루프 안에서 위젯 생성 시 key가 고유해야 함
    for idx, (goal_name, score) in enumerate(defaults.items()):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_name = st.text_input(
                f"목표 {idx+1}", 
                value=goal_name, 
                key=f"{key_prefix}_name_{idx}",
                help="목표의 내용을 수정할 수 있습니다."
            )
            
        with col2:
            new_score = st.number_input(
                "배점", 
                min_value=0, 
                max_value=100, 
                value=score, 
                step=5,
                key=f"{key_prefix}_score_{idx}",
                help="이 목표의 중요도(점수)입니다."
            )
        
        if new_name: # 이름이 비어있지 않은 경우만 추가
            updated_goals[new_name] = int(new_score)
            total_score += new_score

    # 3. 총점 표시 (가이드용)
    if total_score != 100:
        st.caption(f"⚠️ 현재 총점: **{total_score}점** (연구 표준은 보통 100점 만점입니다)")
    else:
        st.caption(f"✅ 현재 총점: **100점** (완벽합니다)")
        
    return updated_goals

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
        st.title("HCI Lab Negotiation Agent")
        st.markdown("### 협상 AI 에이전트 실험 플랫폼")
        st.info("실험 설정을 완료하고 '협상 시작' 버튼을 눌러주세요.")
        
        with st.container(border=True):
            # 1. 모드 선택
            mode = st.radio(
                "🧪 실험 모드 선택",
                ["CoT+In-context learning", "ReAct+Reflexion"],
                index=0
            )
            if "Reflexion" in mode:
                max_retries = st.slider(
                    "🔄 최대 반성(Retry) 횟수 설정",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="협상 실패 시 전략을 수정하여 재시도할 최대 횟수입니다."
                )
            # 2. 역할 선택
            role = st.selectbox("👤 사용자 역할", ["구매자", "판매자"])
            model_options = {
                "GPT-4o": "gpt-4o",
                "Claude 3 Sonnet": "anthropic/claude-3-sonnet-20240229" 
            }
            # 3. 모델 선택
            selected_label = st.selectbox(
                "🧠 LLM 모델 선택",
                options=list(model_options.keys()),
                index=0
            )
            model_name = model_options[selected_label]


            st.markdown("---")

            st.markdown(f"#### 🎯 나 ({role})의 목표 설정")
            with st.expander("내 목표 상세 편집 (클릭)", expanded=True):
                user_goals_dict = render_priority_editor(role, key_prefix="user")

            # (2) 상대방 목표 설정
            ai_role_name = "판매자" if role == "구매자" else "구매자"
            st.markdown(f"#### 🤖 상대방 ({ai_role_name})의 목표 설정")
            with st.expander("상대방 목표 상세 편집 (클릭)", expanded=False):
                st.info("AI는 이 목표들을 달성하기 위해 전략을 수립합니다.")
                ai_goals_dict = render_priority_editor(ai_role_name, key_prefix="ai")

            st.markdown("---")
            
            # 시작 버튼
            if st.button("🚀 협상 시작하기", use_container_width=True, type="primary"):
                # 세션 초기화 및 그래프 로드
                st.session_state.mode = "CoT" if "CoT+In-context learning" in mode else "Reflexion"
                st.session_state.user_role = role
                st.session_state.model_name = model_name
                st.session_state.config["configurable"]["thread_id"] = str(uuid.uuid4())
                st.session_state.messages = [] # 화면 표시용 메시지 초기화
                
                st.session_state.graph = build_graph(st.session_state.mode)
                
                # 초기 실행 (Setup -> 첫 발화 유도)
                # setup_node가 초기 state를 반환하므로 이를 반영해야 함
                init_inputs = {
                    "user_role": role, 
                    "model": model_name, 
                    "messages": [],
                    "user_priority_inputs": user_goals_dict,
                    "ai_priority_inputs": ai_goals_dict,
                    "max_retries": max_retries
                }
                
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
    chat_placeholder = st.empty()
    def render_messages():
        with chat_placeholder.container():
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar=msg.get("avatar")):
                    st.markdown(msg["content"])
    render_messages()

    # 2. 사용자 입력 처리
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 UI 표시
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # 3. 그래프 실행 및 응답 대기
        with st.spinner("상대방이 생각 중입니다..."):
            inputs = {"messages": [HumanMessage(content=prompt)]}
            
            reset_triggered = False
            existing_contents = set(msg["content"] for msg in st.session_state.messages)
              
            for event in st.session_state.graph.stream(inputs, st.session_state.config):
                for node, data in event.items():
                    
                    # Negotiator 노드
                    if node in ["negotiator", "ai_agent"]:
                        if "messages" in data and data["messages"]:
                            ai_msg = data["messages"][-1]
                            content = ai_msg.content

                            if not content:
                                continue
                
                            if content in existing_contents:
                                continue
                            
                            if not reset_triggered:
                                # 이미 그려진 메시지들과 섞이지 않도록 새 컨테이너 사용
                                with st.chat_message("assistant", avatar="🤖"):
                                    st.markdown(content)
                            
                            # 화면에 보이는 것과 별개로 기록에는 남김
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": content, 
                                "avatar": "🤖"
                            })

                            existing_contents.add(content)

                    # B. Evaluator 노드
                    elif node == "evaluator":
                        result_text = data.get("final_result", "")
                        if not reset_triggered:
                            with st.status("⚖️ 협상 평가 진행 중...", expanded=True) as status:
                                st.write(result_text)
                                score_info = f"구매자 점수: {data.get('buyer_reward')} / 판매자 점수: {data.get('seller_reward')}"
                                st.info(score_info)
                                status.update(label="평가 완료", state="complete")

                    #  Reflector 노드
                    elif node == "reflector":
                        reflections = data.get("reflections", [])
                        if reflections:
                            snapshot = st.session_state.graph.get_state(st.session_state.config)
                            state_values = snapshot.values

                            current_reflections = state_values.get("reflections", [])
                            max_retries = state_values.get("max_retries", 3)
                            current_count = len(current_reflections)+1

                            st.session_state.messages = []
                            warning_msg = (f"**[Self-Reflection]** ({current_count}/{max_retries}회)\n"
                                           "목표 달성에 실패했습니다. 전략을 수정하여 다시 협상합니다.")
                            st.session_state.messages.append({
                                    "role": "system",
                                    "content": warning_msg,
                                    "avatar": "🔄"
                            })
                            
                            reset_triggered = True
                            
                            st.toast("전략 수정 중... 대화를 재설정합니다.", icon="🔄")
            if reset_triggered:
                st.rerun()

            current_state = st.session_state.graph.get_state(st.session_state.config)
            
            if current_state.values.get("is_finished") and not current_state.next:
                 st.success("🎉 협상이 최종 종료되었습니다!")
                 st.balloons()