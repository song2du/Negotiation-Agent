import streamlit as st
import uuid
from core.graphs import build_graph
from langchain_core.messages import AIMessage
from core.scenarios import PRIORITIES

def render_priority_editor(role, key_prefix):
    """
    PRIORITIES 딕셔너리에 정의된 목표들을 가져와서
    사용자가 이름과 배점을 수정할 수 있는 입력 폼을 렌더링함.
    """
    defaults = PRIORITIES.get(role, {})
    
    updated_goals = {}
    total_score = 0
    
    for idx, (goal_name, score) in enumerate(defaults.items()):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_name = st.text_input(
                f"목표 {idx+1}", 
                value=goal_name, 
                key=f"{role}_{key_prefix}_name_{idx}",
                help="목표의 내용을 수정할 수 있습니다."
            )
            
        with col2:
            new_score = st.number_input(
                "배점", 
                min_value=0, 
                max_value=100, 
                value=score, 
                step=5,
                key=f"{role}_{key_prefix}_score_{idx}",
                help="이 목표의 중요도(점수)입니다."
            )
        
        if new_name: 
            updated_goals[new_name] = int(new_score)
            total_score += new_score

    if total_score != 100:
        st.caption(f"⚠️ 현재 총점: **{total_score}점** (연구 표준은 보통 100점 만점입니다)")
    else:
        st.caption(f"✅ 현재 총점: **100점** (완벽합니다)")
        
    return updated_goals

def render_setup_screen():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True) # 상단 여백
        st.title("HCI Lab Negotiation Agent")
        st.markdown("### 협상 AI 에이전트 실험 플랫폼")
        st.info("실험 설정을 완료하고 '협상 시작' 버튼을 눌러주세요.")
        
        with st.container(border=True):
            # 모드 선택
            mode = st.radio(
                "🧪 실험 모드 선택",
                [
                    "Baseline",
                    "CoT",
                    "IRP+SVI"
                ],
                index=0
            )
            # 선택된 모드 설명
            mode_descriptions = {
                "Baseline": "기본 에이전트",
                "CoT": "CoT, ICL (JSON 출력 강제, 중재자 피드백 제거, 우선순위 수정)",
                "IRP+SVI": "CoT, ICL, Few-shot, RAG Tools, 협상 전략 설명 강화"
            }
            st.caption(f"{mode_descriptions.get(mode, '')}")
            
            # 역할 선택
            role = st.selectbox("👤 사용자 역할", ["구매자", "판매자"])
            model_options = {
                "GPT-4o": "gpt-4o",
                "Claude 3 Sonnet": "anthropic/claude-3-sonnet-20240229" 
            }
            
            # 모델 선택
            selected_label = st.selectbox(
                "🧠 LLM 모델 선택",
                options=list(model_options.keys()),
                index=0
            )
            model_name = model_options[selected_label]


            st.markdown("---")

            st.markdown(f"#### 🎯 나 ({role})의 목표 설정")
            with st.expander("내 목표 상세 편집 (클릭)", expanded=False):
                user_goals_dict = render_priority_editor(role, key_prefix="user")

            # 상대방 목표 설정
            ai_role_name = "판매자" if role == "구매자" else "구매자"
            st.markdown(f"#### 🧑‍💻 상대방 ({ai_role_name})의 목표 설정")
            with st.expander("상대방 목표 상세 편집 (클릭)", expanded=False):
                st.info("상대방은 이 목표들을 달성하기 위해 전략을 수립합니다.")
                ai_goals_dict = render_priority_editor(ai_role_name, key_prefix="ai")

            st.markdown("---")
            
            # 시작 버튼
            if st.button("🚀 협상 시작하기", use_container_width=True, type="primary"):
                # 세션 초기화 및 그래프 로드
                if "Baseline" in mode:
                    st.session_state.mode = "baseline"
                elif "CoT" in mode:
                    st.session_state.mode = "cot_previous"
                elif "IRP+SVI" in mode:
                    st.session_state.mode = "cot_upgrade"
                st.session_state.user_role = role
                st.session_state.model_name = model_name
                st.session_state.config["configurable"]["thread_id"] = str(uuid.uuid4())
                st.session_state.messages = [] # 화면 표시용 메시지 초기화
                
                st.session_state.graph = build_graph(st.session_state.mode)
                
                # 초기 실행
                init_inputs = {
                    "user_role": role, 
                    "model": model_name, 
                    "messages": [],
                    "user_priority_inputs": user_goals_dict,
                    "ai_priority_inputs": ai_goals_dict,
                    "mode": st.session_state.mode
                }
                
                # Setup 단계 실행 
                with st.spinner("협상 준비 중..."):
                    try:
                        # stream 모드로 실행하여 첫 메시지를 가져옴
                        for event in st.session_state.graph.stream(init_inputs, st.session_state.config):
                            for node, data in event.items():
                                # 노드별 출력 처리 
                                if "messages" in data and data["messages"]:
                                    last_msg = data["messages"][-1]
                                    if isinstance(last_msg, AIMessage):
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": last_msg.content,
                                            "avatar": "🧑‍💻"
                                        })
                    except Exception as e:
                        st.error(f"초기화 중 오류 발생: {e}")
                        st.stop()

                st.session_state.is_started = True
                st.rerun()