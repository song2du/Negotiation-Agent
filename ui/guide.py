import streamlit as st

EXPERIMENT_ORDER = [
    ("Pure LLM", "pure_llm", "기본 LLM 에이전트와 협상합니다."),
    ("LLM + RAG", "llm_rag", "정책 데이터베이스가 추가된 에이전트와 협상합니다."),
    ("LLM + RAG + Nego Strategy", "llm_rag_nego_strategy", "협상 전략이 탑재된 에이전트와 협상합니다."),
]


def render_guide_screen():
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("HCI Lab 협상 실험")
        st.caption("AI 협상 에이전트와의 상호작용 연구에 참여해 주셔서 감사합니다.")

        with st.container(border=True):
            st.markdown("#### 📋 실험 안내")
            st.markdown(
                "이 실험에서는 **세 가지 조건**의 AI 협상 에이전트와 순서대로 협상을 진행합니다.  \n"
                "각 실험이 끝나면 이 화면으로 돌아와 다음 조건을 진행해 주세요."
            )
            st.markdown("---")
            st.markdown("**진행 순서**")

            completed = st.session_state.get("completed_modes", [])
            for i, (label, key, desc) in enumerate(EXPERIMENT_ORDER, 1):
                if key in completed:
                    st.success(f"**{i}. {label}** — {desc}  ✅ 완료")
                else:
                    st.info(f"**{i}. {label}** — {desc}")

        st.markdown("<br>", unsafe_allow_html=True)

        completed = st.session_state.get("completed_modes", [])
        all_keys = [key for _, key, _ in EXPERIMENT_ORDER]

        if all(k in completed for k in all_keys):
            st.success("🎉 모든 실험이 완료되었습니다! 참여해 주셔서 감사합니다.")
            return

        with st.container(border=True):
            st.markdown("#### 📌 실험 진행 방법")
            st.markdown(
                "1. **확인** 버튼을 클릭하면 실험 설정 화면으로 이동합니다.\n"
                "2. 실험 모드와 역할을 선택하고 **협상 시작** 버튼을 누르세요.\n"
                "3. 시나리오와 역할 안내를 확인한 후 협상을 진행하세요.\n"
                "4. 협상 종료 후 평가와 설문을 완료하면 이 화면으로 돌아옵니다.\n"
                "5. 위 순서(① → ② → ③)대로 모든 조건을 반복합니다."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("확인 — 실험 설정으로 이동", use_container_width=True, type="primary"):
            st.session_state.screen = "setup"
            st.rerun()
