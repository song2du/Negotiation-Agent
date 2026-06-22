import streamlit as st
from core.scenarios import PRIORITIES, SCENARIOS


def render_explain_screen():
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("역할 및 시나리오 안내")
        st.markdown("협상을 시작하기 전에 아래 내용을 꼼꼼히 읽어주세요.")

        user_role = st.session_state.get("user_role", "")
        ai_role = "판매자" if user_role == "구매자" else "구매자"

        # 역할 안내
        with st.container(border=True):
            st.markdown(f"#### 👤 내 역할: **{user_role}**")
            st.markdown(f"상대방(AI)의 역할: **{ai_role}**")

        st.markdown("<br>", unsafe_allow_html=True)

        # 시나리오
        with st.container(border=True):
            st.markdown("#### 📖 시나리오")
            scenario_text = SCENARIOS.get(user_role, "시나리오를 찾을 수 없습니다.")
            st.write(scenario_text)

        st.markdown("<br>", unsafe_allow_html=True)

        # 우선순위
        with st.container(border=True):
            st.markdown("#### 🎯 내 우선순위 (목표 및 배점)")
            st.caption("협상에서 아래 목표를 달성할수록 높은 점수를 받습니다.")

            user_priorities = st.session_state.get("user_priority_inputs") or PRIORITIES.get(user_role, {})
            for item, score in user_priorities.items():
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.write(f"• {item}")
                with col_b:
                    st.markdown(f"**{score}점**")

        st.markdown("<br>", unsafe_allow_html=True)

        st.warning(
            f"상대방({ai_role})도 자신만의 우선순위에 따라 협상을 진행합니다. "
            "서로의 목표가 다를 수 있으며, 자신의 점수를 최대한 지키며 합의에 도달하는 것이 목표입니다."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.error(
            "⚠️ **협상 진행 안내**  \n"
            "최대 턴 수는 **20턴**입니다.  \n"
            "협상이 타결되거나 더 이상 진전이 없다고 판단되면 **협상 종료 버튼**을 눌러주세요."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("✅ 내용을 숙지했습니다. 협상 시작!", use_container_width=True, type="primary"):
            st.session_state.screen = "chat"
            st.rerun()
