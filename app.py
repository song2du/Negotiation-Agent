import streamlit as st
import uuid

from ui.guide import render_guide_screen
from ui.setup import render_setup_screen
from ui.explain import render_explain_screen
from ui.chat import render_chat_screen

st.set_page_config(
    page_title="HCI Negotiation Agent",
    layout="wide",
    page_icon="🤝"
)

def init_session_state():
    defaults = {
        "screen": "guide",             # "guide" | "setup" | "explain" | "chat"
        "completed_modes": [],         # 완료된 모드 목록
        "messages": [],                # 채팅 기록 (UI용)
        "graph": None,                 # LangGraph 객체
        "config": {"configurable": {"thread_id": str(uuid.uuid4())}},
        "negotiation_status": "진행 중",
        "form_step": None,
        "human_evaluation": {},
        "survey_results": {},
        "show_end_success": False,
        "user_priority_inputs": {},
        "ai_priority_inputs": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    init_session_state()

    screen = st.session_state.screen

    if screen == "guide":
        render_guide_screen()
    elif screen == "setup":
        render_setup_screen()
    elif screen == "explain":
        render_explain_screen()
    else:
        render_chat_screen()

if __name__ == "__main__":
    main()
