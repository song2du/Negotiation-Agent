import streamlit as st
import uuid

from ui.setup import render_setup_screen
from ui.chat import render_chat_screen

st.set_page_config(
    page_title="HCI Negotiation Agent", 
    layout="wide", 
    page_icon="🤝"
)

def init_session_state():
    """
    세션 스테이트(전역 변수) 초기화 함수.
    앱이 처음 실행될 때 필요한 변수들이 없으면 기본값을 생성합니다.
    """
    defaults = {
        "is_started": False,           # 협상 시작 여부
        "messages": [],                # 채팅 기록
        "graph": None,                 # LangGraph 객체
        "config": {"configurable": {"thread_id": str(uuid.uuid4())}}, # 그래프 설정
        "negotiation_status": "진행 중" # 현재 상태
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    # 세션 초기화 실행
    init_session_state()

    # 화면 라우팅
    # is_started 값에 따라 '설정 화면'과 '채팅 화면' 중 하나만 보여줍니다.
    if not st.session_state.is_started:
        render_setup_screen()
    else:
        render_chat_screen()

if __name__ == "__main__":
    main()