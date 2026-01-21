import streamlit as st
from graph import create_graph
import uuid
from langchain_core.messages import HumanMessage


st.set_page_config(page_title="HCI 협상 시뮬레이터", layout="wide")
st.title("🤝 협상 실험")

if "graph" not in st.session_state:
    st.session_state.graph = create_graph()
    st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    st.session_state.messages = []

# 사이드바 설정
with st.sidebar:
    role = st.radio("역할 선택", ["구매자", "판매자"])
    model = st.selectbox(
        "협상에 사용할 모델을 선택해 주세요.",
        ("gpt-4o",  "claude-3-5-sonnet-latest", "gpt-5.2", "claude-4-5-sonnet-latest")
    )
    if st.button("협상 시작/초기화"):
        st.session_state.config["configurable"]["thread_id"] = str(uuid.uuid4())
        st.session_state.messages = []
        # 초기 실행 (AI 선공)
        init_state = {"user_role": role, "messages": [], "model":model}
        for event in st.session_state.graph.stream(init_state, st.session_state.config):
            for node, data in event.items():
                if "messages" in data and data["messages"]:
                    msg_content = data["messages"][-1].content
                    # AI 노드나 중재자 노드의 메시지만 UI에 추가
                    if node == "ai_agent":
                        d_name, d_avatar = "AI 에이전트", "🤖"
                    elif node == "mediator":
                        d_name, d_avatar = "시스템 중재자", "⚖️"
                    else: continue
                    
                    st.session_state.messages.append({
                        "role": d_name, 
                        "content": msg_content, 
                        "avatar": d_avatar
                    })
                else:
                    # setup 노드처럼 메시지가 없는 경우 로그 출력 (디버깅용)
                    print(f"Node {node} finished without new messages.")
        st.rerun()

# 채팅창 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 그래프 업데이트 및 재개
    st.session_state.graph.update_state(
        st.session_state.config, {"messages": [HumanMessage(content=prompt)]})
    
    with st.spinner("에이전트와 중재자가 대화를 분석 중입니다..."):
        for event in st.session_state.graph.stream(None, st.session_state.config):
            for node, data in event.items():

                if node in ["ai_agent", "mediator"] and "messages" in data and data["messages"]:
                    res_content = data["messages"][-1].content

                    if node == "ai_agent":
                        display_name, avatar = "AI 에이전트", "🤖"
                    else:
                        display_name, avatar = "시스템 중재자", "⚖️"

                    st.session_state.messages.append({
                        "role": display_name, 
                        "content": res_content, 
                        "avatar": avatar
                    })
                    with st.chat_message(display_name, avatar=avatar):
                        st.markdown(res_content)
        current_state = st.session_state.graph.get_state(st.session_state.config)
        if current_state.values.get("is_finished"):
            st.success("🎉 협상이 종료되었습니다! 결과가 저장되었습니다.")
