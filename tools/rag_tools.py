import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

def get_openai_api_key():
    """온라인(Secrets)과 로컬(.env) 환경을 모두 지원하는 키 로더"""
    # 1. Streamlit Secrets 확인 (온라인 배포 환경 우선)
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        # st.secrets가 활성화되지 않은 환경(일반 파이썬 스크립트 실행 등)일 경우 pass
        pass

    # 2. OS 환경 변수 확인 (로컬 .env 또는 시스템 설정)
    return os.getenv("OPENAI_API_KEY")

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_FILE_DIR, "..", "naver_pay_db")
DB_PATH = os.path.abspath(DB_PATH)

api_key = get_openai_api_key()

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY를 찾을 수 없습니다. "
        "로컬이면 .env 파일을, 온라인이면 Streamlit Secrets 설정을 확인하세요."
    )

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name='text-embedding-3-small'
)


client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="refund_policy", embedding_function=openai_ef)

@tool
def policy_search_tool(query: str):
    """
    네이버페이 규정 검색
    대화 기록을 입력하면, 내부적으로 최적의 검색어를 생성하여 규정 찾아옴
    """
    results = collection.query(query_texts=[query], n_results=3)
    
    retrieved_docs = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        doc_with_meta = f"--- [출처: {meta['path']}] ---\n{doc}\n태그: {meta['tags']}"
        retrieved_docs.append(doc_with_meta)
    
    return "\n\n".join(retrieved_docs)