import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_FILE_DIR, "..", "law_db")
DB_PATH = os.path.abspath(DB_PATH)

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-small"
)


def _get_collection():
    """ingest 후 컬렉션이 재생성돼도 항상 최신 참조를 반환."""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name="refund_policy", embedding_function=_embedding_fn)


@tool
def policy_search_tool(query: str):
    """
    환불/반품/교환 규정 검색 툴.

    [중요: 쿼리(query) 작성 규칙]
    1. '토트넘', '유니폼'과 같은 특정 상품명이나 구체적인 상황 설명은 쿼리에 절대 포함하지 마세요. (검색 실패의 주 원인이 됩니다)
    2. '환불 규정', '교환 조건', '단순 변심 반품', '배송비 부담', '파손 보상' 등 일반적이고 포괄적인 정책 키워드 위주로 검색하세요.
    3. '리뷰 철회 정책' 같은 RAG 문서에 없을 법한 구체적 협상 조건 대신, 규정의 핵심 키워드인 '반품 기준' 등으로 검색하세요.
    """
    collection = _get_collection()
    results = collection.query(query_texts=[f"query: {query}"], n_results=3)
    
    retrieved_docs = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        source = meta.get('path') or meta.get('source') or meta.get('category') or 'unknown'
        doc_with_meta = f"--- [출처: {source}] ---\n{doc}"
        retrieved_docs.append(doc_with_meta)
    
    return "\n\n".join(retrieved_docs)