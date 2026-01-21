import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name='text-embedding-3-small'
)

client = chromadb.PersistentClient(path="./naver_pay_db")
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