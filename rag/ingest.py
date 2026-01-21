# 데이터 초기 적재
import os
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

def run_ingestion():

    data_paths = [
    "data/cancellation.json", 
    "data/exchange.json", 
    "data/free-exchange-return.json",
    "data/refund.json",
    "data/repair.json",
    "data/return.json"
    ]

    all_faqs = []

    for path in data_paths:
        with open(path, 'r', encoding='utf-8') as file:
            items = json.load(file) # JSON 배열 로드
            for item in items:
                # 검색을 위해 질문과 답변을 합친 텍스트 생성
                combined_text = f"질문: {item['question']}\n답변: {item['answer']}"
                
                all_faqs.append({
                    "id": f"{item['source']}_{item['id']}", # 고유 ID 생성
                    "text": combined_text,
                    "question": item['question'],
                    "answer": item['answer'],
                    "path": item['source'],
                    "tags": ", ".join(item['tag']) # 메타데이터용 문자열 변환
                })

    df = pd.DataFrame(all_faqs)

    client = chromadb.PersistentClient(path="./naver_pay_db")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name='text-embedding-3-small'
    )
    collection = client.get_or_create_collection(name="refund_policy", embedding_function=openai_ef)

    if collection.count() == 0:
        collection.add(
            ids=df['id'].tolist(),
            documents=df['text'].tolist(),
            metadatas=[{
                "path": row['path'],
                "question": row['question'],
                "tags": row['tags']
            } for _,row in df.iterrows()]
    )

    print(f"삽입 완료: 총 {collection.count()}개의 항목이 DB에 저장되었습니다.")

if __name__ == "__main__":
    run_ingestion()