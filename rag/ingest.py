# 데이터 초기 적재
import json
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", "coupang_db"))


def _normalize_metadata_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def run_ingestion():
    data_paths = ["data/coupang_processed.json"]

    ids = []
    documents = []
    metadatas = []

    for path in data_paths:
        with open(path, "r", encoding="utf-8") as file:
            items = json.load(file)

        source_label = os.path.basename(path)

        for item in items:
            document_text = item.get("document")
            if not document_text:
                question = item.get("question")
                answer = item.get("answer")
                if question and answer:
                    document_text = f"[{item.get('category', '규정')}] {question}: {answer}"

            if not document_text:
                continue

            item_id = item.get("id")
            if item_id is None:
                item_id = len(ids) + 1

            metadata = {"path": source_label}
            raw_metadata = item.get("metadata", {})
            if isinstance(raw_metadata, dict):
                for key, value in raw_metadata.items():
                    normalized_value = _normalize_metadata_value(value)
                    if normalized_value is not None:
                        metadata[key] = normalized_value

            ids.append(f"{source_label}_{item_id}")
            documents.append(document_text)
            metadatas.append(metadata)

    client = chromadb.PersistentClient(path=DB_PATH)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    collection = client.get_or_create_collection(
        name="refund_policy",
        embedding_function=openai_ef,
    )

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    print(f"삽입 완료: 총 {collection.count()}개의 항목이 DB에 저장되었습니다.")


if __name__ == "__main__":
    run_ingestion()