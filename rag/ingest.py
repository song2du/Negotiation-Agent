# 데이터 초기 적재
import json
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", "data", "law.json"))
DB_PATH = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", "law_db"))


def _normalize_metadata_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def run_ingestion():
    data_paths = [DATA_PATH]

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
                # law.json 등 규정 파일은 `document` 필드에 내용이 들어있어야 합니다.
                # `question`/`answer` 폴백을 제거하고, 문서가 없으면 항목을 건너뜁니다.
                item_id = item.get("id", "(no id)")
                print(f"경고: {source_label}의 항목 {item_id}에 'document'가 없습니다. 건너뜁니다.")
                continue

            item_id = item.get("id")
            if item_id is None:
                item_id = len(ids) + 1

            metadata = {"path": source_label}
            source = item.get("source")
            if source:
                metadata["source"] = source

            raw_metadata = item.get("metadata", {})
            if isinstance(raw_metadata, dict):
                for key, value in raw_metadata.items():
                    normalized_value = _normalize_metadata_value(value)
                    if normalized_value is not None:
                        metadata[key] = normalized_value

            ids.append(f"{source_label}_{item_id}")
            documents.append(document_text)
            metadatas.append(metadata)

    # Ensure DB directory exists for ChromaDB persistent storage
    os.makedirs(DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    openai_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-small"
    )
    collection_name = "refund_policy"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef,
    )

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    print(f"삽입 완료: 총 {collection.count()}개의 항목이 DB에 저장되었습니다.")


if __name__ == "__main__":
    run_ingestion()