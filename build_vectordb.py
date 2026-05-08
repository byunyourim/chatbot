import json
import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
'''
pip install chromadb
pip install google-generativeai
pip install google-genai
'''


load_dotenv()

# ── Gemini 임베딩 모델 ───────────────────────────────────────────────
# .env 파일에 GEMINI_API_KEY=... 필요
embedding_fn = embedding_functions.GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
)

# ── faq_data.json 로드 ────────────────────────────────────────────────
with open("faq_data.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# ── ChromaDB 로컬 클라이언트 생성 ─────────────────────────────────────
client = chromadb.PersistentClient(path="chroma_db")

# 기존 컬렉션 삭제 후 재생성
try:
    client.delete_collection("faq_collection")
    print("기존 컬렉션 삭제 완료")
except Exception:
    pass

collection = client.create_collection(
    name="faq_collection",
    embedding_function=embedding_fn
)

# ── FAQ 전체를 documents 리스트로 변환 → ChromaDB가 자동 임베딩 ───────
docs, ids, metas = [], [], []

for category, items in faq_data.items():
    for i, item in enumerate(items):
        question = item["질문"]
        answer = item["답변"]

        docs.append(f"질문: {question}\n답변: {answer}")
        ids.append(f"{category}_{i}")
        metas.append({
            "category": category,
            "question": question,
            "answer": answer
        })

collection.add(
    documents=docs,
    ids=ids,
    metadatas=metas
)

print(f"완료 : {len(docs)}개 FAQ 임베딩 저장 → chroma_db/ 폴더")