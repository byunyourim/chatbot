import json
import os
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

# ── Gemini 임베딩 모델 (무료, 다운로드 불필요) ────────────────────────
embedding_fn = GoogleGenerativeAiEmbeddingFunction(
    api_key=os.getenv('GEMINI_API_KEY'),
    model_name="models/gemini-embedding-001"
)

# ── faq_data.json 로드 ────────────────────────────────────────────────
with open('faq_data.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

# ── ChromaDB 로컬 클라이언트 생성 ─────────────────────────────────────
client = chromadb.PersistentClient(path='chroma_db')

# 기존 컬렉션 삭제 후 재생성
try:
    client.delete_collection('faq_collection')
    print("기존 컬렉션 삭제 완료")
except:
    pass

collection = client.create_collection(
    name='faq_collection',
    embedding_function=embedding_fn
)

# ── FAQ 전체를 documents 리스트로 변환 → ChromaDB가 자동 임베딩 ───────
docs, ids, metas = [], [], []

for category, items in faq_data.items():
    for i, item in enumerate(items):
        docs.append(item['질문'] + ' ' + item['답변'])
        ids.append(f'{category}_{i}')
        metas.append({
            'category': category,
            'question': item['질문'],
            'answer':   item['답변']
        })

collection.add(documents=docs, ids=ids, metadatas=metas)

print(f'완료 : {len(docs)}개 FAQ 임베딩 저장 → chroma_db/ 폴더')