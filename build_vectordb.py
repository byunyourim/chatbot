import json
import os
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from google import genai
from dotenv import load_dotenv

# pip install chromadb google-genai python-dotenv

load_dotenv()


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-001"):
        self._client = genai.Client(api_key=api_key)
        self._model = model_name

    def __call__(self, input: Documents) -> Embeddings:
        result = self._client.models.embed_content(
            model=self._model,
            contents=list(input),
        )
        return [list(e.values) for e in result.embeddings]

    @staticmethod
    def name() -> str:
        return "gemini_embedding"


embedding_fn = GeminiEmbeddingFunction(
    api_key=os.getenv('GEMINI_API_KEY'),
    model_name="gemini-embedding-001",
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