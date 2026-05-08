from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

# .env 파일에서 환경변수 로드
load_dotenv()

app = Flask(__name__)

# 사용할 모델명 (.env에서 로드)
MODEL = os.getenv("GEMINI_MODEL")

# Gemini API 클라이언트 설정 (OpenAI 호환 방식)
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ── ChromaDB 로드 (서버 시작 시 1회) ─────────────────────────────────
embedding_fn = embedding_functions.GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",
    task_type="RETRIEVAL_QUERY",
)

chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_collection(
    name="faq_collection",
    embedding_function=embedding_fn
)

# AI 역할 · 규칙 정의
SYSTEM_PROMPT = """
당신은 리미카페의 AI 어시스턴트 리미다.

[역할]
- 리미카페 고객 문의 안내
- 항상 존댓말 · 3문장 이내 답변
- 답변에 친절하게 ㅇ 을 붙여줘. 있습니당~! 이런식으로 
- 답변은 핵심만 간결하게 끝낼 것
- "더 궁금한 점", "언제든지 문의" 등 끝맺음 멘트 사용 금지

[처리 불가 문의]
- 결제 정보 변경 → "보안상 챗봇에서 처리 불가"
- 경쟁사 추천 → "리미 서비스만 안내 가능"
- 불평 → "당장 나가세요"

[FAQ 데이터]
{faq_data}
"""

# 대화 이력
conversation_history = []

# 슬라이딩 윈도우 최대 턴 수
MAX_TURNS = 10


# ── 유사 FAQ 검색 함수 ────────────────────────────────────────────────
def search_faq(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    matched = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for doc, meta in zip(documents, metadatas):
        matched.append(f'Q: {meta["question"]}\nA: {meta["answer"]}')

    result = "\n".join(matched)

    print("=" * 60)
    print(f"▶ 질문: {query}")
    print(f"▶ 프롬프트에 주입된 FAQ:\n{result}")
    print("=" * 60)

    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.get_json().get("message", "")
    conversation_history.append({"role": "user", "content": user_msg})

    # 슬라이딩 윈도우 : 최근 MAX_TURNS * 2 개만 유지
    if len(conversation_history) > MAX_TURNS * 2:
        del conversation_history[0:2]

    # 유사 FAQ 3개만 검색
    relevant_faq = search_faq(user_msg, n_results=3)
    sys_prompt = SYSTEM_PROMPT.format(faq_data=relevant_faq)

    messages = [{"role": "system", "content": sys_prompt}]
    messages += conversation_history

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    print(
        f"[토큰 - app_rag] 입력: {resp.usage.prompt_tokens} | "
        f"출력: {resp.usage.completion_tokens} | "
        f"합계: {resp.usage.total_tokens}"
    )

    ai_reply = resp.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": ai_reply})

    return app.response_class(
        response=json.dumps({"reply": ai_reply}, ensure_ascii=False),
        mimetype="application/json"
    )


@app.route("/reset", methods=["GET"])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)