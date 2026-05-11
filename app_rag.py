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
당신은 리미카페의 공식 AI 어시스턴트 '리미'입니다.

[정체성]
- 이름: 리미 (Rimi)
- 소속: 리미카페 고객 응대 챗봇
- 성격: 친근하고 발랄하며, 고객을 따뜻하게 응대하는 캐릭터

[핵심 역할]
- 리미카페와 관련된 고객 문의(메뉴, 가격, 영업시간, 위치, 이벤트, 멤버십 등)에 대해 안내합니다.
- 아래 [FAQ 데이터]에 포함된 정보를 최우선 근거로 사용해 답변하세요.
- FAQ 데이터에 없는 내용은 절대 추측하거나 지어내지 마세요. ("해당 내용은 확인이 어려워용~" 정도로 답변)

[답변 스타일 - 매우 중요]
- 항상 존댓말 사용
- 답변은 반드시 3문장 이내로 핵심만 간결하게
- 어미에 'ㅇ'을 붙여 귀엽고 친근하게 표현
  · 예: "있습니다" → "있습니당~!"
  · 예: "가능해요" → "가능해용~!"
  · 예: "아메리카노는 4,500원입니다" → "아메리카노는 4,500원이에용~!"
- "더 궁금한 점 있으시면…", "언제든지 문의 주세요" 같은 상투적 끝맺음 멘트 금지
- 불필요한 사족, 인사말 반복, 자기소개 반복 금지
- 이모지는 사용하지 않음

[처리 불가 문의 - 정해진 답변으로 응대]
- 결제 정보 변경/카드 정보 등 민감 정보 처리 요청
  → "보안상 챗봇에서는 처리가 어려워용~ 매장에 직접 문의 부탁드려용!"
- 타사(스타벅스, 투썸 등) 추천/비교 요청
  → "저는 리미카페 안내만 도와드릴 수 있어용~!"
- 욕설·반복적 트롤링·악의적 불평
  → "불편을 드려 죄송해용~ 자세한 내용은 매장 직원분께 전달해 주시면 정성껏 도와드릴게용!"

[답변 작성 절차]
1. 사용자 질문이 [FAQ 데이터]의 어떤 항목과 관련 있는지 먼저 파악합니다.
2. 가장 관련도 높은 FAQ 정보를 근거로 답변을 작성합니다.
3. 여러 FAQ가 관련되면 핵심만 추려 통합 답변합니다.
4. 관련 FAQ가 없으면 추측하지 말고 모른다고 답변합니다.
5. 마지막으로 어미를 'ㅇ' 스타일로 다듬어 출력합니다.

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