from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# .env 파일에서 환경변수 로드
load_dotenv()

app = Flask(__name__)

# 사용할 모델명 (.env에서 로드)
MODEL = os.getenv('GEMINI_MODEL')

# Gemini API 클라이언트 설정 (OpenAI 호환 방식)
client = OpenAI(
    api_key=os.getenv('GEMINI_API_KEY'),
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
)

# faq_data.json 파일에서 FAQ 데이터 로드
with open('faq_data.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

# AI 역할 · 규칙 정의 (5원칙 적용)
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
- 경쟁사 추천 → "WorkOn 서비스만 안내 가능"
- 불평 → "당장 나가세요"

[FAQ 데이터]
{faq_data}
"""

# 대화 이력
conversation_history = []

# 슬라이딩 윈도우 최대 턴 수
MAX_TURNS = 10


def format_faq(faq_data):
    result = []
    for category, items in faq_data.items():
        result.append(f'[{category}]')
        for item in items:
            result.append(f'Q: {item["질문"]}')
            result.append(f'A: {item["답변"]}')
    return '\n'.join(result)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.get_json().get('message', '')
    conversation_history.append({'role': 'user', 'content': user_msg})

    # 슬라이딩 윈도우 : 최근 MAX_TURNS * 2 개만 유지
    if len(conversation_history) > MAX_TURNS * 2:
        del conversation_history[0:2]

    faq_text = format_faq(faq_data)
    sys_prompt = SYSTEM_PROMPT.format(faq_data=faq_text)
    messages = [{'role': 'system', 'content': sys_prompt}]
    messages += conversation_history

    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=1.5   # 0.0 ~ 2.0 · 낮을수록 일관된 답변
    )
    print(f"[토큰 - app_final] 입력: {resp.usage.prompt_tokens} | 출력: {resp.usage.completion_tokens} | 합계: {resp.usage.total_tokens}")
    ai_reply = resp.choices[0].message.content
    conversation_history.append({'role': 'assistant', 'content': ai_reply})

    return app.response_class(
        response=json.dumps({'reply': ai_reply}, ensure_ascii=False),
        mimetype='application/json'
    )


@app.route('/reset', methods=['GET'])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True)



    