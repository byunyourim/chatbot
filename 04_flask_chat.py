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
    api_key=os.getenv('GEMINI_API_KEY'),       # .env에서 API 키 로드
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/'  # Gemini 엔드포인트
)


# 루트 경로 → 채팅 UI 페이지 반환
@app.route('/')
def index():
    return render_template('index.html')

# /chat 경로 → 사용자 메시지를 받아 AI 응답 반환
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '')  # 요청 JSON에서 메시지 추출

    # Gemini API 호출
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'user', 'content': user_msg}],
        temperature=1.5
    )

    reply = response.choices[0].message.content

    # ensure_ascii=False → 한글이 깨지지 않도록 UTF-8 그대로 반환
    return app.response_class(
        response=json.dumps({'reply': reply}, ensure_ascii=False),
        mimetype='application/json'
    )

if __name__ == '__main__':
    app.run(debug=True)  # debug=True → 코드 수정 시 서버 자동 재시작


# curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d "{\"message\": \"안녕하세요\"}"