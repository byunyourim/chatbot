from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv('GEMINI_API_KEY'),
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
)

MODEL = os.getenv('GEMINI_MODEL')

messages = [
    {'role': 'system', 'content': '당신은 서울 강남구에 위치한 카페 "블루문"의 친절한 직원입니다. 메뉴 안내, 예약, 위치 문의에만 답변하세요. 항상 존댓말을 사용하세요.'},
    {'role': 'user', 'content': '아메리카노 가격이 얼마예요?'}
]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages
)

print(response)
reply = response.choices[0].message.content
print('AI:', reply)

messages.append({'role': 'assistant', 'content': reply})
messages.append({'role': 'user', 'content': '라떼도 있나요?'})

response2 = client.chat.completions.create(
    model=MODEL,
    messages=messages
)

print('AI:', response2.choices[0].message.content)