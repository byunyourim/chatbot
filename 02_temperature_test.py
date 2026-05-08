from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv('GEMINI_API_KEY'),
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
)

MODEL = os.getenv('GEMINI_MODEL')

question = '외로운 로봇의 하루를 세 문장으로 써줘'

resp_low = client.chat.completions.create(
    model=MODEL,
    messages=[{'role': 'user', 'content': question}],
    temperature=0.0
)

resp_high = client.chat.completions.create(
    model=MODEL,
    messages=[{'role': 'user', 'content': question}],
    temperature=0.3
)

print('=== temperature 0.0 (결정론적) ===')
print(resp_low.choices[0].message.content)

print('\n=== temperature 1.5 (창의적) ===')
print(resp_high.choices[0].message.content)