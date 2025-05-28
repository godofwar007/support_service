from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)


def generate_response(text):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": text}],
        model="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.9,
        n=1,
        timeout=15
    )

    if response and response.choices:
        return response.choices[0].message.content
    else:
        return 'ничего нет('
