import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HAI_API_KEY"),
    base_url="https://factchat-cloud.mindlogic.ai/v1/gateway",
)

response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[
        {"role": "user", "content": "In one sentence, explain what an LLM gateway is."}
    ],
)

print(response.choices[0].message.content)
