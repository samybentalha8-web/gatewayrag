import os
import requests
from dotenv import load_dotenv

load_dotenv()

response = requests.get(
    "https://factchat-cloud.mindlogic.ai/v1/gateway/models/",
    headers={"Authorization": f"Bearer {os.getenv('HAI_API_KEY')}"},
)

print(response.status_code)
print(response.json())
