import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("Key loaded:", api_key[:10], "...")  # debug check

client = genai.Client(api_key=api_key)

models = client.models.list()

for m in models:
    print(m.name)