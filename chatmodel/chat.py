from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "sk-..."

model = init_chat_model(
    model="gpt-5.4",
    temperature=0.5,
    max_tokens=2048,
    timeout=15,
    max_retries=3,
    streaming=False,)  

conversation =[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
    ] 

response = model.chat("What is the capital of France?")
print(response)