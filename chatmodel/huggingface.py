from dotenv import load_dotenv
load_dotenv()

import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
)
model = ChatHuggingFace(llm=llm)

try:
    response = model.invoke("What is the capital of France?")
    print(response)
except Exception as e:
    print("Error:", e)