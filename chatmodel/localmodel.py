from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
import torch

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float32,
    device_map="auto",
    max_new_tokens=100,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)

response = llm.invoke(
    "what is Data Science? "
)

print(response.content)