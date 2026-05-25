from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic-AI"

# =========================
# INITIALIZE MODEL
# =========================

model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.3,
    max_tokens=2000,
    timeout=20,
    max_retries=3,
    streaming=False
)

# =========================
# CREATE PROMPT TEMPLATE
# =========================

code_review_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",#for large prompt we use """ jsdvkj"""
            """
You are a senior software engineer and expert code reviewer.

Your responsibilities:
- Analyze the provided code carefully.
- Detect syntax errors.
- Detect logical bugs.
- Detect runtime issues.
- Detect bad coding practices.
- Suggest performance improvements.
- Explain WHY the issue happens.
- Provide the exact fix.
- Suggest better coding approaches.

Rules:
- Be precise and technical.
- Explain clearly for beginners.
- Do not give vague answers.
- If no issues are found, say:
  'No major issues detected.'

Return output in this format:

## Bug Found
- Explain the issue

## Why It Happens
- Technical explanation

## Fix
- Step-by-step fix

## Corrected Code
- Provide corrected code

## Improvements
- Optional improvements
"""),

    (
            "human",
            """
Review this code carefully and identify all possible issues:
{code}
"""
        )
    ]
)

# =========================
# CREATE CHAIN
# =========================

chain = code_review_prompt | model

# =========================
# USER CODE INPUT
# =========================

test_code = input("Enter code to review:\n")

# =========================
# INVOKE CHAIN
# =========================

response = chain.invoke(
    {
        "code": test_code
    }
)

# =========================
# PRINT RESPONSE
# =========================

print("\n🤖 AI Code Review:\n")

print(response.content)