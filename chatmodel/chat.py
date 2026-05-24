from dotenv import load_dotenv
load_dotenv()
import time

from langchain.chat_models import init_chat_model

try:

    # Initialize model
    model = init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        temperature=1,      # lower = more focused
        max_tokens=1500,       # limits response size
        timeout=10,
        max_retries=3,
        streaming=True
    )

    # Better prompt structure
    messages = [
        (
            "system",
            "You are a concise AI assistant. "
            "Keep answers short, clear, and useful."
        ),
        (
            "human",
            "Write  a poem on AI in the style of Shakespeare."
        )
    ]

    print("✅ API Connected Successfully\n")
    print("🤖 AI Response:\n")
    
    #Stat
    start_time = time.time()

    # STREAMING RESPONSE
    for chunk in model.stream(messages):

        # Print tokens live
        if chunk.content:
            print(chunk.content, end="", flush=True)
    #End
    end_time = time.time()

    #Latency
    latency = end_time - start_time
    print(f"\n\n⏱️ Response Time: {latency:.2f} seconds")

    print("\n")

except Exception as e:

    print("\n❌ API Connection Failed")
    print("Error:", e)