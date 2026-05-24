from langchain_huggingface import HuggingFaceEmbeddings

print("Generating embeddings for 'Hello World'...")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
print("Embedding model loaded successfully.")

#coverting text to vector
vector = embeddings.embed_query("WHY U EXIST IN THIS FUCKING WORLD")

print("📏 Vector Length:", len(vector))

print("\n🔹 First 5 Values:\n")

print(vector[:5])