import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Load the saved FAISS index
index = faiss.read_index("faiss_index.bin")


# 3. Load the saved chunks
with open("chunks.pkl", "rb") as file:
    chunks = pickle.load(file)


def retrieve_relevant_chunks(question, top_k=3):
    """
    Convert the user's question into an embedding
    and retrieve the most relevant document chunks.
    """

    # Convert question into embedding
    question_embedding = model.encode([question])

    # Convert to FAISS-compatible format
    question_embedding = np.array(question_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(question_embedding, top_k)

    results = []

    for rank, chunk_index in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "page_number": chunks[chunk_index]["page_number"],
            "text": chunks[chunk_index]["text"],
            "distance": float(distances[0][rank])
        })

    return results


# Test retrieval
question = "What is mentioned about employee satisfaction?"

results = retrieve_relevant_chunks(question)


print("Question:", question)
print("\nRetrieved chunks:")

for result in results:
    print(f"\n--- Result {result['rank']} ---")
    print(f"Page: {result['page_number']}")
    print(f"Distance: {result['distance']}")
    print(result["text"])