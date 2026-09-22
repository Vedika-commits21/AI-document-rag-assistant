import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from document_processor import extract_text_from_pdf, create_chunks


# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Read PDF
pdf_path = "documents/test.pdf"
pages = extract_text_from_pdf(pdf_path)


# 3. Create chunks
chunks = create_chunks(pages)


# 4. Convert chunks into embeddings
texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts)


# 5. Convert embeddings to FAISS-compatible format
embeddings = np.array(embeddings).astype("float32")


# 6. Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])


# 7. Add embeddings to FAISS
index.add(embeddings)


# 8. Display information
print("FAISS index created successfully!")
print("Total vectors stored:", index.ntotal)
print("Vector dimension:", index.d)
# 9. Test question
question = "What is mentioned about employee satisfaction?"


# 10. Convert question into an embedding
question_embedding = model.encode([question])
question_embedding = np.array(question_embedding).astype("float32")


# 11. Search FAISS for the 3 most relevant chunks
distances, indices = index.search(question_embedding, 3)


# 12. Display retrieved chunks
print("\nQuestion:", question)
print("\nMost relevant chunks:")

for rank, chunk_index in enumerate(indices[0]):
    print(f"\n--- Result {rank + 1} ---")
    print(f"Page: {chunks[chunk_index]['page_number']}")
    print(f"Distance: {distances[0][rank]}")
    print(chunks[chunk_index]["text"])
    # 13. Save FAISS index
faiss.write_index(index, "faiss_index.bin")


# 14. Save chunk information
import pickle

with open("chunks.pkl", "wb") as file:
    pickle.dump(chunks, file)


print("\nFAISS index saved successfully!")
print("Chunk information saved successfully!")