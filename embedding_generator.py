from sentence_transformers import SentenceTransformer
from document_processor import extract_text_from_pdf, create_chunks


# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. PDF se text extract karo
pdf_path = "documents/test.pdf"
pages = extract_text_from_pdf(pdf_path)


# 3. Text ko chunks mein divide karo
chunks = create_chunks(pages)


# 4. Har chunk ka embedding generate karo
texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts)


# 5. Basic information print karo
print("Embeddings created successfully!")
print("Total chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)