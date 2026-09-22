import urllib.request
import json

from retriever import retrieve_relevant_chunks


def generate_answer(question, top_k=3):
    # 1. Retrieve relevant chunks from FAISS
    results = retrieve_relevant_chunks(question, top_k=top_k)

    # 2. Create context from retrieved chunks
    context_parts = []

    for result in results:
        context_parts.append(
            f"[Page {result['page_number']}]\n{result['text']}"
        )

    context = "\n\n".join(context_parts)

    # 3. Create a strict prompt for the LLM
    prompt = f"""
You are a document question-answering assistant.

Answer the user's question ONLY using the information provided
in the document context below.

Rules:
- Do not use outside knowledge.
- Do not make up information.
- If the answer is not present in the context, clearly say:
  "The answer is not available in the provided document."
- Keep the answer concise and clear.
- Mention the relevant page number(s) when possible.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    # 4. Send the context + question to the local LLM
    url = "http://localhost:11434/api/generate"

    data = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    # 5. Get LLM response
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return {
        "answer": result["response"],
        "sources": results
    }


# Test the complete RAG pipeline
question = "What is mentioned about employee satisfaction?"

result = generate_answer(question)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(result["answer"])

print("\nSources:")
for source in result["sources"]:
    print(
        f"- Page {source['page_number']} "
        f"(distance: {source['distance']:.4f})"
    )