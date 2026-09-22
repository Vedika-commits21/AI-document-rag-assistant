\# AI Document RAG Assistant



An AI-powered document question-answering system that allows users to upload PDF documents and ask questions about their content.



The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded document and generate answers using a locally running Large Language Model.



\## Features



\- Upload PDF documents

\- Extract text from PDF files

\- Split documents into smaller chunks

\- Generate semantic embeddings

\- Store embeddings using FAISS

\- Retrieve relevant document chunks for a query

\- Generate answers using a local LLM

\- Streamlit-based user interface

\- Runs locally without requiring OpenAI API



\## Tech Stack



\- Python

\- Streamlit

\- PyMuPDF

\- Sentence Transformers

\- FAISS

\- Ollama

\- Llama 3.2 3B

\- NumPy



\## RAG Pipeline



```text

PDF Upload

&#x20;   ↓

Text Extraction

&#x20;   ↓

Text Chunking

&#x20;   ↓

Embedding Generation

&#x20;   ↓

FAISS Vector Store

&#x20;   ↓

Semantic Retrieval

&#x20;   ↓

Relevant Context

&#x20;   ↓

Local LLM (Llama 3.2 3B)

&#x20;   ↓

Generated Answer
Project Structure
AI-Document-RAG-Assistant/
│
├── app.py
├── document_processor.py
├── embedding_generator.py
├── rag_pipeline.py
├── retriever.py
├── vector_store.py
├── requirements.txt
├── .gitignore
└── README.md
How It Works
The user uploads a PDF document through the Streamlit interface.
PyMuPDF extracts text from the document.
The extracted text is divided into smaller chunks.
Sentence Transformers converts the chunks into numerical embeddings.
FAISS stores and searches these embeddings efficiently.
When the user asks a question, the system retrieves the most relevant document chunks.
The retrieved context is passed to the local Llama 3.2 3B model through Ollama.
The model generates an answer based on the retrieved document context.
Installation

Clone the repository:

git clone https://github.com/Vedika-commits21/ai-document-rag-assistant.git
cd ai-document-rag-assistant

Create and activate a virtual environment:

python -m venv venv

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Ollama Setup

Install Ollama and make sure the required model is available locally:

ollama pull llama3.2:3b

Make sure Ollama is running before starting the application.

Run the Application
streamlit run app.py

The Streamlit application will open in the browser.

Why RAG?

Traditional LLM applications may not have access to the user's private documents. RAG solves this by retrieving relevant information from the uploaded documents and providing it as context to the language model before generating an answer.

This helps the application produce responses grounded in the uploaded document rather than relying only on the model's pre-trained knowledge.

Future Improvements
Support for multiple document formats
Multi-document conversations
Improved hybrid retrieval
Conversation memory
Source citations for retrieved chunks
Deployment using cloud infrastructure
Authentication and user-specific document storage
Author

Vedika Pathak

B.Tech – Artificial Intelligence & Data Science

