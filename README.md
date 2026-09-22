# 🤖 AI Document RAG Assistant

### 📄 Ask Questions. Retrieve Knowledge. Generate Answers.

An AI-powered **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and interact with them through natural-language questions.

The system combines **semantic search, vector retrieval, and a locally running Large Language Model** to generate answers grounded in the uploaded document.

---

## ✨ What is this project?

Traditional LLM applications rely mainly on the knowledge stored inside the model.

This project follows a different approach:

> **Retrieve relevant information from the document first → provide it as context → generate the answer using an LLM.**

This makes the application useful for interacting with documents such as:

- 📚 Study material
- 📑 Technical documentation
- 📄 Reports
- 📝 Notes
- 📖 Research documents
- 💼 Business documents

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 📤 PDF Upload | Upload documents directly through the Streamlit interface |
| 📄 Text Extraction | Extract text from PDF documents using PyMuPDF |
| ✂️ Text Chunking | Break large documents into smaller searchable chunks |
| 🧠 Embeddings | Convert text into semantic vector representations |
| 🔎 Semantic Retrieval | Retrieve the most relevant document sections |
| 🗃️ FAISS Vector Store | Efficient similarity search over document embeddings |
| 🤖 Local LLM | Generate answers using Llama 3.2 3B |
| 🔒 Local Processing | Document processing and LLM inference run locally |
| 🖥️ Streamlit UI | Simple interactive interface for document Q&A |

---

## 🧠 RAG Architecture

```text
                  ┌──────────────────┐
                  │    PDF Upload    │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Text Extraction│
                  │    (PyMuPDF)     │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Text Chunking   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │    Embeddings    │
                  │ Sentence         │
                  │ Transformers     │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  FAISS Vector DB │
                  └────────┬─────────┘
                           │
                    User Question
                           │
                           ▼
                  ┌──────────────────┐
                  │ Semantic Search  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Relevant Context │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Llama 3.2 3B    │
                  │     via Ollama   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Generated Answer│
                  └──────────────────┘
⚙️ Technology Stack
🐍 Backend & Processing
Python
PyMuPDF
NumPy
🧠 AI & Machine Learning
Sentence Transformers
all-MiniLM-L6-v2
FAISS
Ollama
Llama 3.2 3B
🖥️ Interface
Streamlit
🛠️ Development
Git
GitHub
VS Code
🔄 How the System Works
1️⃣ Document Ingestion

The user uploads a PDF through the Streamlit application.

2️⃣ Text Extraction

PyMuPDF extracts readable text from the uploaded document.

3️⃣ Chunking

The extracted text is divided into smaller chunks so that relevant sections can be retrieved efficiently.

4️⃣ Embedding Generation

Each chunk is converted into a numerical vector using:

all-MiniLM-L6-v2

These vectors represent the semantic meaning of the text.

5️⃣ Vector Storage

The generated embeddings are stored and searched using:

FAISS
6️⃣ User Query

The user asks a natural-language question about the uploaded document.

7️⃣ Semantic Retrieval

The query is converted into an embedding and compared with the document vectors.

The most relevant chunks are retrieved.

8️⃣ Context + LLM

The retrieved document context is provided to:

Llama 3.2 3B

running locally through:

Ollama
9️⃣ Answer Generation

The LLM generates an answer using the retrieved document context.

📁 Project Structure
AI-Document-RAG-Assistant/
│
├── 📄 app.py
├── 📄 document_processor.py
├── 📄 embedding_generator.py
├── 📄 rag_pipeline.py
├── 📄 retriever.py
├── 📄 vector_store.py
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 .gitignore
│
└── 📁 documents/

Generated files, virtual environments, cache files and local documents are excluded from the Git repository using .gitignore.

🛠️ Installation & Setup
1. Clone the repository
git clone https://github.com/Vedika-commits21/ai-document-rag-assistant.git
cd ai-document-rag-assistant
2. Create a virtual environment
python -m venv venv
3. Activate the environment
Windows
venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
🤖 Ollama Setup

Install Ollama and pull the required model:

ollama pull llama3.2:3b

Make sure Ollama is running before launching the application.

▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

💡 Why RAG?

RAG combines two important capabilities:

Information Retrieval
        +
Large Language Models
        ↓
Context-Aware Answers

Instead of asking the LLM to answer entirely from its pre-trained knowledge, the system first retrieves relevant information from the uploaded document.

This approach helps the generated response stay grounded in the available document context.

🔐 Local & Privacy-Oriented Architecture

The project is designed to work with a local LLM through Ollama rather than requiring an external OpenAI API.

User Document
      ↓
Local Processing
      ↓
Local Embeddings
      ↓
Local FAISS Retrieval
      ↓
Local LLM
      ↓
Answer

This makes the project suitable for experimenting with document-based AI workflows while keeping the core processing on the user's machine.

🔮 Future Improvements
📚 Multi-document support
💬 Conversation memory
🔎 Hybrid search
📌 Source citations for retrieved chunks
📊 Retrieval evaluation
🌐 Cloud deployment
🔐 Authentication
👥 User-specific document collections
📄 Support for additional document formats
🎯 Learning Outcomes

This project demonstrates practical experience with:

Retrieval-Augmented Generation
Vector databases
Semantic search
Text embeddings
LLM integration
Document processing
Local AI inference
Python application development
Streamlit application development
Git & GitHub
👩‍💻 Author
Vedika Pathak
B.Tech — Artificial Intelligence & Data Science

📌 GitHub: Vedika-commits21

⭐ If you found this project interesting, consider giving the repository a star!



B.Tech – Artificial Intelligence & Data Science

