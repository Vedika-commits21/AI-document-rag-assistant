import streamlit as st
import urllib.request
import urllib.error
import json
import html

import faiss
import numpy as np
import pymupdf

from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Document RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading AI embedding model...")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL APP
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99,102,241,0.10),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(14,165,233,0.08),
                transparent 25%
            ),
            #0b1020;

        color: #f8fafc !important;
    }


    /* ========================================================
       HIDE STREAMLIT HEADER
       ======================================================== */

    header {
        visibility: hidden;
    }


    /* ========================================================
       MAIN CONTAINER
       ======================================================== */

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(148,163,184,0.15);
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #f8fafc !important;
    }


    /* ========================================================
       BRAND
       ======================================================== */

    .brand {
        font-size: 25px;
        font-weight: 900;
        letter-spacing: 1px;
        margin-bottom: 5px;
        color: #ffffff !important;
    }

    .brand-subtitle {
        color: #cbd5e1 !important;
        font-size: 13px;
        margin-bottom: 30px;
        font-weight: 600;
    }


    /* ========================================================
       NORMAL STREAMLIT HEADINGS
       ======================================================== */

    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    h1,
    h2,
    h3,
    h4 {
        color: #ffffff !important;
        font-weight: 900 !important;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-weight: 800 !important;
        font-size: 14px !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 900 !important;
        font-size: 36px !important;
    }


    /* ========================================================
       TECH CARDS
       ======================================================== */

    .tech-card {
        padding: 18px;
        border-radius: 16px;

        background: rgba(15,23,42,0.80);

        border: 1px solid rgba(148,163,184,0.18);

        margin-bottom: 12px;

        box-shadow:
            0 8px 25px rgba(0,0,0,0.12);
    }

    .tech-card-title {
        font-size: 12px;
        color: #cbd5e1 !important;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 800;
    }

    .tech-card-value {
        font-size: 15px;
        font-weight: 900;
        color: #ffffff !important;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status {
        display: inline-block;

        margin-top: 18px;

        padding: 7px 13px;

        border-radius: 20px;

        background: rgba(34,197,94,0.10);

        border: 1px solid rgba(34,197,94,0.25);

        color: #86efac !important;

        font-size: 12px;

        font-weight: 800;
    }


    /* ========================================================
       ANSWER CARD
       ======================================================== */

    .answer-box {
        padding: 24px;

        border-radius: 20px;

        background:
            linear-gradient(
                135deg,
                rgba(30,41,59,0.95),
                rgba(15,23,42,0.92)
            );

        border: 1px solid rgba(139,92,246,0.35);

        box-shadow:
            0 10px 40px rgba(0,0,0,0.20);

        margin-top: 20px;
        margin-bottom: 12px;
    }

    .answer-label {
        color: #c4b5fd !important;

        font-size: 13px;

        font-weight: 900 !important;

        letter-spacing: 1.5px;
    }


    /* ========================================================
       SOURCE SECTION
       ======================================================== */

    .sources-heading {
        font-size: 24px;

        font-weight: 900 !important;

        color: #ffffff !important;

        margin-top: 32px;

        margin-bottom: 16px;
    }


    /* ========================================================
       SOURCE CARD
       ======================================================== */

    .source-box {
        padding: 20px 22px;

        margin: 14px 0;

        border-radius: 18px;

        background:
            linear-gradient(
                135deg,
                rgba(30,41,59,0.96),
                rgba(15,23,42,0.90)
            );

        border: 1px solid rgba(139,92,246,0.30);

        box-shadow:
            0 8px 28px rgba(0,0,0,0.18);

        transition: all 0.2s ease;
    }

    .source-box:hover {
        border-color: rgba(167,139,250,0.60);

        transform: translateY(-2px);

        box-shadow:
            0 12px 35px rgba(0,0,0,0.28);
    }


    /* ========================================================
       SOURCE TITLE
       ======================================================== */

    .source-title {
        font-size: 16px;

        font-weight: 900 !important;

        color: #ffffff !important;

        margin-bottom: 13px;

        line-height: 1.5;
    }


    /* ========================================================
       SOURCE BADGE
       ======================================================== */

    .source-rank {
        display: inline-block;

        padding: 4px 9px;

        margin-right: 9px;

        border-radius: 8px;

        background: rgba(139,92,246,0.20);

        border: 1px solid rgba(167,139,250,0.30);

        color: #c4b5fd !important;

        font-size: 10px;

        font-weight: 900 !important;

        letter-spacing: 0.8px;
    }


    /* ========================================================
       PAGE NUMBER
       ======================================================== */

    .source-page {
        color: #c4b5fd !important;

        font-weight: 900 !important;
    }


    /* ========================================================
       SOURCE TEXT
       ======================================================== */

    .source-text {
        font-size: 14px;

        color: #e2e8f0 !important;

        line-height: 1.75;

        font-weight: 600 !important;
    }


    /* ========================================================
       SOURCE META
       ======================================================== */

    .source-meta {
        margin-top: 13px;

        padding-top: 10px;

        border-top:
            1px solid rgba(148,163,184,0.12);

        font-size: 11px;

        color: #94a3b8 !important;

        font-weight: 700 !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        border-radius: 12px;

        border:
            1px solid rgba(139,92,246,0.40);

        background: #7c3aed;

        color: #ffffff !important;

        font-weight: 900;

        padding: 10px 18px;
    }

    .stButton > button:hover {
        background: #6d28d9;

        border-color: #a78bfa;
    }


    /* ========================================================
       INPUT
       ======================================================== */

    div[data-baseweb="input"] {
        border-radius: 12px;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    section[data-testid="stFileUploaderDropzone"] {
        background: rgba(15,23,42,0.65);

        border:
            1px dashed rgba(139,92,246,0.45);

        border-radius: 15px;
    }

    section[data-testid="stFileUploaderDropzone"] * {
        color: #e2e8f0 !important;
    }


    /* ========================================================
       CAPTIONS
       ======================================================== */

    [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "processed" not in st.session_state:
    st.session_state.processed = False

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_mode" not in st.session_state:
    st.session_state.last_mode = ""


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_bytes):

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "page_number": page_number + 1,
                    "text": text
                }
            )

    document.close()

    return pages


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    pages,
    chunk_size=700,
    overlap=100
):

    chunks = []

    chunk_id = 0

    for page in pages:

        text = " ".join(
            page["text"].split()
        )

        if not text:
            continue

        start = 0

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text)
            )

            chunk_text = (
                text[start:end]
                .strip()
            )

            if len(chunk_text) >= 40:

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page_number": page["page_number"],
                        "text": chunk_text
                    }
                )

                chunk_id += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_documents(uploaded_files):

    all_chunks = []

    document_names = []

    for uploaded_file in uploaded_files:

        pdf_bytes = uploaded_file.getvalue()

        pages = extract_pdf_pages(
            pdf_bytes
        )

        chunks = create_chunks(
            pages
        )

        for chunk in chunks:

            chunk["source"] = (
                uploaded_file.name
            )

            all_chunks.append(
                chunk
            )

        document_names.append(
            uploaded_file.name
        )

    if not all_chunks:

        return None, [], []

    index = build_index(
        all_chunks
    )

    return (
        index,
        all_chunks,
        document_names
    )


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    question,
    text
):

    question_words = {
        word.lower().strip(".,?!:;()[]{}")
        for word in question.split()
        if len(word) > 2
    }

    if not question_words:
        return 0.0

    text_lower = text.lower()

    matched = sum(
        1
        for word in question_words
        if word in text_lower
    )

    return matched / len(
        question_words
    )


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def retrieve_chunks(
    question,
    top_k=8
):

    index = st.session_state.index

    chunks = st.session_state.chunks

    if index is None or not chunks:
        return []

    # --------------------------------------------------------
    # Semantic embedding
    # --------------------------------------------------------

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    # Retrieve more candidates first,
    # then rank them using hybrid scoring.
    search_k = min(
        max(top_k * 3, 20),
        len(chunks)
    )

    distances, indices = index.search(
        question_embedding,
        search_k
    )

    candidates = []

    # --------------------------------------------------------
    # Hybrid ranking
    # --------------------------------------------------------

    for rank, chunk_index in enumerate(
        indices[0]
    ):

        if chunk_index < 0:
            continue

        chunk = chunks[
            chunk_index
        ]

        distance = float(
            distances[0][rank]
        )

        semantic_score = (
            1.0 / (1.0 + distance)
        )

        kw_score = keyword_score(
            question,
            chunk["text"]
        )

        final_score = (
            0.75 * semantic_score
            +
            0.25 * kw_score
        )

        candidates.append(
            {
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "distance": distance,
                "score": final_score
            }
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Remove duplicate chunks
    # --------------------------------------------------------

    results = []

    seen = set()

    for candidate in candidates:

        unique_key = (
            candidate["source"],
            candidate["page_number"],
            candidate["text"][:120]
        )

        if unique_key in seen:
            continue

        seen.add(unique_key)

        candidate["rank"] = (
            len(results) + 1
        )

        results.append(
            candidate
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# DOCUMENT OVERVIEW
# ============================================================

def get_document_overview(
    max_sources=8
):

    chunks = st.session_state.chunks

    if not chunks:
        return []

    # Group chunks by source document
    documents = {}

    for chunk in chunks:

        source = chunk["source"]

        if source not in documents:
            documents[source] = []

        documents[source].append(
            chunk
        )

    selected = []

    for source, source_chunks in documents.items():

        if not source_chunks:
            continue

        total = len(source_chunks)

        sample_count = min(
            max_sources,
            total
        )

        if total <= sample_count:

            selected.extend(
                source_chunks
            )

        else:

            positions = np.linspace(
                0,
                total - 1,
                sample_count,
                dtype=int
            )

            for position in positions:

                selected.append(
                    source_chunks[position]
                )

    return selected[:max_sources]


# ============================================================
# OVERVIEW QUESTION DETECTOR
# ============================================================

def is_overview_question(
    question
):

    q = question.lower().strip()

    overview_patterns = [

        "what is mentioned in document",

        "what is mentioned in the document",

        "what is in the document",

        "what does this document contain",

        "what does the document contain",

        "summarize the document",

        "summary of the document",

        "give me an overview",

        "document overview",

        "what topics are covered",

        "what topics does this document cover",

        "tell me about this document",

        "what is this document about",

        "explain this document",

        "what topics are in this document"

    ]

    return any(
        pattern in q
        for pattern in overview_patterns
    )


# ============================================================
# GENERATE ANSWER USING OLLAMA
# ============================================================

def generate_answer(
    question
):

    # --------------------------------------------------------
    # Retrieval mode
    # --------------------------------------------------------

    if is_overview_question(
        question
    ):

        source_chunks = (
            get_document_overview(
                max_sources=8
            )
        )

        mode = "overview"

    else:

        source_chunks = (
            retrieve_chunks(
                question,
                top_k=8
            )
        )

        mode = "search"

    if not source_chunks:

        return (
            "No relevant content could be retrieved "
            "from the uploaded document.",
            [],
            mode
        )

    # --------------------------------------------------------
    # Build document context
    # --------------------------------------------------------

    context_parts = []

    for chunk in source_chunks:

        context_parts.append(
            f"""
[Document: {chunk['source']}]
[Page {chunk['page_number']}]

{chunk['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    # Keep local LLM context manageable
    max_context_chars = 12000

    if len(context) > max_context_chars:

        context = (
            context[:max_context_chars]
            + "\n[Context truncated]"
        )

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    if mode == "overview":

        instruction = """
The user wants an overview of the uploaded document.

Create a useful and structured overview using ONLY the
provided document context.

Mention information that is actually visible in the context,
such as:
- major subjects
- units or sections
- important topics
- practical or laboratory subjects
- academic areas
- document purpose when clearly visible

Do not say the answer is unavailable merely because the
question is broad.

Do not invent information.
"""

    else:

        instruction = """
The user wants information about a specific topic.

Use the retrieved document sections to answer the question.

If the topic appears in the context:
- explain the relevant information clearly
- mention the relevant page number when possible

If the retrieved context does not contain useful information
about the requested topic, say that the requested topic was
not found in the retrieved document sections.

Do not invent information.
"""

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI Document Intelligence Assistant.

{instruction}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    # --------------------------------------------------------
    # Ollama
    # --------------------------------------------------------

    url = (
        "http://localhost:11434/api/generate"
    )

    data = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192
        }
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(
            data
        ).encode("utf-8"),
        headers={
            "Content-Type":
                "application/json"
        }
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=180
        ) as response:

            result = json.loads(
                response
                .read()
                .decode("utf-8")
            )

        answer = result.get(
            "response",
            ""
        ).strip()

    except urllib.error.URLError:

        raise Exception(
            "Ollama is not running. "
            "Please start Ollama and try again."
        )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources = []

    for rank, chunk in enumerate(
        source_chunks
    ):

        sources.append(
            {
                "rank": rank + 1,
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "distance": chunk.get(
                    "distance",
                    0.0
                )
            }
        )

    return (
        answer,
        sources,
        mode
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            🧠 RAG ASSISTANT
        </div>

        <div class="brand-subtitle">
            AI Document Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### 📄 Documents"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents."
    )

    if uploaded_files:

        current_names = [
            file.name
            for file in uploaded_files
        ]

        # Process only when uploaded
        # document set changes
        if (
            current_names
            != st.session_state.documents
        ):

            with st.spinner(
                "Processing documents..."
            ):

                (
                    index,
                    chunks,
                    document_names
                ) = process_documents(
                    uploaded_files
                )

                st.session_state.index = (
                    index
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.documents = (
                    document_names
                )

                st.session_state.processed = (
                    bool(chunks)
                )

                st.session_state.last_question = ""
                st.session_state.last_answer = ""
                st.session_state.last_sources = []
                st.session_state.last_mode = ""

        st.success(
            f"{len(uploaded_files)} document(s) ready"
        )

        for file in uploaded_files:

            st.caption(
                f"📄 {file.name}"
            )

    else:

        st.info(
            "Upload a PDF to start."
        )

    st.markdown("---")

    st.markdown(
        "### ⚙️ AI Stack"
    )

    st.markdown(
        """
        <div class="tech-card">
            <div class="tech-card-title">
                LLM
            </div>
            <div class="tech-card-value">
                Llama 3.2 · 3B
            </div>
        </div>

        <div class="tech-card">
            <div class="tech-card-title">
                Vector Database
            </div>
            <div class="tech-card-value">
                FAISS
            </div>
        </div>

        <div class="tech-card">
            <div class="tech-card-title">
                Embeddings
            </div>
            <div class="tech-card-value">
                all-MiniLM-L6-v2
            </div>
        </div>

        <div class="tech-card">
            <div class="tech-card-title">
                AI Runtime
            </div>
            <div class="tech-card-value">
                Ollama · Local
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.processed:

        st.markdown(
            """
            <div class="status">
                ● DOCUMENT INDEX READY
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="status">
                ● LOCAL AI ONLINE
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DOCUMENT STATS
# ============================================================

if st.session_state.processed:

    total_chunks = len(
        st.session_state.chunks
    )

    total_pages = len(
        {
            (
                chunk["source"],
                chunk["page_number"]
            )
            for chunk in st.session_state.chunks
        }
    )

    vector_dimension = 384

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Pages",
            total_pages
        )

    with col2:

        st.metric(
            "🧩 Chunks",
            total_chunks
        )

    with col3:

        st.metric(
            "🔎 Vector Dimension",
            vector_dimension
        )


# ============================================================
# ASK DOCUMENT
# ============================================================

st.markdown(
    "### 💬 Ask your document"
)

question = st.text_input(
    "Question",
    placeholder=(
        "Try: Search Python lab topics, "
        "What topics are covered in this syllabus?, "
        "Explain Unit 4..."
    ),
    label_visibility="collapsed"
)

ask_button = st.button(
    "✨ Ask AI"
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    if not st.session_state.processed:

        st.warning(
            "Please upload a PDF document first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching document and generating answer..."
        ):

            try:

                (
                    answer,
                    sources,
                    mode
                ) = generate_answer(
                    question
                )

                st.session_state.last_question = (
                    question
                )

                st.session_state.last_answer = (
                    answer
                )

                st.session_state.last_sources = (
                    sources
                )

                st.session_state.last_mode = (
                    mode
                )

            except Exception as error:

                st.error(
                    str(error)
                )


# ============================================================
# DISPLAY ANSWER
# ============================================================

if st.session_state.last_answer:

    st.markdown(
        "### 🤖 AI RESPONSE"
    )

    st.markdown(
        st.session_state.last_answer
    )


# ============================================================
# DISPLAY SOURCES
# ============================================================

if st.session_state.last_sources:

    if st.session_state.last_mode == "overview":
        st.markdown(
            "### 📚 Document Context"
        )
    else:
        st.markdown(
            "### 📚 Retrieved Sources"
        )

    for source in st.session_state.last_sources:

        # Create a clean Streamlit card
        with st.container(border=True):

            # ------------------------------------------------
            # SOURCE HEADER
            # ------------------------------------------------

            st.markdown(
                f"**SOURCE {source['rank']}**  ·  "
                f"📄 **{source['source']}**  ·  "
                f"**Page {source['page_number']}**"
            )

            # ------------------------------------------------
            # SOURCE TEXT
            # ------------------------------------------------

            text = source["text"]

            if len(text) > 450:
                text = (
                    text[:450].rstrip()
                    + "..."
                )

            st.write(text)

            # ------------------------------------------------
            # SOURCE META
            # ------------------------------------------------

            if st.session_state.last_mode == "overview":

                st.caption(
                    "📌 Document overview context"
                )

            else:

                st.caption(
                    f"🔎 Semantic distance: "
                    f"{source['distance']:.4f}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Document RAG Assistant • "
    "Python + Streamlit + FAISS + MiniLM + Ollama"
)