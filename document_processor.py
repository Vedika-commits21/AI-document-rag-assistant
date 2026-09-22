import re
import pymupdf


# ==================================================
# CLEAN TEXT
# ==================================================

def clean_text(text):
    """
    Clean unnecessary spaces and blank lines
    from extracted PDF text.
    """

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    return text.strip()


# ==================================================
# EXTRACT PDF FROM FILE PATH
# ==================================================

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file path.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        text = clean_text(text)

        if text.strip():

            pages.append({
                "page_number": page_number + 1,
                "text": text
            })

    document.close()

    return pages


# ==================================================
# EXTRACT PDF FROM UPLOADED FILE
# ==================================================

def extract_text_from_bytes(pdf_bytes):
    """
    Extract text directly from an uploaded PDF.

    Useful for Streamlit file uploads.
    """

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        text = clean_text(text)

        if text.strip():

            pages.append({
                "page_number": page_number + 1,
                "text": text
            })

    document.close()

    return pages


# ==================================================
# SMART CHUNKING
# ==================================================

def create_chunks(
    pages,
    chunk_size=500,
    overlap=80
):
    """
    Create word-aware chunks from document pages.

    Each chunk contains:
    - chunk_id
    - page_number
    - text
    """

    chunks = []

    chunk_id = 0

    for page in pages:

        page_number = page["page_number"]
        text = page["text"]

        words = text.split()

        if not words:
            continue

        start = 0

        while start < len(words):

            current_words = []
            current_length = 0

            for word in words[start:]:

                additional_length = len(word)

                if current_words:
                    additional_length += 1

                if (
                    current_length + additional_length
                    > chunk_size
                ):
                    break

                current_words.append(word)
                current_length += additional_length

            if not current_words:
                current_words.append(
                    words[start]
                )

            chunk_text = " ".join(
                current_words
            )

            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page_number,
                "text": chunk_text
            })

            chunk_id += 1

            # Approximate character overlap
            overlap_words = max(
                1,
                overlap // 6
            )

            overlap_words = min(
                overlap_words,
                len(current_words) - 1
            )

            step = len(current_words) - overlap_words

            start += step

    return chunks