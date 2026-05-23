import re


def clean_text(text):
    """
    Cleans extra spaces and broken formatting from extracted PDF text.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=180, overlap=40):
    """
    Splits text into word-based chunks instead of character-based chunks.

    chunk_size = number of words in one chunk
    overlap = repeated words between chunks
    """

    text = clean_text(text)

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks