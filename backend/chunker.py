def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Splits long text into smaller chunks.

    chunk_size = maximum characters in one chunk
    overlap = repeated characters between chunks
              so context is not lost between chunks
    """

    chunks = []

    # Start from beginning of text
    start = 0

    while start < len(text):
        # End position of current chunk
        end = start + chunk_size

        # Extract chunk
        chunk = text[start:end]

        # Remove extra spaces/newlines
        chunk = chunk.strip()

        # Add chunk only if it has content
        if chunk:
            chunks.append(chunk)

        # Move forward but keep some overlap
        start = end - overlap

    return chunks