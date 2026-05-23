import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer


# Folder where FAISS index and metadata will be stored
VECTOR_DB_FOLDER = "vector_db"

# FAISS index file path
INDEX_FILE = os.path.join(VECTOR_DB_FOLDER, "paper_index.faiss")

# Metadata file path
METADATA_FILE = os.path.join(VECTOR_DB_FOLDER, "metadata.pkl")

# Small, fast embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

# Global model variable so model loads only once
embedding_model = None


def get_embedding_model():
    """
    Loads embedding model only once.

    Embedding model converts text into numerical vectors.
    These vectors help us search similar chunks.
    """

    global embedding_model

    if embedding_model is None:
        embedding_model = SentenceTransformer(MODEL_NAME)

    return embedding_model


def create_vector_store(chunks, filename):
    """
    Converts text chunks into embeddings and stores them in FAISS.

    Current version:
    - One uploaded PDF overwrites old vector DB.
    - Later we will support multiple PDFs.
    """

    os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

    model = get_embedding_model()

    # Convert chunks into vector embeddings
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # FAISS needs float32 format
    embeddings = embeddings.astype("float32")

    # Embedding dimension, example: 384
    dimension = embeddings.shape[1]

    # Since embeddings are normalized, inner product behaves like cosine similarity
    index = faiss.IndexFlatIP(dimension)

    # Add all embeddings to FAISS index
    index.add(embeddings)

    # Store original chunk text separately as metadata
    metadata = []

    for i, chunk in enumerate(chunks):
        metadata.append({
            "filename": filename,
            "chunk_id": i,
            "text": chunk
        })

    # Save FAISS index
    faiss.write_index(index, INDEX_FILE)

    # Save metadata
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    return {
        "total_vectors": len(chunks),
        "embedding_dimension": dimension
    }


def is_noisy_reference_chunk(text):
    """
    Removes reference-heavy chunks from retrieval results.
    """

    lower_text = text.lower()

    if lower_text.count("proc.") >= 2:
        return True

    if lower_text.count("et al.") >= 2:
        return True

    if lower_text.count("references") >= 1 and lower_text.count("[") >= 3:
        return True

    return False


def search_vector_store(query, top_k=3):
    """
    Searches FAISS vector database and returns top relevant chunks.
    """

    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return []

    # Keep top_k in safe range
    top_k = max(1, min(int(top_k), 10))

    model = get_embedding_model()

    # Load FAISS index
    index = faiss.read_index(INDEX_FILE)

    # Load chunk metadata
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    # Convert user question into embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    # Search extra chunks first because we may filter noisy chunks
    search_k = min(top_k + 8, index.ntotal)

    scores, indices = index.search(query_embedding, search_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = metadata[idx]
        text = item["text"]

        # Skip reference/noisy chunks
        if is_noisy_reference_chunk(text):
            continue

        results.append({
            "score": float(score),
            "filename": item["filename"],
            "chunk_id": item["chunk_id"],
            "text": text
        })

        # Return only requested number of useful chunks
        if len(results) == top_k:
            break

    return results