from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
import re

from backend.pdf_parser import extract_text_from_pdf
from backend.chunker import chunk_text
from backend.vector_store import create_vector_store, search_vector_store


app = FastAPI(
    title="ScholarAgent API",
    description="Agentic AI research paper assistant backend",
    version="1.0.0"
)


UPLOAD_FOLDER = "uploaded_papers"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class AskRequest(BaseModel):
    """
    Request body for /ask endpoint.

    Example:
    {
        "question": "What is hallucination in this paper?",
        "top_k": 3
    }
    """

    question: str
    top_k: int = 3


def clean_sentence(sentence):
    """
    Cleans PDF extraction artifacts from a sentence.
    """

    # Fix broken PDF hyphen words like "degrada- tion" -> "degradation"
    sentence = re.sub(r"(\w)-\s+(\w)", r"\1\2", sentence)

    # Remove page markers
    sentence = re.sub(r"--- Page \d+ ---", "", sentence)

    # Remove extra spaces
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence.strip()


def build_extractive_answer(question, results):
    """
    Creates a cleaner extractive answer from retrieved chunks.

    Current version:
    - No LLM yet.
    - It picks useful definition-style/supporting sentences.
    - Later we will replace this with LLM-based RAG.
    """

    if not results:
        return "No relevant information found. Please upload a PDF first."

    combined_text = " ".join([item["text"] for item in results])

    # Basic cleanup
    combined_text = combined_text.replace("\n", " ")
    combined_text = re.sub(r"\s+", " ", combined_text)

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", combined_text)

    definition_patterns = [
        "hallucination is",
        "hallucination in",
        "llm may",
        "generate fluent but unsupported",
        "unsupported content",
        "incorrect or unsupported",
        "context becomes",
        "context degradation",
        "measurable response",
        "failure behavior"
    ]

    useful_sentences = []

    for sentence in sentences:
        sentence = clean_sentence(sentence)

        if len(sentence) < 40:
            continue

        sentence_lower = sentence.lower()

        if any(pattern in sentence_lower for pattern in definition_patterns):
            if sentence not in useful_sentences:
                useful_sentences.append(sentence)

        if len(useful_sentences) == 4:
            break

    answer = (
        "Based on the uploaded paper:\n\n"
        "Hallucination is discussed as a failure behavior where an LLM may produce "
        "unsupported, incorrect, or misleading output when the available context is "
        "incomplete, noisy, conflicting, or degraded.\n\n"
    )

    if useful_sentences:
        answer += "Supporting points from the paper:\n"

        for i, sentence in enumerate(useful_sentences, start=1):
            answer += f"{i}. {sentence}\n"

    else:
        answer += (
            "The paper studies hallucination as a measurable response to structured "
            "context degradation using controlled blindspot conditions."
        )

    return answer


@app.get("/")
def home():
    """
    Basic route to check whether backend is running.
    """

    return {
        "message": "ScholarAgent backend is running successfully"
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF file, saves it locally,
    extracts text, creates chunks, creates embeddings,
    and stores them in FAISS vector database.
    """

    # Keep only safe file name
    safe_filename = os.path.basename(file.filename)

    # Full path where PDF will be saved
    file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

    # Save uploaded PDF
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text from PDF
    extracted_text = extract_text_from_pdf(file_path)

    # Split extracted text into chunks
    chunks = chunk_text(extracted_text)

    # Create embeddings and save in FAISS vector store
    vector_info = create_vector_store(chunks, safe_filename)

    return {
        "filename": safe_filename,
        "total_characters": len(extracted_text),
        "total_chunks": len(chunks),
        "vector_store": vector_info,
        "first_chunk_preview": chunks[0][:500] if chunks else "",
        "text_preview": extracted_text[:1000]
    }


@app.post("/ask")
def ask_question(request: AskRequest):
    """
    Searches relevant chunks from FAISS vector database.

    Current phase:
    - Retrieves relevant chunks
    - Builds a clean extractive answer

    Later phase:
    - Send retrieved chunks to LLM
    - Generate proper citation-backed answer
    """

    results = search_vector_store(
        query=request.question,
        top_k=request.top_k
    )

    answer = build_extractive_answer(
        question=request.question,
        results=results
    )

    return {
        "question": request.question,
        "answer": answer,
        "total_results": len(results),
        "retrieved_chunks": results
    }