from fastapi import FastAPI, UploadFile, File
import shutil
import os

from backend.pdf_parser import extract_text_from_pdf
from backend.chunker import chunk_text


app = FastAPI(
    title="ScholarAgent API",
    description="Agentic AI research paper assistant backend",
    version="1.0.0"
)

UPLOAD_FOLDER = "uploaded_papers"

# Create upload folder if it does not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    """
    Basic route to check if backend is running.
    """
    return {
        "message": "ScholarAgent backend is running successfully"
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, save it locally, extract text, and split it into chunks.
    """

    # Keep filename safe
    safe_filename = os.path.basename(file.filename)

    file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

    # Save uploaded PDF file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract full text from PDF
    extracted_text = extract_text_from_pdf(file_path)

    # Split extracted text into chunks
    chunks = chunk_text(extracted_text)

    return {
        "filename": safe_filename,
        "total_characters": len(extracted_text),
        "total_chunks": len(chunks),
        "first_chunk_preview": chunks[0][:500] if chunks else "",
        "text_preview": extracted_text[:1000]
    }