from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader

app = FastAPI(
    title="Secure RAG Document Assistant API",
    description="Backend API for a secure document question-answering system.",
    version="0.3.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


@app.get("/")
def read_root():
    return {"message": "Secure RAG backend is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a PDF, TXT, or Markdown file.",
        )

    document_id = uuid4()
    saved_filename = f"{document_id}{extension}"
    file_path = UPLOAD_DIR / saved_filename

    content = await file.read()
    file_path.write_bytes(content)

    return {
        "document_id": str(document_id),
        "original_filename": file.filename,
        "stored_filename": saved_filename,
        "message": "Document uploaded successfully.",
    }


@app.get("/documents")
def list_documents():
    documents = []

    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            documents.append(
                {
                    "document_id": file_path.stem,
                    "stored_filename": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "size_bytes": file_path.stat().st_size,
                }
            )

    return {"documents": documents}


@app.get("/documents/{document_id}/text")
def read_document_text(document_id: str):
    matches = list(UPLOAD_DIR.glob(f"{document_id}.*"))

    if not matches:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = matches[0]
    extension = file_path.suffix.lower()

    if extension in {".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8", errors="replace")

    elif extension == ".pdf":
        reader = PdfReader(BytesIO(file_path.read_bytes()))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

    else:
        raise HTTPException(status_code=400, detail="Text extraction is not supported.")

    return {
        "document_id": document_id,
        "filename": file_path.name,
        "text": text,
    }