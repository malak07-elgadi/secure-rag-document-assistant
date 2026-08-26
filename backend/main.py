from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader

app = FastAPI(
    title="Secure RAG Document Assistant API",
    description="Backend API for a secure document question-answering system.",
    version="0.4.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


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


def get_document_path(document_id: str) -> Path:
    try:
        UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID.")

    matches = list(UPLOAD_DIR.glob(f"{document_id}.*"))

    if not matches:
        raise HTTPException(status_code=404, detail="Document not found.")

    return matches[0]


def extract_text_from_file(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="replace")

    if extension == ".pdf":
        reader = PdfReader(BytesIO(file_path.read_bytes()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    raise HTTPException(status_code=400, detail="Text extraction is not supported.")


def split_text_into_chunks(text: str) -> list[str]:
    clean_text = " ".join(text.split())

    if not clean_text:
        return []

    chunks = []
    start = 0

    while start < len(clean_text):
        end = min(start + CHUNK_SIZE, len(clean_text))

        if end < len(clean_text):
            last_space = clean_text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunks.append(clean_text[start:end].strip())

        if end == len(clean_text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


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
    file_path = get_document_path(document_id)
    text = extract_text_from_file(file_path)

    return {
        "document_id": document_id,
        "filename": file_path.name,
        "text": text,
    }


@app.get("/documents/{document_id}/chunks")
def get_document_chunks(document_id: str):
    file_path = get_document_path(document_id)
    text = extract_text_from_file(file_path)
    chunk_texts = split_text_into_chunks(text)

    chunks = [
        {
            "chunk_id": index,
            "text": chunk_text,
            "character_count": len(chunk_text),
        }
        for index, chunk_text in enumerate(chunk_texts)
    ]

    return {
        "document_id": document_id,
        "filename": file_path.name,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }