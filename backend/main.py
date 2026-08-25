from fastapi import FastAPI

app = FastAPI(
    title="Secure RAG Document Assistant API",
    description="Backend API for a secure document question-answering system.",
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {"message": "Secure RAG backend is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}