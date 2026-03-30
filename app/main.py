from fastapi import FastAPI
from pydantic import BaseModel

from app.rag_pipeline import get_qa_chain, ask_question

app = FastAPI(title="RAG Chatbot API")

qa_chain = None


class QueryRequest(BaseModel):
    question: str


# =========================
# Startup
# =========================
@app.on_event("startup")
def startup_event():
    global qa_chain
    print("🚀 Starting server...")

    qa_chain = get_qa_chain()

    print("✅ Server ready!")


# =========================
# Routes
# =========================
@app.get("/")
def home():
    return {"message": "RAG Chatbot API is running"}


@app.post("/ask")
def ask(req: QueryRequest):
    result = ask_question(qa_chain, req.question)
    return result