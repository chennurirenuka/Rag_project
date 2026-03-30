import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.ingest import load_documents, split_documents
from app.config import DATA_PATH, VECTOR_DB_PATH


# =========================
# 🔥 Prompt
# =========================
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant.

Rules:
- Answer ONLY from the given context
- If not found, say "I don't know"
- Be concise

Context:
{context}

Question:
{question}

Answer:
"""
)


# =========================
# Create Vector DB
# =========================
def create_vector_store():
    print("📄 Loading documents...")

    docs = load_documents(DATA_PATH)

    if not docs:
        raise ValueError("No documents found")

    chunks = split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vectorstore.save_local(VECTOR_DB_PATH)

    print("✅ Vector DB created!")

    return vectorstore


# =========================
# Load Vector DB
# =========================
def load_vector_store():
    embeddings = OpenAIEmbeddings()

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =========================
# Create RAG Chain (NEW WAY)
# =========================
def get_qa_chain():
    try:
        vectorstore = load_vector_store()
    except:
        print("⚠️ Creating new DB...")
        vectorstore = create_vector_store()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # 🔥 LCEL Chain (modern)
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    print("✅ RAG Chain ready!")

    return chain


# =========================
# Ask Function
# =========================
def ask_question(chain, query: str):
    try:
        answer = chain.invoke(query)

        return {
            "answer": answer
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}"
        }