import os
from functools import lru_cache
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama.llms import OllamaLLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from src.api.drive_routes import router as drive_router

load_dotenv()

app = FastAPI(
    title="Enterprise Knowledge Hub",
    description="API for document ingestion and querying using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(drive_router, prefix="/api/v1", tags=["Google Drive"])

# Cache expensive model initializations
@lru_cache()
def get_llm():
    return OllamaLLM(model=os.getenv("MODEL"))

@lru_cache()
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define prompt template as a constant
PROMPT_TEMPLATE = """You are an intelligent assistant helping with enterprise knowledge management.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Helpful Answer: Let me help you with that."""

# Create prompt template once
QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Welcome to Enterprise Knowledge Hub API"}

@lru_cache()
def get_vectorstore():
    return FAISS.load_local(
        os.getenv("VECTORSTORE_PATH"),
        get_embeddings(),
        allow_dangerous_deserialization=True  # Only use if you trust the source
    )

@lru_cache()
def get_qa_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": QA_PROMPT,
            "verbose": True
        }
    )

@app.post("/query")
async def process_query(query_body: QueryRequest):
    try:
        qa_chain = get_qa_chain()
        
        result = qa_chain({"query": query_body.query})
        return {
            "response": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
