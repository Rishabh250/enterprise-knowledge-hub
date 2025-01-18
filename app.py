import os
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

# Load environment variables
load_dotenv()

# Initialize FastAPI app
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

# Initialize models and embeddings
llm_model = OllamaLLM(model=os.getenv("MODEL"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define prompt template
PROMPT_TEMPLATE = """You are an intelligent assistant helping with enterprise knowledge management.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Helpful Answer: Let me help you with that."""

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Welcome to Enterprise Knowledge Hub API"}

@app.post("/query")
async def process_query(query_body: QueryRequest):
    try:
        vectorstore = FAISS.load_local(
            os.getenv("VECTORSTORE_PATH"), 
            embeddings,
            allow_dangerous_deserialization=True  # Only use if you trust the source
        )

        # Create prompt
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Create retrieval chain
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True
            }
        )
        
        # Get response
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