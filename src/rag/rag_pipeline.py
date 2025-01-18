from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

class RAGPipeline:
    def __init__(self, llm_model: str, vectorstore: FAISS):
        self.llm = OllamaLLM(model=llm_model)
        self.vectorstore = vectorstore
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        """Create the retrieval QA chain with custom prompt"""
        # Create custom prompt template
        prompt_template = """You are an intelligent assistant helping with enterprise knowledge management. Your goal is to provide accurate, concise, and helpful answers based on the provided context.

        Instructions:
        1. Use ONLY the information from the context provided
        2. If the context doesn't contain enough information, say "I don't have enough information to answer that"
        3. Cite specific details from the context when possible
        4. Keep responses clear and well-structured
        
        Context:
        {context}
        
        Question: {question}
        
        Answer: Let me help you with that information."""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create retriever with custom search parameters
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3,  # Number of relevant chunks to retrieve
                "fetch_k": 5  # Number of documents to initially fetch before filtering
            }
        )

        # Create QA chain with custom prompt
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,  # Include source documents in response
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": True
            }
        )

    def process_query(self, query: str) -> dict:
        """
        Process a query through the RAG pipeline
        
        Returns:
            dict: Contains response text and source documents
        """
        try:
            result = self.qa_chain({"query": query})
            return {
                "answer": result["result"],
                "sources": [doc.metadata for doc in result["source_documents"]]
            }
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}") 