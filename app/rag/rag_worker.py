import os
from celery import Celery, chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Configure Celery
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("rag_tasks", broker=redis_url, backend=redis_url)

# Note: We rely on docker-compose --queues=rag and explicit apply_async(queue='rag') calls
# instead of automatic task routing

@celery_app.task
def pdf_reader_task(pdf_path: str):
    """
    Celery task to load and read PDF documents.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of document objects with page content and metadata
    """
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Convert documents to serializable format
        serialized_docs = []
        for doc in docs:
            serialized_docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "status": "success",
            "message": f"Successfully loaded {len(docs)} pages from {pdf_path}",
            "documents": serialized_docs,
            "doc_count": len(docs)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading PDF: {str(e)}",
            "documents": [],
            "doc_count": 0
        }

@celery_app.task
def transform_documents_task(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Celery task to split documents into chunks.
    
    Args:
        documents (list): List of document dictionaries with page_content and metadata
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    try:
        # Reconstruct document objects from serialized format
        from langchain.schema import Document
        doc_objects = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) 
                      for doc in documents]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(doc_objects)
        
        # Convert chunks to serializable format
        serialized_chunks = []
        for chunk in chunks:
            serialized_chunks.append({
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            })
        
        return {
            "status": "success",
            "message": f"Successfully created {len(chunks)} chunks",
            "chunks": serialized_chunks,
            "chunk_count": len(chunks)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error transforming documents: {str(e)}",
            "chunks": [],
            "chunk_count": 0
        }

@celery_app.task
def create_vectorstore_task(chunks: list, persist_directory: str = "./chroma_db"):
    """
    Celery task to create vector embeddings and store in Chroma vectorstore.
    
    Args:
        chunks (list): List of text chunks with page_content and metadata
        persist_directory (str): Directory to persist the vectorstore
        
    Returns:
        dict: Status and vectorstore information
    """
    try:
        # Reconstruct document objects from serialized format
        from langchain.schema import Document
        chunk_objects = [Document(page_content=chunk["page_content"], metadata=chunk["metadata"]) 
                        for chunk in chunks]
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vectorstore
        db = Chroma.from_documents(
            documents=chunk_objects,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Get collection information
        collection_name = db._collection.name
        collection_id = db._collection.id
        
        return {
            "status": "success",
            "message": f"Successfully created vectorstore with {len(chunks)} chunks",
            "vectorstore_path": persist_directory,
            "chunk_count": len(chunks),
            "embedding_model": "OpenAI",
            "collection_name": collection_name,
            "collection_id": collection_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating vectorstore: {str(e)}",
            "vectorstore_path": None,
            "chunk_count": 0
        }

@celery_app.task
def extract_documents_for_chain(pdf_result):
    """
    Simple extractor task to get documents from PDF result for chaining.
    """
    if pdf_result["status"] == "error":
        raise Exception(pdf_result["message"])
    return pdf_result["documents"]

@celery_app.task
def extract_chunks_for_chain(transform_result):
    """
    Simple extractor task to get chunks from transform result for chaining.
    """
    if transform_result["status"] == "error":
        raise Exception(transform_result["message"])
    return transform_result["chunks"]

@celery_app.task
def query_vectorstore_task(question: str, persist_directory: str = "./individual_chroma_db", top_k: int = 3):
    """
    Celery task to query the vectorstore and generate an answer using RAG with LCEL format.
    
    Args:
        question (str): The question to ask
        persist_directory (str): Directory where the vectorstore is persisted
        top_k (int): Number of relevant chunks to retrieve
        
    Returns:
        dict: Answer and retrieved context
    """
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        
        print(f"DEBUG: Loading vectorstore from {persist_directory}")
        
        # Load existing vectorstore
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        # Check if vectorstore has documents
        collection = db._collection
        print(f"DEBUG: Vectorstore collection count: {collection.count()}")
        
        if collection.count() == 0:
            return {
                "status": "error",
                "message": f"Vectorstore at {persist_directory} is empty. Run /rag-individual-tasks/ first to populate it.",
                "answer": None,
                "retrieved_chunks": [],
                "sources": []
            }
        
        # Create retriever
        retriever = db.as_retriever(search_kwargs={"k": top_k})
        
        # Test retrieval first
        print(f"DEBUG: Testing retrieval for question: {question}")
        retrieved_docs = retriever.get_relevant_documents(question)
        print(f"DEBUG: Retrieved {len(retrieved_docs)} documents")
        
        for i, doc in enumerate(retrieved_docs):
            print(f"DEBUG: Doc {i+1} preview: {doc.page_content[:100]}...")
        
        if not retrieved_docs:
            return {
                "status": "error",
                "message": f"No relevant documents found for question: {question}",
                "answer": "I don't know - no relevant context was retrieved.",
                "retrieved_chunks": [],
                "sources": []
            }
        
        # Create LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context.
Think step by step before providing an answer. Just say "I don't know" 
if you're not sure of an answer.

<context>
{context}
</context>

Question: {question}
""")
        
        # Format documents function for context
        def format_docs(docs):
            formatted = "\n\n".join(doc.page_content for doc in docs)
            print(f"DEBUG: Formatted context length: {len(formatted)} characters")
            print(f"DEBUG: Context preview: {formatted[:200]}...")
            return formatted
        
        # Create RAG chain using LCEL format
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print(f"DEBUG: Invoking RAG chain with question: {question}")
        
        # Generate answer using LCEL chain
        answer = rag_chain.invoke(question)
        
        print(f"DEBUG: Generated answer: {answer}")
        
        # Extract sources and chunks
        sources = []
        retrieved_chunks = []
        
        for doc in retrieved_docs:
            retrieved_chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
            # Extract source information
            source_info = doc.metadata.get("source", "Unknown")
            if source_info not in sources:
                sources.append(source_info)
        
        return {
            "status": "success",
            "answer": answer,
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "sources": sources,
            "context_used": len(retrieved_docs)
        }
        
    except Exception as e:
        print(f"DEBUG: Error in query_vectorstore_task: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error querying vectorstore: {str(e)}",
            "answer": None,
            "retrieved_chunks": [],
            "sources": []
        }

# Configure Celery settings
celery_app.conf.update(
    task_serializer='json', # How tasks are converted for storage/transport
    result_serializer='json',  # How task results are converted for storage
    accept_content=['json'],  # Only accept JSON format (security)
    result_expires=3600, # Task results expire after 1 hour (3600 seconds)
    timezone='UTC',   # Use UTC timezone for all timestamps
    enable_utc=True, # Enable UTC timezone handling
)
