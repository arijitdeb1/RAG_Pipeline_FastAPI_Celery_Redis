"""
RAG Utilities Module
Contains utility functions for RAG pipeline operations that can be called directly
without going through FastAPI endpoints.
"""

import os
import sys
from typing import Dict, Any, Optional
from celery import chain

# Add the app directory to Python path to import worker modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rag.rag_worker import pdf_reader_task, transform_documents_task, create_vectorstore_task

def run_rag_tasks_step_by_step(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    persist_directory: str = "./individual_tasks_chroma_db"
) -> Dict[str, Any]:
    """
    Run RAG tasks step by step with individual task monitoring.
    This is a direct copy of the FastAPI endpoint function for use in other applications.
    
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each chunk for text splitting
        chunk_overlap (int): Overlap between chunks
        persist_directory (str): Directory to persist the vectorstore
        
    Returns:
        Dict[str, Any]: Result dictionary with task IDs and status information
    """
    
    try:
        # Step 1: PDF Reading
        pdf_task = pdf_reader_task.apply_async(args=[pdf_path], queue='rag')
        pdf_result = pdf_task.get()  # Wait for completion
        
        if pdf_result["status"] == "error":
            return {
                "status": "error",
                "message": pdf_result["message"],
                "completed_tasks": ["pdf_task"],
                "task_ids": {
                    "pdf_task_id": pdf_task.id,
                    "transform_task_id": None,
                    "vectorstore_task_id": None
                }
            }
        
        # Step 2: Document Transformation
        transform_task = transform_documents_task.apply_async(
            args=[pdf_result["documents"], chunk_size, chunk_overlap], 
            queue='rag'
        )
        transform_result = transform_task.get()  # Wait for completion
        
        if transform_result["status"] == "error":
            return {
                "status": "error", 
                "message": transform_result["message"],
                "completed_tasks": ["pdf_task", "transform_task"],
                "task_ids": {
                    "pdf_task_id": pdf_task.id,
                    "transform_task_id": transform_task.id,
                    "vectorstore_task_id": None
                }
            }
        
        # Step 3: Vectorstore Creation
        vectorstore_task = create_vectorstore_task.apply_async(
            args=[transform_result["chunks"], persist_directory],
            queue='rag'
        )
        
        return {
            "status": "vectorstore_task_running",
            "message": "PDF and transform tasks completed. Vectorstore creation in progress.",
            "completed_tasks": ["pdf_task", "transform_task"],
            "running_tasks": ["vectorstore_task"],
            "task_ids": {
                "pdf_task_id": pdf_task.id,
                "transform_task_id": transform_task.id,
                "vectorstore_task_id": vectorstore_task.id
            },
            "intermediate_results": {
                "pdf_result": pdf_result,
                "transform_result": transform_result
            },
            "pipeline_config": {
                "pdf_path": pdf_path,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "persist_directory": persist_directory
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Workflow error: {str(e)}",
            "task_ids": {
                "pdf_task_id": pdf_task.id if 'pdf_task' in locals() else None,
                "transform_task_id": transform_task.id if 'transform_task' in locals() else None,
                "vectorstore_task_id": None
            }
        }

def run_rag_tasks_with_chain(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    persist_directory: str = "./individual_tasks_chroma_db"
) -> Dict[str, Any]:
    """
    Run RAG tasks step by step using Celery chains for sequential execution.
    Returns the same response format as run_rag_tasks_step_by_step.
    """
    try:
        # Create a chain of tasks
        rag_chain = chain(
            pdf_reader_task.s(pdf_path),
            transform_documents_task.s(chunk_size, chunk_overlap),
            create_vectorstore_task.s(persist_directory)
        )
        result = rag_chain.apply_async(queue='rag')
        chain_result = result.get()  # Wait for completion

        # Extract task IDs (Celery does not expose all intermediate IDs in chain, so only root)
        return {
            "status": chain_result.get("status", "completed"),
            "message": chain_result.get("message", "RAG pipeline completed."),
            "completed_tasks": ["pdf_task", "transform_task", "vectorstore_task"],
            "task_ids": {
                "pdf_task_id": result.id,
                "transform_task_id": None,
                "vectorstore_task_id": None
            },
            "intermediate_results": chain_result.get("intermediate_results", {}),
            "pipeline_config": {
                "pdf_path": pdf_path,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "persist_directory": persist_directory
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "completed_tasks": [],
            "task_ids": {
                "pdf_task_id": None,
                "transform_task_id": None,
                "vectorstore_task_id": None
            },
            "intermediate_results": {},
            "pipeline_config": {
                "pdf_path": pdf_path,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "persist_directory": persist_directory
            }
        }

def check_task_status(task_id: str) -> Dict[str, Any]:
    """
    Check the status of a Celery task by its ID.
    
    Args:
        task_id (str): The Celery task ID
        
    Returns:
        Dict[str, Any]: Task status information
    """
    from celery.result import AsyncResult
    from rag.rag_worker import celery_app  # Import the same celery app instance
    
    try:
        if not task_id:
            return {"task_id": task_id, "status": "Error", "error": "No task ID provided"}
            
        # Use the same celery app instance that created the tasks
        task_result = AsyncResult(task_id, app=celery_app)
        
        # Check task state
        if task_result.ready():
            if task_result.successful():
                return {"task_id": task_id, "status": "Completed", "result": task_result.get()}
            else:
                return {"task_id": task_id, "status": "Error", "error": str(task_result.result)}
        else:
            # Task is still pending/running
            state = task_result.state
            if state == 'PENDING':
                return {"task_id": task_id, "status": "Pending", "state": state}
            elif state in ['RETRY', 'PROGRESS']:
                return {"task_id": task_id, "status": "Running", "state": state}
            else:
                return {"task_id": task_id, "status": "Pending", "state": state}
    except Exception as e:
        return {"task_id": task_id, "status": "Error", "error": str(e)}

def query_vectorstore_direct(
    question: str,
    persist_directory: str = "./individual_chroma_db",
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Query the vectorstore directly without going through FastAPI.
    
    Args:
        question (str): The question to ask
        persist_directory (str): Directory where the vectorstore is persisted
        top_k (int): Number of relevant chunks to retrieve
        
    Returns:
        Dict[str, Any]: Query result with answer and context
    """
    from rag.rag_worker import query_vectorstore_task
    
    try:
        # Query the vectorstore synchronously
        query_task = query_vectorstore_task.apply_async(
            kwargs={
                "question": question,
                "persist_directory": persist_directory,
                "top_k": top_k
            },
            queue='rag'
        )
        
        result = query_task.get()  # Wait for completion
        
        if result["status"] == "error":
            return {
                "status": "error",
                "message": result["message"],
                "answer": None,
                "retrieved_chunks": [],
                "sources": []
            }
        
        return {
            "status": "success",
            "question": question,
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
            "sources": result["sources"]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Query error: {str(e)}",
            "answer": None,
            "retrieved_chunks": [],
            "sources": []
        }
