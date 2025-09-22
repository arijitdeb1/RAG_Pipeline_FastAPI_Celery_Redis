import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from celery.result import AsyncResult
from worker import process_data, add_numbers, divide_numbers, aggregate_results
from celery import chain, group, chord
from rag.rag_worker import pdf_reader_task, transform_documents_task, create_vectorstore_task, query_vectorstore_task, extract_documents_for_chain, extract_chunks_for_chain
from pydantic import BaseModel

app = FastAPI(title="FastAPI and Celery Example")

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Celery demo!"}

@app.post("/start-task/")
async def start_processing_task(data: str):
    """Endpoint to start a background task."""
    task = process_data.apply_async(args=[data], queue='default')
    return {"task_id": task.id, "status": "Task started"}

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Endpoint to get the status of a task by its ID."""
    task_result = AsyncResult(task_id)
    if task_result.ready():
        return {"task_id": task_id, "status": "Completed", "result": task_result.get()}
    else:
        return {"task_id": task_id, "status": "Pending"}

@app.post("/add/{x}/{y}")
async def add(x: int, y: int):
    """Endpoint to run a calculation task."""
    task = add_numbers.apply_async(args=[x, y], queue='default')
    return {"task_id": task.id, "status": "Task submitted"}

@app.post("/divide/{x}/{y}")
async def divide(x: int, y: int):
    """Endpoint to run a division task."""
    task = divide_numbers.apply_async(args=[x, y], queue='default')
    return {"task_id": task.id, "status": "Task submitted"}

@app.post("/calculate-chain/{x}/{y}/{z}")
async def calculate_chain(x: int, y: int, z: int):
    """Endpoint to run a chained calculation task."""
    # Create the chain - both tasks will go to 'default' queue
    task_chain = chain(
        add_numbers.s(x, y).set(queue='default'),
        divide_numbers.s(z).set(queue='default')
    )

    # Trigger the chain and get the result object for the final task
    result = task_chain.apply_async()

    return {"task_id": result.id, "status": "Task chain submitted"}

@app.post("/calculate-chord/{x}/{y}/{z}")
async def calculate_chord(x: int, y: int, z: int):
    """
    Endpoint to run a chord, executing tasks in parallel.
    The sum (x+y) and division (x/z) are performed simultaneously.
    The results are then aggregated by a final callback task.
    """
    # Define the group of tasks to be executed in parallel (the header)
    header = group(
        add_numbers.s(x, y).set(queue='default'),
        divide_numbers.s(x, z).set(queue='default')
    )

    # Define the callback task (the body)
    callback = aggregate_results.s().set(queue='default')

    # Create the chord and trigger it
    result = chord(header)(callback)

    return {"task_id": result.id, "status": "Chord task submitted"}

@app.post("/rag-individual-tasks/")
async def run_rag_individual_tasks(filename: str = Query(..., description="PDF filename in uploaded_files folder")):
    """Endpoint to run individual RAG tasks step by step with a PDF from uploaded_files folder."""
    temp_pdf_path = f"uploaded_files/{filename}"
    # Step 1: Read PDF - specify queue='rag'
    pdf_task = pdf_reader_task.apply_async(args=[temp_pdf_path], queue='rag')
    pdf_result = pdf_task.get()

    if pdf_result["status"] == "error":
        return {"status": "error", "message": pdf_result["message"]}

    # Step 2: Transform documents - specify queue='rag'
    transform_task = transform_documents_task.apply_async(
        args=[pdf_result["documents"], 1000, 200],
        queue='rag'
    )
    transform_result = transform_task.get()

    if transform_result["status"] == "error":
        return {"status": "error", "message": transform_result["message"]}

    # Step 3: Create vectorstore - specify queue='rag'
    vectorstore_task = create_vectorstore_task.apply_async(
        args=[transform_result["chunks"], "./individual_chroma_db"],
        queue='rag'
    )
    vectorstore_result = vectorstore_task.get()

    return {
        "status": "success",
        "pdf_result": pdf_result,
        "transform_result": transform_result,
        "vectorstore_result": vectorstore_result
    }

class QueryRequest(BaseModel):
    question: str
    persist_directory: str = "./individual_chroma_db"
    top_k: int = 3

@app.post("/query-vectorstore/")
async def query_vectorstore(request: QueryRequest):
    """Endpoint to query the vectorstore and generate an answer."""
    
    # Query the vectorstore - specify queue='rag'
    query_task = query_vectorstore_task.apply_async(
        kwargs={
            "question": request.question,
            "persist_directory": request.persist_directory,
            "top_k": request.top_k
        },
        queue='rag'
    )
    
    return {
        "status": "started",
        "task_id": query_task.id,
        "message": "Query task started. Use /task-status/{task_id} to check progress."
    }

@app.post("/query-vectorstore-sync/")
async def query_vectorstore_sync(request: QueryRequest):
    """Endpoint to query the vectorstore and get immediate answer (blocking)."""
    
    # Query the vectorstore synchronously - specify queue='rag'
    query_task = query_vectorstore_task.apply_async(
        kwargs={
            "question": request.question,
            "persist_directory": request.persist_directory,
            "top_k": request.top_k
        },
        queue='rag'
    )
    
    result = query_task.get()  # Wait for completion
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return {
        "status": "success",
        "question": request.question,
        "answer": result["answer"],
        "retrieved_chunks": result["retrieved_chunks"],
        "sources": result["sources"]
    }

class SimpleRagChainRequest(BaseModel):
    pdf_path: str = "/app/context.pdf"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    persist_directory: str = "./simple_chroma_db"

@app.post("/simple-rag-chain/")
async def run_simple_rag_chain(request: SimpleRagChainRequest):
    """
    Endpoint to run the simple RAG chain using extract tasks.
    This uses the original approach with extractor tasks for compatibility.
    """
    # Create the RAG chain using original tasks with extractors
    rag_chain = chain(
        pdf_reader_task.s(request.pdf_path).set(queue='rag'),
        extract_documents_for_chain.s().set(queue='rag'),
        transform_documents_task.s(request.chunk_size, request.chunk_overlap).set(queue='rag'),
        extract_chunks_for_chain.s().set(queue='rag'),
        create_vectorstore_task.s(request.persist_directory).set(queue='rag')
    )
    
    # Execute the chain
    result = rag_chain.apply_async()
    
    return {
        "status": "started",
        "task_id": result.id,
        "message": "Simple RAG pipeline chain started. Use /task-status/{task_id} to check progress.",
        "pipeline_config": {
            "pdf_path": request.pdf_path,
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "persist_directory": request.persist_directory
        }
    }

class IndividualTasksRequest(BaseModel):
    pdf_path: str = "/app/context.pdf"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    persist_directory: str = "./individual_tasks_chroma_db"

@app.post("/rag-tasks-step-by-step/")
async def run_rag_tasks_step_by_step(request: IndividualTasksRequest):
    """
    Endpoint to run RAG tasks step by step with individual task monitoring.
    This approach runs each task sequentially and returns all task IDs.
    """
    
    try:
        # Step 1: PDF Reading
        pdf_task = pdf_reader_task.apply_async(args=[request.pdf_path], queue='rag')
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
            args=[pdf_result["documents"], request.chunk_size, request.chunk_overlap], 
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
            args=[transform_result["chunks"], request.persist_directory],
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
                "pdf_path": request.pdf_path,
                "chunk_size": request.chunk_size,
                "chunk_overlap": request.chunk_overlap,
                "persist_directory": request.persist_directory
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
