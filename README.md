**RAG pipelines** are everywhere now — powering chatbots, assistants, and enterprise search. 
Instead of revisiting what RAG is, let’s talk about how to make it production-ready using one
of the best approaches: leveraging **FastAPI**, **Celery**, and **Redis**, a trio that forms a 
distributed, asynchronous, and fault-tolerant architecture — unlike monolithic or sync-only setups,
this design scales horizontally and stays responsive under heavy workloads. And this is where 
distributed tracing becomes key: it gives you visibility across API calls, background tasks, 
and caching layers, ensuring smooth debugging and performance tuning.

_A prior knowledge or understanding of **Python**, **LangChain**, and **Docker** will be helpful
for setting up, customizing, and troubleshooting this pipeline._

A distributed trace is a collection of events, called "spans" that are linked together to show 
the end-to-end flow of a request.
Here’s a quick overview of the components:
- **FastAPI**: The web framework that handles incoming requests, creating a new trace for each.
- **Celery**: Celery is a distributed task queue system that allows you to run background tasks 
              asynchronously.
- **Redis**: An in-memory data structure store used as a message broker between **FastAPI**
           and **Celery**, and also as a caching layer.
- **Flower**: A web-based tool for monitoring and administrating Celery clusters.

## Architecture Overview
- The architecture consists of FastAPI (handling API requests), Celery (processing background tasks), Redis (serving as both a message broker and cache), and Flower (monitoring Celery workers). 
- **FastAPI** receives requests and delegates long-running or resource-intensive tasks to Celery via **Redis**. 
- **Celery** workers process these tasks in the background, so FastAPI can quickly respond to users without waiting for long-running operations to finish. Output from one task will be persisted in Redis cache for quick retrieval by subsequent tasks.Task status and results are also stored in Redis and can be queried by the FastAPI app.
- **Flower** provides real-time monitoring of task execution and worker status. 
- **Docker** is used to containerize and orchestrate all components for consistent deployment.
- A **Gradio** interface is included for easy interaction with the RAG pipeline.

![img.png](img.png)

## Step-by-step implementation
**Step 1: Install dependencies**
Refer `requirements.txt` for the full list of dependencies. Key packages include: celery[redis],fastapi, uvicorn

**Step 2: Define Celery tasks**
Lets' first evaluate the main components using some basic tasks:
- `app/worker.py`  - this file sets up the Celery app and configures it to use Redis as the message broker. 
- It also defines a sample task `add_numbers`and `divide_numbers` that adds two numbers and divides two numbers respectively.

**Step 3: Create FastAPI app**
Let's create a FastAPI app that will send above tasks to Celery worker and retrieve results.
- `app/main.py` - this file sets up the FastAPI app and defines endpoints to trigger Celery tasks.
- `@app.post("/add/{x}/{y}")` - endpoint to add two numbers using Celery task.
- `@app.post("/divide/{x}/{y}")` - endpoint to divide two numbers using Celery task.
- `@app.get("/task-status/{task_id}")` - endpoint to get the result of a Celery task using its task ID.

**Step 4: Dockerize the application**
Refer `Dockerfile` and `docker-compose.yml` for containerizing the FastAPI app, Celery worker, Redis, and Flower.
- `Dockerfile` - defines the Docker image for the FastAPI app and Celery worker.
- Register a `celery-worker` service in `docker-compose.yml` which will start a Celery worker process using the `worker.py` module and will listen to 'default' queue from Redis broker.
- Register a `flower` service in `docker-compose.yml` which will start Flower to monitor Celery workers.
- Register a `redis` service in `docker-compose.yml` which will start a Redis server.
- Environment variables are used to configure Redis URL.

**Step 5: Run the application**
- Use `docker-compose up --build` to build and start all services.
- Keep the command running to see logs from FastAPI, Celery worker, and Flower.
- Access FastAPI at `http://localhost:8000`, Flower at `http://localhost:5555`.
- Review the logs or Flower to verify if expected tasks are registered to the worker.
- Execute `curl --location --request POST 'localhost:8000/add/5/10'` to trigger the add task and add two numbers(5, 10).
- Result will be returned immediately with a task ID and initial status of `Task submitted`.
- Monitor the task status using `curl --location --request GET 'localhost:8000/task-status/<task_id>'` until it shows `COMPLETED` and the result.
- Keep an eye on Flower dashboard to see task progress and worker status in real-time.

**Step 6: Extend to RAG pipeline**
Now that the basic setup is working, you can extend the Celery tasks to implement the RAG pipeline:
- `app/rag/rag_worker.py` - define Celery tasks for each step of the RAG pipeline: PDF reading, text splitting, vectorstore(ChormaDB) creation.
- Register the rag_worker service in `docker-compose.yml` which will start a Celery worker process using the `rag_worker.py` module and will listen to 'rag' queue from Redis broker.
- Execute following  `curl --location --request POST 'localhost:8000/rag-individual-tasks/?filename=world_population_info.pdf'` to trigger the RAG pipeline tasks sequentially i.e output of one task is input to the next task.

**Step 7: Add Gradio interface**
- `gradio_app.py` - create a Gradio interface to interact with the RAG pipeline.
- Run the Gradio app using `python gradio_app.py` and access it at `http://localhost:7860`.
- Gradio app will require OpenAI API key.
- User can upload a PDF file and trigger the RAG pipeline.
- Display task status and results in the Gradio interface.
- Once the RAG pipeline tasks are completed, user can ask questions based on the ingested PDF content.

## Deep dive into RAG tasks and overall flow
1. Configure a Celery app in `rag_worker.py` to use Redis as the message broker.
2. Define individual Celery tasks for each step of the RAG pipeline:
   - `pdf_reader_task`: Reads and extracts text from the uploaded PDF file using PyMuPDF.
   - `text_splitter_task`: Splits the extracted text into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
   - `vectorstore_task`: Creates a Chroma vectorstore from the text chunks using OpenAI embeddings.
3. `celery_app.conf.update` is used to configure Celery settings like serialization format, timezone, task result expiration,etc.

4. Now, How to execute the tasks?
   - `app/rag/rag_utils.py` defines multiple ways to execute the tasks. Lets' understand them one by one:
   - Function `run_rag_tasks_step_by_step`: Executes the tasks sequentially, where the output of one task is passed as input to the next. 
   - This is done using `apply_async()` method to send each task to the Celery worker and then using `get()` method to wait for the result before proceeding to the next task. Use `apply_async` when you want to execute tasks asynchronously and get a task ID immediately.
   - `apply_async()` accepts both positional and keyword arguments, plus additional options. Like passing `queue='rag'` to specify the queue.
   - `delay()`: A shortcut to call a task asynchronously with default options. It is equivalent to `apply_async()` without any extra options. Use `delay` when you want a quick and simple way to execute a task asynchronously without needing to customize options.
   

   - Function `run_rag_tasks_in_chain`: Demonstrates how to use Celery chains to link tasks together so that the output of one task is automatically passed as input to the next task and sequentially executed.
   - Use shorthand `.s()` to create signatures for each task in the chain i.e. pass arguments to corresponding task.
   - After creating the chain, call `apply_async()` on the entire chain to send it to the Celery worker for execution.
   - Use `chain` when you have a series of tasks that need to be executed in a specific order, where each task depends on the result of the previous one.
   - Use `group` when you have multiple tasks that can run independently and you want to execute them in parallel, collecting all their results.
   - Use `chord` to run above group of tasks in parallel and then execute a final callback task once all tasks in the group are complete, using their combined results.

5. Configure Celery workers to listen to different queues:
   - In `docker-compose.yml`, we define two Celery worker services: `celery-worker` listens to the 'default' queue for basic tasks, while `rag_worker` listens to the 'rag' queue for RAG-specific tasks.
   - This separation allows us to allocate resources and manage workloads effectively, ensuring that long-running RAG tasks do not block simpler tasks.
   - Both workers are configured to use the same Redis broker for communication.
   - Monitor the workers and tasks using Flower at `http://localhost:5555`.
   - Using different queues helps in prioritizing tasks and scaling specific parts of the application as needed.Configure Celery workers to listen to different queues for better task management.

6. How to trigger the workflow?
   - We can trigger the RAG tasks using FastAPI endpoints defined in `main.py` or Gradio interface in `gradio_app.py`.

