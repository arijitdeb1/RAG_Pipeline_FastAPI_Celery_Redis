
import os
import time
import logging
from celery import Celery

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configure Celery to use Redis as the message broker and results backend
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("tasks", broker=redis_url, backend=redis_url)

# Note: We rely on docker-compose --queues=default and explicit apply_async(queue='default') calls
# instead of automatic task routing

@celery_app.task
def process_data(data: str):
    """A long-running task to simulate processing data."""
    print(f"Starting to process data: {data}")
    time.sleep(20)  # Simulate a 10-second heavy task
    result = f"Processed: {data}"
    print(f"Finished processing. Resultt: {result}")
    return result

@celery_app.task
def add_numbers(x, y):
    """A simple task to demonstrate a computation."""
    result = x + y
    return result


@celery_app.task(bind=True, max_retries=5)
def divide_numbers(self, x, y):
    """A task that retries if a division by zero error occurs."""
    try:
        if y == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        result = x / y
        return result
    except ZeroDivisionError as exc:
        # Log the error and the retry attempt
        logger.warning(
            f"Retrying division for task {self.request.id} "
            f"on attempt {self.request.retries + 1}/{self.max_retries}."
        )
        # Re-queue the task with an exponential backoff
        # First retry after 2 seconds, second after 4, third after 8, etc.
        countdown = 2 ** self.request.retries
        raise self.retry(exc=exc, countdown=countdown)
    
@celery_app.task
def aggregate_results(results):
    """
    A callback task that aggregates the results from a group of tasks.
    The 'results' argument is automatically a list of the return values
    from the tasks in the group.
    """
    return sum(results)
