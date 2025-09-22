import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)
task_id = '1df0cdf6-f373-4b13-8728-465b8659b2a2'
result_key = f'celery-task-meta-{task_id}'

task_result_raw = r.get(result_key)

if task_result_raw:
    # Decode the result from bytes to a UTF-8 string
    task_result_str = task_result_raw.decode('utf-8')

    # Convert the JSON string back into a Python dictionary
    task_data = json.loads(task_result_str)

    # Now you can validate specific fields in the data
    if task_data.get('status') == 'SUCCESS':
        print(f"Task {task_id} completed successfully.")
        print(f"Result: {task_data.get('result')}")
    else:
        print(f"Task {task_id} status: {task_data.get('status')}")
else:
    print(f"Task {task_id} data not found in Redis.")
