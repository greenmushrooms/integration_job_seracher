# llm-queue Python Client

Python client for the [llm-queue](../) async LLM task queue.

## Install

```bash
pip install -e /path/to/llm-queue/client
```

Or from the job_searcher project:
```bash
pip install -e ../llm-queue/client
```

## Usage

```python
import os
from llm_queue import LLMQueueClient

client = LLMQueueClient(
    dsn=os.environ["DATABASE_URL"],
    worker_url="http://localhost:8080",  # optional, for control API
)

# Push a single task
task_id = client.push_task("job_eval", {
    "job_id": 123,
    "description": "Senior Data Engineer...",
    "resume": "John Doe, 10 years experience...",
    "company": "Acme Corp",
    "title": "Senior Data Engineer",
})

# Wait for result (blocking)
result = client.wait_for_result(task_id, timeout=300)
print(result)  # {"verdict": "Lateral", "match_scores": {...}, ...}

# Push batch
task_ids = client.push_batch("job_eval", [
    {"job_id": j["id"], "description": j["desc"], "resume": resume}
    for j in jobs
])
results = client.wait_for_batch(task_ids, timeout=600)

# Non-blocking check
result = client.get_result(task_id)  # None if not done yet

# Control (requires worker_url or uses DB directly)
client.pause("job_eval")
client.resume("job_eval")
client.cancel_pending("job_eval")
status = client.status()

# Use as context manager
with LLMQueueClient(dsn=...) as client:
    task_id = client.push_task("job_eval", payload)
    result = client.wait_for_result(task_id)
```
