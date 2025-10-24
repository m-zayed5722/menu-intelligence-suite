"""Worker queue setup for background jobs."""
import os

from redis import Redis
from rq import Queue

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Redis connection
redis_conn = Redis.from_url(REDIS_URL)

# Create queue
job_queue = Queue("mis-queue", connection=redis_conn)


def enqueue_job(func, *args, **kwargs):
    """Enqueue a background job."""
    job = job_queue.enqueue(func, *args, **kwargs)
    return job.id
