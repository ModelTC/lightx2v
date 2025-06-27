import requests
from loguru import logger


response = requests.get("http://localhost:8000/v1/local/video/generate/service_status")
logger.info(response.json())


response = requests.get("http://localhost:8000/v1/local/video/generate/get_all_tasks")
logger.info(response.json())


response = requests.post("http://localhost:8000/v1/local/video/generate/task_status", json={"task_id": "test_task_001"})
logger.info(response.json())
