from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "zhiyuan",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,
    task_soft_time_limit=1740,
    broker_connection_retry_on_startup=True,
)

from app.workers import tasks  # noqa: E402,F401
