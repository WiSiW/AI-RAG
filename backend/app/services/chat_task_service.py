import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import redis
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import AppError, ConflictError, NotFoundError
from app.db.models import GenerationTask, Message
from app.db.session import SessionLocal
from app.schemas import ChatRequest, ChatTaskCreateRequest, GenerationTaskRead
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

TERMINAL_TASK_STATUSES = {"completed", "cancelled", "failed"}


class ChatTaskService:
    def __init__(
        self,
        chat_service: ChatService | None = None,
        redis_client: redis.Redis | None = None,
    ):
        self.chat_service = chat_service or ChatService()
        self.redis = redis_client or redis.from_url(settings.redis_url, decode_responses=True)

    @staticmethod
    def events_key(task_id: str) -> str:
        return f"zhiyuan:chat:task:{task_id}:events"

    @staticmethod
    def state_key(task_id: str) -> str:
        return f"zhiyuan:chat:task:{task_id}:state"

    @staticmethod
    def cancel_key(task_id: str) -> str:
        return f"zhiyuan:chat:task:{task_id}:cancel"

    @staticmethod
    def seq_key(task_id: str) -> str:
        return f"zhiyuan:chat:task:{task_id}:seq"

    def _require_redis(self) -> None:
        try:
            self.redis.ping()
        except Exception as exc:
            raise AppError(
                "任务式问答需要 Redis 和 Celery，当前任务服务不可用",
                status_code=503,
                code="chat_task_unavailable",
            ) from exc

    def _touch_redis_keys(self, task_id: str) -> None:
        ttl = settings.chat_task_event_ttl_seconds
        for key in (
            self.events_key(task_id),
            self.state_key(task_id),
            self.seq_key(task_id),
        ):
            self.redis.expire(key, ttl)

    def create_task(self, db: Session, request: ChatTaskCreateRequest) -> GenerationTask:
        self._require_redis()
        existing = db.scalar(
            select(GenerationTask).where(
                GenerationTask.client_request_id == request.client_request_id
            )
        )
        if existing is not None:
            return existing

        knowledge_bases = self.chat_service._get_kbs(db, request.knowledge_base_ids)
        self.chat_service._validate_document(db, knowledge_bases, request.document_id)
        llm_profile = self.chat_service._llm_profile(
            db, knowledge_bases, request.llm_profile_id
        )
        conversation = self.chat_service.memory.get_or_create(
            db,
            kb_ids=request.knowledge_base_ids,
            conversation_id=request.conversation_id,
            llm_profile_id=llm_profile.id,
            first_query=request.query,
            document_id=request.document_id,
        )
        active = db.scalar(
            select(GenerationTask).where(
                GenerationTask.conversation_id == conversation.id,
                GenerationTask.status.in_(("queued", "running")),
            )
        )
        if active is not None:
            raise ConflictError(
                f"当前会话正在生成回答，活动任务：{active.id}",
                code="conversation_busy",
            )

        task_id = str(uuid.uuid4())
        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        request_payload = request.model_dump(exclude={"client_request_id"}, mode="json")
        task = GenerationTask(
            id=task_id,
            conversation_id=conversation.id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            client_request_id=request.client_request_id,
            request_payload=request_payload,
            status="queued",
            phase="retrieving",
        )
        db.add(task)
        self.chat_service.memory.start_exchange(
            db,
            conversation,
            question=request.query,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            generation_task_id=task_id,
            commit=False,
        )
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            existing = db.scalar(
                select(GenerationTask).where(
                    GenerationTask.client_request_id == request.client_request_id
                )
            )
            if existing is not None:
                return existing
            active = db.scalar(
                select(GenerationTask).where(
                    GenerationTask.conversation_id == conversation.id,
                    GenerationTask.status.in_(("queued", "running")),
                )
            )
            if active is not None:
                raise ConflictError(
                    f"当前会话正在生成回答，活动任务：{active.id}",
                    code="conversation_busy",
                ) from exc
            raise

        db.refresh(task)
        try:
            self.publish_event(
                task,
                "queued",
                {
                    "conversation_id": task.conversation_id,
                    "user_message_id": task.user_message_id,
                    "assistant_message_id": task.assistant_message_id,
                    "status": "queued",
                },
            )
            from app.workers.celery_app import celery_app

            async_result = celery_app.send_task("zhiyuan.run_chat_generation", args=[task.id])
            task.celery_task_id = getattr(async_result, "id", None)
            db.commit()
        except Exception as exc:
            logger.exception("failed to dispatch chat generation task task_id=%s", task.id)
            task.status = "failed"
            task.error = str(exc)
            task.finished_at = datetime.now(UTC)
            assistant = db.get(Message, task.assistant_message_id)
            if assistant is not None:
                assistant.status = "failed"
                assistant.usage = {**(assistant.usage or {}), "error": str(exc)}
            db.commit()
            try:
                self.publish_event(
                    task,
                    "error",
                    {"message": "生成任务投递失败", "status": "failed"},
                )
            except Exception:
                logger.exception("failed to publish dispatch error task_id=%s", task.id)
            raise AppError(
                "生成任务投递失败", status_code=503, code="chat_task_unavailable"
            ) from exc
        return task

    def get_task(self, db: Session, task_id: str) -> GenerationTask:
        task = db.get(GenerationTask, task_id)
        if task is None:
            raise NotFoundError("生成任务不存在")
        self._mark_stale_if_needed(db, task)
        return task

    def task_read(self, db: Session, task: GenerationTask) -> GenerationTaskRead:
        assistant = db.get(Message, task.assistant_message_id)
        state = self._read_state(task.id)
        if task.status in TERMINAL_TASK_STATUSES:
            content = assistant.content if assistant is not None else str(state.get("content") or "")
            citations = assistant.citations if assistant is not None else []
        else:
            content = str(state.get("content") or (assistant.content if assistant else ""))
            raw_citations = state.get("citations")
            if isinstance(raw_citations, str):
                try:
                    citations = json.loads(raw_citations)
                except json.JSONDecodeError:
                    citations = []
            else:
                citations = raw_citations if isinstance(raw_citations, list) else []
        return GenerationTaskRead(
            id=task.id,
            conversation_id=task.conversation_id,
            user_message_id=task.user_message_id,
            assistant_message_id=task.assistant_message_id,
            status=task.status,
            phase=task.phase or "retrieving",
            cancel_requested=bool(task.cancel_requested),
            last_event_id=state.get("last_event_id") or task.last_event_id,
            content=content,
            citations=citations,
            error=task.error,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
        )

    def cancel_task(self, db: Session, task_id: str) -> GenerationTask:
        task = self.get_task(db, task_id)
        if task.status in TERMINAL_TASK_STATUSES:
            return task
        task.cancel_requested = True
        self.redis.set(self.cancel_key(task.id), "1", ex=settings.chat_task_event_ttl_seconds)
        if task.status == "queued":
            task.status = "cancelled"
            task.finished_at = datetime.now(UTC)
            assistant = db.get(Message, task.assistant_message_id)
            if assistant is not None:
                assistant.status = "stopped"
                assistant.usage = {
                    **(assistant.usage or {}),
                    "stopped": True,
                    "stopped_stage": task.phase or "retrieving",
                }
        db.commit()
        if task.status == "cancelled":
            self.publish_event(
                task,
                "cancelled",
                {
                    "message_id": task.assistant_message_id,
                    "stopped_stage": task.phase or "retrieving",
                    "status": "cancelled",
                },
            )
        if task.celery_task_id:
            try:
                from app.workers.celery_app import celery_app

                celery_app.control.revoke(task.celery_task_id, terminate=False)
            except Exception:
                logger.warning("failed to revoke celery chat task task_id=%s", task.id, exc_info=True)
        return task

    def _read_state(self, task_id: str) -> dict[str, Any]:
        try:
            return self.redis.hgetall(self.state_key(task_id))
        except Exception:
            logger.warning("failed to read chat task state task_id=%s", task_id, exc_info=True)
            return {}

    def _mark_stale_if_needed(self, db: Session, task: GenerationTask) -> None:
        if task.status != "running" or task.heartbeat_at is None:
            return
        heartbeat = task.heartbeat_at
        if heartbeat.tzinfo is None:
            heartbeat = heartbeat.replace(tzinfo=UTC)
        if datetime.now(UTC) - heartbeat < timedelta(seconds=settings.chat_task_stale_seconds):
            return
        task.status = "failed"
        task.error = "生成任务心跳超时"
        task.finished_at = datetime.now(UTC)
        assistant = db.get(Message, task.assistant_message_id)
        if assistant is not None:
            assistant.status = "failed"
            assistant.usage = {**(assistant.usage or {}), "error": task.error}
        db.commit()
        try:
            self.publish_event(
                task,
                "error",
                {"message": task.error, "status": "failed"},
            )
        except Exception:
            logger.warning("failed to publish stale task error task_id=%s", task.id, exc_info=True)

    def publish_event(self, task: GenerationTask, event: str, data: dict[str, Any]) -> str:
        seq = int(self.redis.incr(self.seq_key(task.id)))
        payload = {
            "task_id": task.id,
            "conversation_id": task.conversation_id,
            "assistant_message_id": task.assistant_message_id,
            "seq": seq,
            **data,
        }
        stream_id = str(
            self.redis.xadd(
                self.events_key(task.id),
                {"event": event, "data": json.dumps(payload, ensure_ascii=False, default=str)},
                maxlen=settings.chat_task_stream_maxlen,
                approximate=True,
            )
        )
        task.last_event_id = stream_id
        current = self._read_state(task.id)
        content = str(current.get("content") or "")
        if event == "token":
            content += str(payload.get("content") or "")
        citations = current.get("citations")
        if event == "citations":
            citations = payload.get("citations") or []
        state = {
            "status": str(payload.get("status") or current.get("status") or task.status),
            "phase": str(payload.get("phase") or current.get("phase") or task.phase or "retrieving"),
            "content": content,
            "citations": json.dumps(citations or [], ensure_ascii=False),
            "last_event_id": stream_id,
            "error": str(payload.get("message") or current.get("error") or ""),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.redis.hset(self.state_key(task.id), mapping=state)
        self._touch_redis_keys(task.id)
        return stream_id

    async def run(self, task_id: str) -> None:
        db = SessionLocal()
        publisher = _TaskEventPublisher(self, task_id)
        heartbeat_task: asyncio.Task | None = None
        try:
            task = db.get(GenerationTask, task_id)
            if task is None or task.status in TERMINAL_TASK_STATUSES:
                return
            if task.cancel_requested or self.redis.exists(self.cancel_key(task_id)):
                self._finalize_cancelled_queued(db, task)
                return
            task.status = "running"
            task.started_at = datetime.now(UTC)
            task.heartbeat_at = datetime.now(UTC)
            assistant = db.get(Message, task.assistant_message_id)
            if assistant is not None:
                assistant.status = "streaming"
            db.commit()
            self._update_task_state(task, status="running", phase="retrieving")
            heartbeat_task = asyncio.create_task(self._heartbeat(task_id))

            payload = dict(task.request_payload or {})
            request = ChatRequest(
                **payload,
                conversation_id=task.conversation_id,
                stream=True,
            )

            def cancel_requested() -> bool:
                return bool(self.redis.exists(self.cancel_key(task_id)))

            terminal_status: str | None = None
            async for raw_event in self.chat_service.stream_chat(
                db,
                request,
                task_id=task.id,
                assistant_message_id=task.assistant_message_id,
                cancel_requested=cancel_requested,
            ):
                event, data = _parse_sse_event(raw_event)
                if event == "stage":
                    phase = str(data.get("stage") or task.phase or "retrieving")
                    task.phase = phase
                    task.heartbeat_at = datetime.now(UTC)
                    data = {**data, "phase": phase, "status": "running"}
                    db.commit()
                elif event in {"retrieval_plan", "retrieval_step", "retrieval_done"}:
                    task.heartbeat_at = datetime.now(UTC)
                    db.commit()
                if event == "done":
                    terminal_status = "completed"
                    data = {**data, "status": "completed"}
                elif event == "cancelled":
                    terminal_status = "cancelled"
                    data = {**data, "status": "cancelled"}
                elif event == "error":
                    terminal_status = "failed"
                    data = {**data, "status": "failed"}
                await publisher.publish(event, data)
                if terminal_status:
                    break
            await publisher.flush()
            if terminal_status is None:
                terminal_status = "cancelled" if cancel_requested() else "failed"
                await publisher.publish(
                    "cancelled" if terminal_status == "cancelled" else "error",
                    {
                        "message_id": task.assistant_message_id,
                        "message": "生成任务异常结束" if terminal_status == "failed" else "",
                        "status": terminal_status,
                    },
                )
            self._finish_task(db, task, terminal_status)
        except Exception as exc:
            logger.exception("chat generation task failed task_id=%s", task_id)
            db.rollback()
            task = db.get(GenerationTask, task_id)
            if task is not None and task.status not in TERMINAL_TASK_STATUSES:
                task.status = "failed"
                task.error = str(exc)
                task.finished_at = datetime.now(UTC)
                assistant = db.get(Message, task.assistant_message_id)
                if assistant is not None:
                    assistant.status = "failed"
                    assistant.usage = {**(assistant.usage or {}), "error": str(exc)}
                db.commit()
                try:
                    await publisher.publish(
                        "error", {"message": str(exc), "status": "failed"}
                    )
                    await publisher.flush()
                except Exception:
                    logger.exception("failed to publish chat task failure task_id=%s", task_id)
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
            db.close()

    async def _heartbeat(self, task_id: str) -> None:
        while True:
            await asyncio.sleep(10)
            db = SessionLocal()
            try:
                task = db.get(GenerationTask, task_id)
                if task is None or task.status not in {"queued", "running"}:
                    return
                task.heartbeat_at = datetime.now(UTC)
                db.commit()
            finally:
                db.close()

    def _finalize_cancelled_queued(self, db: Session, task: GenerationTask) -> None:
        task.status = "cancelled"
        task.finished_at = datetime.now(UTC)
        assistant = db.get(Message, task.assistant_message_id)
        if assistant is not None:
            assistant.status = "stopped"
            assistant.usage = {
                **(assistant.usage or {}),
                "stopped": True,
                "stopped_stage": task.phase or "retrieving",
            }
        db.commit()
        self.publish_event(
            task,
            "cancelled",
            {"message_id": task.assistant_message_id, "status": "cancelled"},
        )

    def _finish_task(self, db: Session, task: GenerationTask, status: str) -> None:
        task.status = status
        task.finished_at = datetime.now(UTC)
        task.heartbeat_at = datetime.now(UTC)
        if status == "failed" and not task.error:
            task.error = "生成回答失败"
        db.commit()
        self._update_task_state(task, status=status, phase=task.phase or "retrieving")

    def _update_task_state(self, task: GenerationTask, *, status: str, phase: str) -> None:
        try:
            self.redis.hset(
                self.state_key(task.id),
                mapping={"status": status, "phase": phase},
            )
            self._touch_redis_keys(task.id)
        except Exception:
            logger.warning("failed to update chat task state task_id=%s", task.id, exc_info=True)

    async def subscribe(self, db: Session, task_id: str, after: str | None) -> AsyncIterator[str]:
        task = self.get_task(db, task_id)
        last_id = after or "0-0"
        stream_key = self.events_key(task_id)
        if task.status in TERMINAL_TASK_STATUSES:
            stream_exists = bool(self.redis.exists(stream_key))
            if not stream_exists or (task.last_event_id and last_id == task.last_event_id):
                yield _sse_with_id(
                    task.last_event_id or "0-0",
                    "snapshot",
                    self.task_read(db, task).model_dump(mode="json"),
                )
                return
        emitted = False
        while True:
            entries = await asyncio.to_thread(
                self.redis.xread,
                {stream_key: last_id},
                count=100,
                block=settings.chat_task_sse_heartbeat_seconds * 1000,
            )
            if not entries:
                yield ": ping\n\n"
                fresh = self.get_task(db, task_id)
                if fresh.status in TERMINAL_TASK_STATUSES:
                    if not emitted:
                        yield _sse_with_id(
                            fresh.last_event_id or "0-0",
                            "snapshot",
                            self.task_read(db, fresh).model_dump(mode="json"),
                        )
                    return
                continue
            for _, rows in entries:
                for stream_id, fields in rows:
                    last_id = stream_id
                    emitted = True
                    event = str(fields.get("event") or "message")
                    data = str(fields.get("data") or "{}")
                    yield f"id: {stream_id}\nevent: {event}\ndata: {data}\n\n"
                    if event in {"done", "error", "cancelled"}:
                        return


class _TaskEventPublisher:
    def __init__(self, service: ChatTaskService, task_id: str):
        self.service = service
        self.task_id = task_id
        self.buffer: list[str] = []
        self.last_flush = time.monotonic()

    async def publish(self, event: str, data: dict[str, Any]) -> None:
        if event != "token":
            await self.flush()
            await asyncio.to_thread(self._publish_sync, event, data)
            return
        self.buffer.append(str(data.get("content") or ""))
        elapsed_ms = (time.monotonic() - self.last_flush) * 1000
        if elapsed_ms >= settings.chat_task_token_flush_ms:
            await self.flush()

    async def flush(self) -> None:
        if not self.buffer:
            return
        content = "".join(self.buffer)
        self.buffer.clear()
        self.last_flush = time.monotonic()
        await asyncio.to_thread(self._publish_sync, "token", {"content": content, "status": "running"})

    def _publish_sync(self, event: str, data: dict[str, Any]) -> None:
        db = SessionLocal()
        try:
            task = db.get(GenerationTask, self.task_id)
            if task is None:
                return
            task.heartbeat_at = datetime.now(UTC)
            db.commit()
            self.service.publish_event(task, event, data)
            db.commit()
        finally:
            db.close()


def _parse_sse_event(raw_event: str) -> tuple[str, dict[str, Any]]:
    event = "message"
    data_lines: list[str] = []
    for line in raw_event.splitlines():
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    payload = json.loads("\n".join(data_lines) or "{}")
    return event, payload if isinstance(payload, dict) else {}


def _sse_with_id(event_id: str, event: str, data: dict[str, Any]) -> str:
    return (
        f"id: {event_id}\n"
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
    )
