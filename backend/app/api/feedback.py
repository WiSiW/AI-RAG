import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.security import get_current_admin
from app.db.models import Feedback, Message
from app.db.session import get_db
from app.schemas import FeedbackCreate, FeedbackRead

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackRead, status_code=201)
def create_feedback(
    request: FeedbackCreate,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> FeedbackRead:
    message = db.get(Message, request.message_id)
    if message is None or message.role != "assistant":
        raise NotFoundError("回答消息不存在")
    feedback = Feedback(
        id=str(uuid.uuid4()),
        message_id=request.message_id,
        rating=request.rating,
        comment=request.comment,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return FeedbackRead.model_validate(feedback)
