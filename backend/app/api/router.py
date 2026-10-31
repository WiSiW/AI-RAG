from fastapi import APIRouter

from app.api import (
    auth,
    chat,
    chunk_evaluations,
    conversations,
    documents,
    feedback,
    health,
    knowledge_base_transfers,
    knowledge_bases,
    model_profiles,
    parsing_engines,
    public_api,
    publications,
    system,
)

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(health.router)
api_router.include_router(system.router)
api_router.include_router(model_profiles.router)
api_router.include_router(parsing_engines.router)
api_router.include_router(knowledge_bases.router)
api_router.include_router(knowledge_base_transfers.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(chunk_evaluations.router)
api_router.include_router(conversations.router)
api_router.include_router(feedback.router)
api_router.include_router(publications.router)
api_router.include_router(public_api.router)
