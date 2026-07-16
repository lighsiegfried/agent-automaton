"""Knowledge Vault API (Phase 6A).

    POST   /knowledge/documents/upload        GET    /knowledge/documents
    GET    /knowledge/documents/{id}          POST   /knowledge/documents/{id}/reindex
    POST   /knowledge/documents/{id}/archive  DELETE /knowledge/documents/{id}/prepare
    DELETE /knowledge/documents/{id}/confirm  POST   /knowledge/search
    POST   /knowledge/ask                     GET    /knowledge/status   GET /knowledge/audit

Uploads validate format/size before any processing; deletion is a two-step
prepare→confirm routed through the shared broker (a wrong phrase, plain "sí", or wake
never confirms). Answers are citation-grounded; nothing is written to memory here.
"""

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core import conversation
from app.core.nl import ServiceCommand
from app.core.pending import get_pending_broker
from app.knowledge.service import get_knowledge_service

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class SearchRequest(BaseModel):
    query: str = Field(default="")
    collection: str = Field(default="")


class AskRequest(BaseModel):
    question: str = Field(default="")
    collection: str = Field(default="")


class PhraseRequest(BaseModel):
    phrase: str = Field(default="")


def _svc():
    return get_knowledge_service()


def _ctx():
    return "es", get_settings(), get_pending_broker()


@router.post("/documents/upload")
async def upload(file: UploadFile = File(...), collections: str = Form(""),
                 tags: str = Form("")) -> dict[str, Any]:
    data = await file.read()
    lang, settings, broker = _ctx()
    args = {"data": data, "filename": file.filename or "document",
            "collections": [c.strip() for c in collections.split(",") if c.strip()],
            "tags": [t.strip() for t in tags.split(",") if t.strip()], "source": "upload"}
    return _svc().import_document(args, lang, settings, broker)


@router.get("/documents")
def list_documents(collection: str = "", include_archived: bool = False) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().list({"collection": collection or None, "include_archived": include_archived},
                       lang, settings, broker)


@router.get("/documents/{doc_id}")
def get_document(doc_id: str) -> dict[str, Any]:
    data = _svc().get(doc_id, get_settings())
    return {"document": data} if data else {"document": None, "error_code": "DOCUMENT_NOT_FOUND"}


@router.post("/documents/{doc_id}/reindex")
def reindex(doc_id: str) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().reindex({"doc_id": doc_id}, lang, settings, broker)


@router.post("/documents/{doc_id}/archive")
def archive(doc_id: str) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().archive({"doc_id": doc_id}, lang, settings, broker)


@router.delete("/documents/{doc_id}/prepare")
def delete_prepare(doc_id: str) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().prepare_delete({"doc_id": doc_id}, lang, settings, broker)


@router.delete("/documents/{doc_id}/confirm")
def delete_confirm(doc_id: str, request: PhraseRequest) -> dict[str, Any]:
    return conversation.dispatch(
        ServiceCommand(intent="confirm", is_confirmation=True,
                       confirmation_phrase=request.phrase or "confirmar eliminación de documento"),
        language="es")


@router.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().search({"query": request.query, "collection": request.collection or None},
                         lang, settings, broker)


@router.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().ask({"question": request.question, "collection": request.collection or None},
                      lang, settings, broker)


@router.get("/status")
def status() -> dict[str, Any]:
    lang, settings, broker = _ctx()
    return _svc().status({}, lang, settings, broker)


@router.get("/audit")
def audit(doc_id: str = "", limit: int = 100) -> dict[str, Any]:
    return {"events": _svc().audit(doc_id or None, limit=limit)}
