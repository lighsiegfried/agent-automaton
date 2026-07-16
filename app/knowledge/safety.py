"""Format validation + untrusted-document boundary (Phase 6A, items 2 & 5).

Two guarantees:
1. Only supported document formats are accepted; executables, scripts, archives (a
   bare .zip/.7z/.rar), encrypted files, and anything mismatching its magic bytes are
   rejected before any processing, within configured size/page limits.
2. Document content is UNTRUSTED. ``detect_injection`` flags in-document attempts to
   change rules, reveal secrets, run commands, send messages, modify memory, or open
   links — these are recorded and shown, but NEVER acted upon. A document can provide
   information, never authorization.

Pure and stdlib-only, so the whole gate is testable without any file libraries.
"""

from __future__ import annotations

import re

from app.knowledge.models import (
    FORMAT_CSV, FORMAT_DOCX, FORMAT_MD, FORMAT_PDF, FORMAT_TXT, SUPPORTED_FORMATS,
)

_EXT_FORMAT = {
    "pdf": FORMAT_PDF, "txt": FORMAT_TXT, "text": FORMAT_TXT, "md": FORMAT_MD,
    "markdown": FORMAT_MD, "docx": FORMAT_DOCX, "csv": FORMAT_CSV,
}

# Magic-byte signatures we explicitly REJECT (disguised executables/archives).
_REJECT_MAGIC = [
    (b"MZ", "executable"), (b"\x7fELF", "executable"), (b"\xca\xfe\xba\xbe", "executable"),
    (b"7z\xbc\xaf\x27\x1c", "archive"), (b"Rar!", "archive"), (b"\x1f\x8b", "archive"),
    (b"BZh", "archive"), (b"\xfd7zXZ", "archive"),
]

# Deletion confirmation phrases (item 9). A plain "sí"/wake never confirms.
DELETE_PHRASES = frozenset({
    "confirmar eliminación de documento", "confirmar eliminacion de documento",
    "delete document", "eliminar documento",
})


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def is_delete_phrase(text: str) -> bool:
    return normalize(text) in DELETE_PHRASES


def _ext(filename: str) -> str:
    return (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""


def validate_upload(filename: str, data: bytes, *, max_file_mb: int) -> dict:
    """(ok, format, error_code, reason). Rejects unsupported/dangerous files up front."""
    from app.core import errors

    ext = _ext(filename)
    fmt = _EXT_FORMAT.get(ext)
    if fmt is None or fmt not in SUPPORTED_FORMATS:
        return {"ok": False, "format": None, "error_code": errors.UNSUPPORTED_FORMAT,
                "reason": f"unsupported file type: .{ext or '?'}"}
    if not data:
        return {"ok": False, "format": fmt, "error_code": errors.EMPTY_DOCUMENT,
                "reason": "the file is empty"}
    if len(data) > max_file_mb * 1024 * 1024:
        return {"ok": False, "format": fmt, "error_code": errors.FILE_TOO_LARGE,
                "reason": f"file exceeds {max_file_mb} MB"}

    head = data[:16]
    for magic, kind in _REJECT_MAGIC:
        if head.startswith(magic):
            return {"ok": False, "format": fmt, "error_code": errors.UNSUPPORTED_FORMAT,
                    "reason": f"rejected {kind} file"}
    # PK\x03\x04 is a zip container — allowed ONLY as .docx, never as a bare archive.
    if head.startswith(b"PK\x03\x04") and fmt != FORMAT_DOCX:
        return {"ok": False, "format": fmt, "error_code": errors.UNSUPPORTED_FORMAT,
                "reason": "rejected archive file"}
    if fmt == FORMAT_PDF and not head.startswith(b"%PDF"):
        return {"ok": False, "format": fmt, "error_code": errors.UNSUPPORTED_FORMAT,
                "reason": "not a valid PDF"}
    if fmt == FORMAT_DOCX and not head.startswith(b"PK"):
        return {"ok": False, "format": fmt, "error_code": errors.UNSUPPORTED_FORMAT,
                "reason": "not a valid DOCX"}
    # Encrypted-PDF marker (a fuller check happens in the extractor).
    if fmt == FORMAT_PDF and b"/Encrypt" in data[:4096]:
        return {"ok": False, "format": fmt, "error_code": errors.ENCRYPTED_FILE,
                "reason": "the PDF is encrypted"}
    # Text formats must not be binary.
    if fmt in (FORMAT_TXT, FORMAT_MD, FORMAT_CSV) and b"\x00" in data[:4096]:
        return {"ok": False, "format": fmt, "error_code": errors.UNSUPPORTED_FORMAT,
                "reason": "binary content in a text file"}
    return {"ok": True, "format": fmt, "error_code": None, "reason": ""}


# --- untrusted-content injection detection -----------------------------------------

_INJECTION_PATTERNS = [
    ("override_rules", re.compile(r"\bignore\s+(?:all\s+|the\s+)?(?:previous|prior|above)\s+(?:rules?|instructions?)\b", re.I)),
    ("override_rules_es", re.compile(r"\bignora\s+(?:las\s+)?(?:reglas|instrucciones)\s+(?:anteriores|previas)\b", re.I)),
    ("reveal_secret", re.compile(r"\breveal\s+(?:your\s+)?(?:system\s+prompt|instructions?|secrets?|passwords?|tokens?)\b", re.I)),
    ("reveal_secret_es", re.compile(r"\b(?:revela|muestra)\s+(?:tus\s+)?(?:secretos?|contrase[nñ]as?|instrucciones)\b", re.I)),
    ("run_command", re.compile(r"\b(?:run|execute|ejecuta)\s+(?:this\s+)?(?:command|shell|script|comando)\b", re.I)),
    ("send_message", re.compile(r"\b(?:send|email|whatsapp|env[ií]a)\s+(?:this\s+|an?\s+)?(?:email|message|correo|mensaje)\b", re.I)),
    ("modify_memory", re.compile(r"\b(?:modify|update|delete|forget|change)\s+(?:the\s+)?(?:memory|memories)\b|\b(?:modifica|borra|olvida)\s+(?:la\s+)?memoria\b", re.I)),
    ("open_link", re.compile(r"\b(?:open|click|download|descarga|abre)\s+(?:this\s+|the\s+|el\s+|este\s+)?(?:link|url|attachment|enlace|archivo)\b", re.I)),
    ("you_are_now", re.compile(r"\byou\s+are\s+now\b|\bact\s+as\b|\bnew\s+instructions?\b|\bahora\s+eres\b", re.I)),
    ("disable_safety", re.compile(r"\b(?:disable|turn\s+off|bypass|desactiva)\s+(?:the\s+)?(?:safety|security|confirmation|seguridad)\b", re.I)),
]


def detect_injection(text: str) -> list[str]:
    """Flag likely in-document instructions. They are NEVER executed regardless —
    this only records what to show the user."""
    if not text:
        return []
    hits = []
    seen = set()
    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(text) and name not in seen:
            seen.add(name)
            hits.append(name)
    return hits[:8]
