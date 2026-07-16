"""Local document Knowledge Vault with citation-grounded RAG (Phase 6A).

Fifi ingests explicitly authorized local documents (PDF/TXT/Markdown/DOCX/CSV) into a
private, owner-scoped vault, extracts their structure (pages/headings/sections/rows),
chunks deterministically, and generates embeddings with a LOCAL model — documents are
never sent to an external API. Retrieval is hybrid (SQLite FTS5 + local vector
similarity) and bounded; every document-derived answer cites filename + page/section
or states that the evidence is insufficient, and no citation is fabricated.

Document content is treated as UNTRUSTED data: in-document instructions ("ignore
previous rules", "reveal secrets", "run commands", "send email", "modify memory",
"open this link") are flagged and never acted upon. A document can provide
information, never authorization. Retrieved facts are never written to personal
memory automatically — "recuerda esto del documento" creates a normal memory proposal
that still needs an explicit memory confirmation. Deleting a document is sensitive and
requires an exact phrase through the global pending broker.

Importing this package registers the knowledge service's intents, NL patterns, and
pending-broker domain with the core dispatcher.
"""

from app.knowledge import service as _service  # noqa: F401  (import registers)
