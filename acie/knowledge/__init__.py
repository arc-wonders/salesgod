"""Product-agnostic knowledge base: schema, store, deterministic retrieval. See docs/KNOWLEDGE_TREE.md."""

from .retrieve import Hit, RetrievalResult, retrieve  # noqa: F401
from .schema import Entry, InterestTag, KnowledgeBase, Offering, Topic  # noqa: F401
from .store import kb_enabled, load_kb  # noqa: F401
