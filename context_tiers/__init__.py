"""context-tiers. Tiered token budget allocation for LLM agent context.

Public API:
    ContextManager   the facade most users want
    ContextStore     in-memory state store
    RedisContextStore  Redis-backed state store
    allocate, build_context  the underlying pieces, importable directly
"""

__version__ = "0.1.0"

from .manager import ContextManager, DEFAULT_SOURCES
from .store import ContextStore
from .redis_store import RedisContextStore
from .allocator import allocate
from .assembler import build_context, context_tokens
from .tokens import Item, count_tokens, count_item
from .pinner import HeuristicPinner, OllamaPinner
from .summarizer import HeuristicSummarizer, OllamaSummarizer

__all__ = [
    "ContextManager", "DEFAULT_SOURCES", "ContextStore", "RedisContextStore",
    "allocate", "build_context", "context_tokens", "Item", "count_tokens",
    "count_item", "HeuristicPinner", "OllamaPinner", "HeuristicSummarizer",
    "OllamaSummarizer",
]
