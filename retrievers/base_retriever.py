from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseRetriever(ABC):
    """Abstract base class for all demonstration retrievers.

    Subclasses implement two methods:
      - index(pool)    — build the retrieval index once from a pool of items
      - retrieve(query, k) — return the k best demonstrations for a query

    All items (pool and query) follow the standard batch-contract dict
    format documented in retrievers/CLAUDE.md.
    """

    @abstractmethod
    def index(self, pool: List[Dict[str, Any]]) -> None:
        """Build the retrieval index from a pool of demonstration items.

        Called once before any retrieve() calls. The pool must contain only
        items that are safe to use as demonstrations (no test-set leakage).

        Args:
            pool: List of batch-contract dicts, each with at minimum:
                  input_text, input_ts, output_text, task_id, options.
        """

    @abstractmethod
    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """Return up to k demonstration items ranked by relevance to query.

        Args:
            query: A single batch-contract dict representing the test item.
            k:     Maximum number of demonstrations to return.

        Returns:
            List of at most k items from the indexed pool, best first.
            Must never include the query item itself.
        """
