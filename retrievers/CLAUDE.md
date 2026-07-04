# retrievers/

Demonstration retrieval components for DR-TSR. Every retriever subclasses
`BaseRetriever` and exposes a two-method contract: build an index from a pool
of candidate demonstrations, then retrieve the best k for any given query.

---

## Interface contract

```python
class BaseRetriever(ABC):
    def index(self, pool: List[Dict[str, Any]]) -> None: ...
    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]: ...
```

**`index(pool)`**  
Build whatever internal data structure (embedding matrix, FAISS index, DTW cache,
etc.) the retriever needs. Called once per experiment run before evaluation starts.
`pool` must be disjoint from the test set — callers are responsible for the split.

**`retrieve(query, k)`**  
Return up to `k` items from the indexed pool, ranked best-first. Must never return
the query item itself (match on `item["id"]` if available).

---

## Batch-contract dict fields

Every item passed to `index()` or returned from `retrieve()` uses the same format
as dataset items (see `data_provider/CLAUDE.md`):

| Field | Type | Description |
|-------|------|-------------|
| `input_text` | `str` | Full prompt (TS serialized, or with `<ts><ts/>` placeholders) |
| `input_ts` | `list[list[float]]` | Raw float arrays, one per series |
| `output_text` | `str` | Gold option letter (`'A'`, `'B'`, …) |
| `task_id` | `str` | Dataset identifier |
| `options` | `list[str]` | Valid option letters |

Metadata fields available on TimeSeriesExam items: `id`, `category`,
`subcategory`, `tid` (template id), `difficulty`, `question`, `answer_text`.

---

## Where retrievers are used

1. **`run_exp.py:177`** — `retriever = None` placeholder; replace with an
   instantiated retriever to activate k-shot retrieval.
2. **`evaluations/timeseriesexam_eval.py`** — `evaluate_tse()` calls
   `retriever.retrieve(item, k=num_shots)` then passes the result to
   `build_icl_prompt()`.

Typical experiment setup:

```python
retriever = SomeRetriever(...)
retriever.index(pool_items)          # build index once
# run_exp.py then passes retriever → evaluate_tse → retrieve per query
```

---

## Adding a new retriever

1. Create `retrievers/<name>.py` subclassing `BaseRetriever`.
2. Implement `index()` and `retrieve()`.
3. Import it in `retrievers/__init__.py` and add it to `__all__`.
4. Wire it into `run_exp.py` (step 7, currently `retriever = None`).

A `utils/retriever.py` registry (mirroring `utils/model.py`) can be added once
there are multiple concrete retrievers to register.
