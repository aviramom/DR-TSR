"""Prompt builder for Experiment 1 retrieval demonstrations.

In this experiment each demonstration is a (full question + time series + answer) triple,
not a bare (series, label) pair as in the legacy UCR format. Demonstrations may come from
completely different templates than the query.

TS is subsampled to ts_max_len points before embedding in the prompt to keep token counts
manageable — the official benchmark has ts_length=1024 which is too long for most LLMs.
"""

_FORMAT_HINT = (
    "Please answer the question and provide the correct option letter, "
    "e.g., A), B), C), D), and option content at the end of your answer. "
    "All information need to answer the question is given. "
    "If you are unsure, please provide your best guess."
)


def _subsample(ts: list, max_len: int) -> list:
    """Evenly subsample a TS to at most max_len points."""
    if len(ts) <= max_len:
        return ts
    step = len(ts) / max_len
    return [ts[int(i * step)] for i in range(max_len)]


def _ts_to_str(ts: list, max_len: int) -> str:
    pts = _subsample(ts, max_len)
    return "[" + ", ".join(f"{x:.4f}" for x in pts) + "]"


def _options_block(options: list) -> str:
    return "\n".join(f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options))


def build_retrieval_prompt(
    query_item,
    demonstrations: list,
    ts_max_len: int = 128,
) -> tuple:
    """Build the full ICL prompt for one query with pre-selected demonstrations.

    Each demonstration shows: question + options + TS + correct answer.
    The query shows:          question + options + TS + format_hint.

    Args:
        query_item: TSEItem for the test query
        demonstrations: ordered list of TSEItem pool items to use as context
        ts_max_len: subsample TS to this many points in the prompt text

    Returns:
        (prompt_str, option_letters)
        option_letters: e.g. ["A", "B", "C"] — used by _extract_predicted_label
    """
    parts = []

    if demonstrations:
        parts.append("Here are some labeled examples:\n")
        for i, demo in enumerate(demonstrations):
            if demo.is_two_series:
                ts_block = (
                    f"Time Series 1: {_ts_to_str(demo.ts1, ts_max_len)}\n"
                    f"Time Series 2: {_ts_to_str(demo.ts2, ts_max_len)}"
                )
            else:
                ts_block = f"Time Series: {_ts_to_str(demo.ts, ts_max_len)}"
            parts.append(
                f"Example {i + 1}: {demo.question}\n"
                f"{_options_block(demo.options)}\n"
                f"{ts_block}\n"
                f"Answer: {demo.answer_letter}) {demo.answer_text}\n"
            )
        parts.append("Now answer the following:\n")

    # Query section
    if query_item.is_two_series:
        ts_block = (
            f"Time Series 1: {_ts_to_str(query_item.ts1, ts_max_len)}\n"
            f"Time Series 2: {_ts_to_str(query_item.ts2, ts_max_len)}"
        )
    else:
        ts_block = f"Time Series: {_ts_to_str(query_item.ts, ts_max_len)}"

    parts.append(
        f"{query_item.question}\n"
        f"{_options_block(query_item.options)}\n"
        f"{ts_block}\n\n"
        f"{_FORMAT_HINT}\n"
    )

    return "\n".join(parts), query_item.option_letters
