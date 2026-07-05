_PLACEHOLDER = "<ts><ts/>"

# TimeSeriesExam series are ~1024 points long. At full resolution + 4 decimals each
# series serializes to ~5.5k tokens, so a 3-shot ICL prompt (up to 8 series) overflows
# a text LLM's context window. We downsample to at most _MAX_POINTS and print 2 decimals
# — ~1k tokens/series — which keeps 3-shot prompts well under 32k. Applied uniformly
# across all k so retriever/shot comparisons stay on the same representation.
_MAX_POINTS = 256
_PRECISION = 2


def _downsample(arr: list, max_points: int) -> list:
    """Evenly stride an array down to at most max_points, keeping the last point."""
    n = len(arr)
    if max_points <= 0 or n <= max_points:
        return arr
    step = (n + max_points - 1) // max_points  # ceil(n / max_points)
    sampled = arr[::step]
    if sampled[-1] is not arr[-1]:
        sampled = sampled + [arr[-1]]
    return sampled


def fill_ts_placeholders(
    input_text: str,
    input_ts: list,
    precision: int = _PRECISION,
    max_points: int = _MAX_POINTS,
) -> str:
    """Replace each <ts><ts/> placeholder with its serialized float array.

    input_ts: list[list[float]] — one array per placeholder, in order.
    Each series is downsampled to at most `max_points` and printed at `precision`
    decimals to keep serialized prompts within the LLM context window.
    If input_text has no placeholders, returns it unchanged (safe no-op).
    """
    if _PLACEHOLDER not in input_text:
        return input_text
    parts = input_text.split(_PLACEHOLDER)
    result = parts[0]
    for i, arr in enumerate(input_ts):
        arr = _downsample(arr, max_points)
        serialized = "[" + ", ".join(f"{v:.{precision}f}" for v in arr) + "]"
        result += serialized + (parts[i + 1] if i + 1 < len(parts) else "")
    return result
