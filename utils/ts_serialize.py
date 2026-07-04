_PLACEHOLDER = "<ts><ts/>"


def fill_ts_placeholders(input_text: str, input_ts: list, precision: int = 4) -> str:
    """Replace each <ts><ts/> placeholder with its serialized float array.

    input_ts: list[list[float]] — one array per placeholder, in order.
    If input_text has no placeholders, returns it unchanged (safe no-op).
    """
    if _PLACEHOLDER not in input_text:
        return input_text
    parts = input_text.split(_PLACEHOLDER)
    result = parts[0]
    for i, arr in enumerate(input_ts):
        serialized = "[" + ", ".join(f"{v:.{precision}f}" for v in arr) + "]"
        result += serialized + (parts[i + 1] if i + 1 < len(parts) else "")
    return result
