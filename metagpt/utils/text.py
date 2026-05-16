from typing import Generator, Sequence

from metagpt.utils.token_counter import TOKEN_MAX, count_output_tokens


def reduce_message_length(
    msgs: Generator[str, None, None],
    model_name: str,
    system_text: str,
    reserved: int = 0,
) -> str:
    """Reduce the length of concatenated message segments to fit within the maximum token size.

    Args:
        msgs: A generator of strings representing progressively shorter valid prompts.
        model_name: The name of the encoding to use. (e.g., "gpt-3.5-turbo")
        system_text: The system prompts.
        reserved: The number of reserved tokens.

    Returns:
        The concatenated message segments reduced to fit within the maximum token size.

    Raises:
        RuntimeError: If it fails to reduce the concatenated message length.
    """
    max_token = TOKEN_MAX.get(model_name, 2048) - count_output_tokens(system_text, model_name) - reserved
    for msg in msgs:
        if model_name not in TOKEN_MAX:
            print(f"[weighted_reduce]❗❗ Can't find {model_name}")

        if count_output_tokens(msg, model_name) < max_token or model_name not in TOKEN_MAX:
            return msg

    raise RuntimeError("fail to reduce message length")

from typing import List

import tiktoken


# ──────────────────────────────────────────────
#  TOKEN COUNTING
# ──────────────────────────────────────────────

def count_tokens(string: str, model: str = "deepseek-chat") -> int:
    """
    Count tokens in a string.
    DeepSeek uses the same tokenizer as GPT (cl100k_base).
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


def chars_to_delete(text: str, max_tokens: int, model: str = "deepseek-chat") -> int:
    """
    Returns how many characters must be removed so that
    count_tokens(text) <= max_tokens.

    Returns 0 if already within limit.
    """
    current_tokens = count_tokens(text, model)
    overflow = current_tokens - max_tokens
    if overflow <= 0:
        return 0

    # Approximate: chars_per_token from the actual text
    total_chars = len(text)
    chars_per_token = total_chars / max(current_tokens, 1)
    return int(overflow * chars_per_token)


# ──────────────────────────────────────────────
#  WEIGHTED TRUNCATION
# ──────────────────────────────────────────────

def truncate_results(
    research_results: list[str],
    chars_to_truncate: int,
) -> list[str]:
    """
    Distribute chars_to_truncate across research_results
    proportionally by each item's character length.

    Each item is truncated from the end by:
        int(weight * chars_to_truncate)
    where weight = len(item) / total_chars.

    Items that would be fully consumed are set to "".
    """
    if chars_to_truncate <= 0:
        return research_results

    total_chars = sum(len(r) for r in research_results)
    if total_chars == 0:
        return research_results

    truncated = []
    for r in research_results:
        weight    = len(r) / total_chars
        cut       = int(weight * chars_to_truncate)
        new_len   = max(0, len(r) - cut)
        truncated.append(r[:new_len])

    return truncated


# ──────────────────────────────────────────────
#  COMBINED HELPER
# ──────────────────────────────────────────────

def fit_to_token_limit(
    research_results: list[str],
    max_tokens: int,
    model: str = "deepseek-chat",
) -> list[str]:
    """
    End-to-end: given a list of strings and a token budget,
    return the list truncated to fit within max_tokens.

    Usage:
        results = fit_to_token_limit(research_results, max_tokens=4096)
    """
    combined = "\n".join(research_results)
    cut_chars = chars_to_delete(combined, max_tokens, model)

    if cut_chars == 0:
        return research_results

    print(f"  Over budget by ~{cut_chars} chars — distributing cuts...")
    return truncate_results(research_results, cut_chars)

def weighted_reduce(
    prompt : str,
    research_results : List[str],
    model_name: str,
    system_text: str,
    decomposition_nums,
    reserved: int = 0):
    
    if model_name not in TOKEN_MAX:
        print(f"[weighted_reduce]❗❗ Can't find {model_name}")

    return prompt.format(
        decomposition_nums=decomposition_nums, 
        search_results=fit_to_token_limit(
            research_results,
            max_tokens=TOKEN_MAX.get(model_name, 2048) - count_output_tokens(system_text, model_name) - reserved))
       

def generate_prompt_chunk(
    text: str,
    prompt_template: str,
    model_name: str,
    system_text: str,
    reserved: int = 0,
) -> Generator[str, None, None]:
    """Split the text into chunks of a maximum token size.

    Args:
        text: The text to split.
        prompt_template: The template for the prompt, containing a single `{}` placeholder. For example, "### Reference\n{}".
        model_name: The name of the encoding to use. (e.g., "gpt-3.5-turbo")
        system_text: The system prompts.
        reserved: The number of reserved tokens.

    Yields:
        The chunk of text.
    """
    paragraphs = text.splitlines(keepends=True)
    current_token = 0
    current_lines = []

    reserved = reserved + count_output_tokens(prompt_template + system_text, model_name)
    # 100 is a magic number to ensure the maximum context length is not exceeded
    max_token = TOKEN_MAX.get(model_name, 2048) - reserved - 100

    while paragraphs:
        paragraph = paragraphs.pop(0)
        token = count_output_tokens(paragraph, model_name)
        if current_token + token <= max_token:
            current_lines.append(paragraph)
            current_token += token
        elif token > max_token:
            paragraphs = split_paragraph(paragraph) + paragraphs
            continue
        else:
            yield prompt_template.format("".join(current_lines))
            current_lines = [paragraph]
            current_token = token

    if current_lines:
        yield prompt_template.format("".join(current_lines))


def split_paragraph(paragraph: str, sep: str = ".,", count: int = 2) -> list[str]:
    """Split a paragraph into multiple parts.

    Args:
        paragraph: The paragraph to split.
        sep: The separator character.
        count: The number of parts to split the paragraph into.

    Returns:
        A list of split parts of the paragraph.
    """
    for i in sep:
        sentences = list(_split_text_with_ends(paragraph, i))
        if len(sentences) <= 1:
            continue
        ret = ["".join(j) for j in _split_by_count(sentences, count)]
        return ret
    return list(_split_by_count(paragraph, count))


def decode_unicode_escape(text: str) -> str:
    """Decode a text with unicode escape sequences.

    Args:
        text: The text to decode.

    Returns:
        The decoded text.
    """
    return text.encode("utf-8").decode("unicode_escape", "ignore")


def _split_by_count(lst: Sequence, count: int):
    avg = len(lst) // count
    remainder = len(lst) % count
    start = 0
    for i in range(count):
        end = start + avg + (1 if i < remainder else 0)
        yield lst[start:end]
        start = end


def _split_text_with_ends(text: str, sep: str = "."):
    parts = []
    for i in text:
        parts.append(i)
        if i == sep:
            yield "".join(parts)
            parts = []
    if parts:
        yield "".join(parts)
