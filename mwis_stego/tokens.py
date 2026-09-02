"""Byte-level view of a tokenizer's vocabulary.

Segmentation ambiguity is a property of *bytes*, not of characters.  A
byte-level BPE vocabulary such as Qwen's routinely splits one CJK character
across several tokens (``混`` -> ``b' \\xe6'``, ``b'\\xb7'``, ``b'\\xb7'``), so a
token's decoded ``str`` may be a lone replacement character and comparing those
with ``startswith`` is meaningless.  Every prefix test in this package is
therefore done on the raw bytes a token contributes to the output stream.
"""

from __future__ import annotations

from functools import lru_cache

__all__ = ["bytes_to_unicode", "ByteVocab"]


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte<->unicode table that byte-level BPE vocabularies are written in."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class ByteVocab:
    """Maps token ids to the exact bytes they emit."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        u2b = {v: k for k, v in bytes_to_unicode().items()}
        added = set(tokenizer.get_added_vocab().values())
        size = len(tokenizer)
        table: list[bytes] = [b""] * size
        for i in range(size):
            tok = tokenizer.convert_ids_to_tokens(i)
            if tok is None:
                continue
            if i in added:
                # Control tokens such as <|im_end|> are literal text, not byte-encoded.
                table[i] = tok.encode("utf-8")
            else:
                table[i] = bytes(u2b[c] for c in tok)
        self.table = table
        self.added = added

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, token_id: int) -> bytes:
        return self.table[token_id]

    def join(self, token_ids) -> bytes:
        return b"".join(self.table[i] for i in token_ids)

    def verify(self, texts) -> None:
        """Assert that re-joining token bytes reproduces the input exactly."""
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            got = self.join(ids)
            want = text.encode("utf-8")
            if got != want:
                raise AssertionError(f"byte roundtrip failed for {text!r}:\n  got  {got!r}\n  want {want!r}")
