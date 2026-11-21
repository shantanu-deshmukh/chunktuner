"""Normalize and lightly clean document text before chunking."""

import re


def preprocess(
    content: str,
    content_type: str,
    strip_patterns: list[str] | None = None,
) -> str:
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    patterns = strip_patterns or []
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE)
    if content_type == "html":
        text = _strip_html_boilerplate(text)
    return text.strip()


def _strip_html_boilerplate(html: str) -> str:
    out = html
    for tag in ("script", "style", "nav", "footer"):
        out = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}>",
            "",
            out,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return out
