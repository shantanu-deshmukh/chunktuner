"""Fetch remote URLs into ``Document`` objects."""

from __future__ import annotations

import ipaddress
import socket
import uuid
from urllib.parse import urlparse

import httpx

from chunktuner.ingestion.preprocessor import preprocess
from chunktuner.models import Document

_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def _is_private_ip(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(host))
        return any(addr in net for net in _PRIVATE_NETWORKS)
    except (OSError, ValueError):
        return False


class URLIngestor:
    """Fetch HTTP(S) resources into a single `Document` (HTML preprocessed to text)."""

    def ingest_url(self, url: str, *, timeout: float = 30.0) -> Document:
        """GET ``url`` and map response body to ``text`` / ``markdown`` / ``html`` content."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")
        host = parsed.hostname or ""
        if _is_private_ip(host):
            raise ValueError(
                f"SSRF guard: {host!r} resolves to a private/loopback address. "
                "Only public URLs are permitted."
            )
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        raw = resp.text
        if "html" in ctype:
            body = preprocess(raw, "html")
            content_type = "html"
        else:
            body = raw
            content_type = "markdown"
        return Document(
            id=str(uuid.uuid4()),
            content=body,
            content_type=content_type,
            source_url=url,
            metadata={"content_type_header": ctype},
        )
