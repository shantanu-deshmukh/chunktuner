"""URL ingestor SSRF guard."""

from __future__ import annotations

import pytest

from chunktuner.ingestion.url_ingestor import URLIngestor


def test_url_ingestor_rejects_loopback() -> None:
    with pytest.raises(ValueError, match="SSRF guard"):
        URLIngestor().ingest_url("http://127.0.0.1/secret")


def test_url_ingestor_rejects_metadata_url() -> None:
    with pytest.raises(ValueError, match="SSRF guard"):
        URLIngestor().ingest_url("http://169.254.169.254/latest/meta-data/")
