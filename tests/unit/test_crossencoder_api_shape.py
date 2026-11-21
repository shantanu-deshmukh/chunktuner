"""D17: the installed sentence-transformers ``CrossEncoder`` still has the API that
``CrossEncoderReranker`` relies on. Inspects the class only; no model is downloaded."""

import inspect

from tests._extras import require_extra

st = require_extra("late", "sentence_transformers")


def test_crossencoder_constructor_accepts_model_and_device_keyword() -> None:
    params = inspect.signature(st.CrossEncoder.__init__).parameters
    names = list(params)
    assert names[1] == "model_name_or_path"
    assert "device" in params


def test_crossencoder_predict_accepts_batch_size_and_progress_bar() -> None:
    params = inspect.signature(st.CrossEncoder.predict).parameters
    assert "batch_size" in params
    assert "show_progress_bar" in params


def test_crossencoder_exposes_a_window_attribute() -> None:
    # 6.x: ``max_seq_length`` (``max_length`` deprecated); 5.x: ``max_length``.
    assert any(
        isinstance(getattr(st.CrossEncoder, attr, None), property)
        for attr in ("max_seq_length", "max_length")
    )
