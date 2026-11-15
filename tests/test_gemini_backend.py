"""'gemini' is a real Gemini backend even when ACIE_EXPR_* (the default reply model) points at the local tuned 8B, and productgen
prefers Gemini over a small local reply model (whose 4k context rejects max_tokens=16000 with HTTP 400)."""

import pytest

from acie import expression as E
from acie.productgen import llm as L

ENV = ("GEMINI_BASE_URL", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_EXTRA_BODY", "GEMINI_MAX_TOKENS",
       "ACIE_PLANNER_BASE_URL", "ACIE_PLANNER_API_KEY", "ACIE_PLANNER_MODEL",
       "ACIE_EXPR_BASE_URL", "ACIE_EXPR_API_KEY", "ACIE_EXPR_MODEL", "ACIE_EXPR_EXTRA_BODY",
       "PRODUCTGEN_BASE_URL", "PRODUCTGEN_MODEL", "PRODUCTGEN_API_KEY", "PRODUCTGEN_EXTRA_BODY")
G = "https://generativelanguage.googleapis.com/v1beta/openai"


@pytest.fixture
def local_8b(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ACIE_EXPR_BASE_URL", "http://127.0.0.1:8011/v1")
    monkeypatch.setenv("ACIE_EXPR_MODEL", "acie-reply-8b")
    monkeypatch.setenv("ACIE_EXPR_API_KEY", "EMPTY")
    monkeypatch.setenv("ACIE_EXPR_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}')
    return monkeypatch


def test_no_gemini_configured_falls_back(local_8b):
    assert E.gemini_settings() is None and E.make_gemini_backend() is None
    c = L.OpenAICompatLLM()
    assert c.base == "http://127.0.0.1:8011/v1" and c.model == "acie-reply-8b"


def test_gemini_from_planner_settings(local_8b):
    local_8b.setenv("ACIE_PLANNER_BASE_URL", G + "/")
    local_8b.setenv("ACIE_PLANNER_API_KEY", "SECRET-G")
    local_8b.setenv("ACIE_PLANNER_MODEL", "gemini-pro-slow")
    g = E.gemini_settings()
    assert g == {"base_url": G, "api_key": "SECRET-G", "model": "gemini-3.5-flash"}  # never the planner's pro model
    b = E.make_gemini_backend()
    assert b.base_url == G and b.model == "gemini-3.5-flash" and b.extra_body == {"reasoning_effort": "minimal"}
    assert "SECRET" not in b.label
    c = L.OpenAICompatLLM()  # productgen: Gemini, not the 4k-context 8B, and without Qwen's extra body
    assert c.base == G and c.model == "gemini-3.5-flash" and c.extra == {} and "SECRET" not in repr(c)
    assert L.configured()


def test_explicit_gemini_and_productgen_override(local_8b):
    local_8b.setenv("GEMINI_BASE_URL", G)
    local_8b.setenv("GEMINI_API_KEY", "K")
    local_8b.setenv("GEMINI_MODEL", "gemini-x")
    assert E.gemini_settings()["model"] == "gemini-x"
    local_8b.setenv("PRODUCTGEN_BASE_URL", "https://other.test/v1")
    local_8b.setenv("PRODUCTGEN_MODEL", "m2")
    c = L.OpenAICompatLLM()
    assert c.base == "https://other.test/v1" and c.model == "m2"


def test_parse_json_merges_several_objects():
    assert L.parse_json('{"entries": [{"id": "a"}]}\n{"entries": [{"id": "b"}]}') == {"entries": [{"id": "a"}, {"id": "b"}]}
    e1, e2 = '{"id": "a", "answer": {"en": "x"}}', '{"id": "b", "answer": {"en": "y"}}'
    assert L.parse_json(f"```json\n{e1}\n{e2}\n```") == {"entries": [{"id": "a", "answer": {"en": "x"}}, {"id": "b", "answer": {"en": "y"}}]}
    with pytest.raises(ValueError):
        L.parse_json('{"a": 1,, }')
