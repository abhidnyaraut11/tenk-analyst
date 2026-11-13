import logging
import os

import pytest
from fastapi.testclient import TestClient

from backend import api

GOOD = {"X-Access-Code": os.environ["ACCESS_CODE"]}


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:  # the "with" runs start-up and shut-down (closes the Neon pool)
        yield c


def body(question="What does GM say about cybersecurity in 2024?", **extra):
    return {"question": question, "chat_id": "test-api", "pipeline": "filters",
            "companies": [], "years": [], **extra}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_missing_and_wrong_access_code(client):
    expected = {"error": "Wrong or missing access code."}
    for headers in [{}, {"X-Access-Code": "not-the-code"}, {"X-Access-Code": ""}]:
        r = client.post("/ask", json=body(), headers=headers)
        assert r.status_code == 401 and r.json() == expected


def test_empty_question(client):
    for q in ["", "   "]:
        r = client.post("/ask", json=body(q), headers=GOOD)
        assert r.status_code == 400 and r.json() == {"error": "The question is empty."}


def test_question_too_long(client):
    r = client.post("/ask", json=body("x" * 1001), headers=GOOD)
    assert r.status_code == 400
    assert r.json() == {"error": "The question is too long. Use at most 1,000 characters."}
    # exactly 1,000 characters is allowed: it must get past this check (the answer service is faked here)
    api.answer, real = (lambda *a, **k: {"answer": "ok"}), api.answer
    try:
        assert client.post("/ask", json=body("x" * 1000), headers=GOOD).status_code == 200
    finally:
        api.answer = real


def test_unknown_pipeline(client):
    r = client.post("/ask", json=body(pipeline="magic"), headers=GOOD)
    assert r.status_code == 400
    assert r.json() == {"error": "Unknown pipeline. Use simple, filters, rerank, decompose, or agent."}


def test_access_code_is_checked_before_the_question(client):
    r = client.post("/ask", json=body(""), headers={"X-Access-Code": "not-the-code"})
    assert r.status_code == 401


def test_service_failure_gives_503_and_leaks_nothing(client, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("sk-or-v1-SECRET-KEY-IN-THE-MESSAGE")
    monkeypatch.setattr(api, "answer", broken)
    with caplog.at_level(logging.ERROR, logger="tenk.api"):
        r = client.post("/ask", json=body(), headers=GOOD)
    assert r.status_code == 503
    assert r.json() == {"error": "The answer service is not available right now. Try again in a minute."}
    assert "SECRET" not in r.text and "Traceback" not in r.text
    # the server log has the error type only, not the message
    assert "RuntimeError" in caplog.text and "SECRET" not in caplog.text


def test_start_up_checks_name_the_variable_but_never_its_value():
    good = {n: "x" for n in api.REQUIRED}
    assert api.check_env(good) is None
    assert api.check_env({**good, "LANGSMITH_API_KEY": ""}) is None     # LangSmith is optional
    missing = {k: v for k, v in good.items() if k != "QDRANT_URL"}
    assert api.check_env(missing) == "Missing environment variable: QDRANT_URL"
    for name in api.REQUIRED + api.OPTIONAL:
        msg = api.check_env({**good, name: "secret-value\nsecond-line"})
        assert name in msg and "line break" in msg and "secret-value" not in msg


def test_ask_returns_the_documented_shape(client):
    r = client.post("/ask", json=body(), headers=GOOD)
    j = r.json()
    print("\nSTATUS:", r.status_code, "KEYS:", sorted(j))
    print("ANSWER:", j["answer"][:200])
    for s in j["sources"]:
        print("  ", s["company"], s["year"], "Item", s["item"])
    assert r.status_code == 200
    assert sorted(j) == ["answer", "chat_id", "pipeline", "sources", "tools_used"]
    assert j["pipeline"] == "filters" and j["chat_id"] == "test-api" and j["tools_used"] == []
    assert len(j["sources"]) == 5 and all(s["company"] == "GM" and s["year"] == 2024 for s in j["sources"])
