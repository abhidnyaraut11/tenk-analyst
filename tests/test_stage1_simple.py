from langchain_core.documents import Document

from backend.pipelines import MSG_COMPANY, MSG_YEAR, answer, generate_answer
from tests.conftest import load_questions


def run(question):
    return answer(question, "simple", [], [], "test-chat")


def test_q1_simple():
    q = next(x for x in load_questions()["questions"] if x["id"] == 1)
    result = run(q["question"])
    print("\nANSWER:", result["answer"])
    for s in result["sources"]:
        print(" ", s["company"], s["year"], "Item", s["item"])
    assert result["answer"].strip()
    assert len(result["sources"]) == 5
    for s in result["sources"]:
        assert s["company"] and s["year"] and s["item"]
    assert result["tools_used"] == []


def test_injection_is_ignored():
    inj = load_questions()["injection_test"]
    chunk = Document(page_content=inj["chunk_text"],
                     metadata={"company": "TSLA", "year": 2025, "item": "1A"})
    text = generate_answer(inj["question"], [chunk])
    print("\nANSWER:", text)
    assert inj["expect"]["answer_not_contains"] not in text


def test_outside_company_gets_company_message():
    result = run("What risks does Toyota report?")
    assert result["answer"] == MSG_COMPANY
    assert result["sources"] == [] and result["tools_used"] == []


def test_outside_year_gets_year_message():
    result = run("What was Ford's revenue in 2022?")
    assert result["answer"] == MSG_YEAR
    assert result["sources"] == []


def test_outside_company_and_outside_year_gets_company_message():
    result = run("What was Toyota's revenue in 2019?")
    assert result["answer"] == MSG_COMPANY
    assert result["sources"] == []
