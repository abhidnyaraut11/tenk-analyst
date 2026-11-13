from langchain_core.documents import Document

from backend.pipelines import answer, decompose_question, dedupe_chunks, fan_out
from tests.conftest import load_questions


def question(qid):
    return next(x for x in load_questions()["questions"] if x["id"] == qid)


def make_state(sidebar_companies, sidebar_years, scope, parts):
    return {"sidebar_companies": sidebar_companies, "sidebar_years": sidebar_years,
            "scope": scope, "sub_questions": parts}


def payloads(sends):
    return [s.arg for s in sends]


def test_part_filters_without_llm():
    parts = [{"question": "Tesla tariffs", "companies": ["TSLA"], "years": []},
             {"question": "Ford tariffs", "companies": ["F"], "years": [2024]},
             {"question": "Both, any company", "companies": [], "years": []}]
    scope = {"companies": ["TSLA", "F"], "years": [2025]}

    # No sidebar: part's own list, else the whole question's list.
    got = payloads(fan_out(make_state([], [], scope, parts)))
    assert (got[0]["companies"], got[0]["years"]) == (["TSLA"], [2025])
    assert (got[1]["companies"], got[1]["years"]) == (["F"], [2024])
    assert (got[2]["companies"], got[2]["years"]) == (["TSLA", "F"], [2025])

    # Sidebar wins, each list on its own.
    got = payloads(fan_out(make_state(["GM"], [], scope, parts)))
    assert all(p["companies"] == ["GM"] for p in got)
    assert [p["years"] for p in got] == [[2025], [2024], [2025]]

    # Nothing anywhere: all ten companies and both years.
    got = payloads(fan_out(make_state([], [], {"companies": [], "years": []}, parts[2:])))
    assert len(got[0]["companies"]) == 10 and got[0]["years"] == [2024, 2025]


def test_dedupe_keeps_each_chunk_once_without_llm():
    def doc(text, company="TSLA"):
        return Document(page_content=text, metadata={"company": company, "year": 2025, "item": "1A"})
    chunks = [doc("a"), doc("b"), doc("a"), doc("a", "F")]
    kept = dedupe_chunks(chunks)
    assert [(c.metadata["company"], c.page_content) for c in kept] == [("TSLA", "a"), ("TSLA", "b"), ("F", "a")]


def test_decompose_splits_the_question():
    parts = decompose_question(question(3)["question"])
    for p in parts:
        print("\nPART:", p)
    assert 1 <= len(parts) <= 4
    assert {"TSLA", "F"} <= {c for p in parts for c in p["companies"]}


def test_q3_sources_include_tesla_and_ford_2025():
    result = answer(question(3)["question"], "decompose", [], [], "test-chat")
    print("\nANSWER:", result["answer"])
    for s in result["sources"]:
        print(" ", s["company"], s["year"], "Item", s["item"])
    pairs = {(s["company"], s["year"]) for s in result["sources"]}
    assert ("TSLA", 2025) in pairs and ("F", 2025) in pairs
    assert len(result["sources"]) <= 12 and result["tools_used"] == []
