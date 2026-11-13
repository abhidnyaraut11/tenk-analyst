from langchain_core.documents import Document

from backend.pipelines import answer
from backend.rag import make_filter, rerank, search
from tests.conftest import load_questions


def question(qid):
    return next(x for x in load_questions()["questions"] if x["id"] == qid)


def test_rerank_keeps_best_five_without_llm():
    # Search 20 chunks (this embeds the question, but makes no LLM call), then rerank to 5.
    q = question(2)["question"]
    found = search(q, k=20, flt=make_filter(["GM"], [2024]))
    kept = rerank(q, found, top_n=5)
    print("\nFound", len(found), "kept", len(kept))
    assert len(found) == 20 and len(kept) == 5
    # The kept chunks must be some of the 20 we gave it, with no repeats.
    assert all(any(k is f for f in found) for k in kept)
    assert len({k.page_content for k in kept}) == 5


def test_rerank_puts_relevant_chunk_first_without_llm():
    docs = [
        Document(page_content="The company sells cars in many countries."),
        Document(page_content="Cybersecurity incidents could disrupt our vehicles and data systems."),
        Document(page_content="Our headquarters lease expires in 2030."),
    ]
    best = rerank("What does the company say about cybersecurity?", docs, top_n=1)
    assert "Cybersecurity" in best[0].page_content


def test_rerank_of_nothing_is_empty():
    assert rerank("anything", [], top_n=5) == []


def test_q2_rerank_all_sources_are_gm_2024():
    result = answer(question(2)["question"], "rerank", [], [], "test-chat")
    print("\nANSWER:", result["answer"])
    for s in result["sources"]:
        print(" ", s["company"], s["year"], "Item", s["item"])
    assert len(result["sources"]) == 5
    assert all(s["company"] == "GM" and s["year"] == 2024 for s in result["sources"])
