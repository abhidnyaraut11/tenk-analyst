import uuid

from backend.pipelines import answer
from tests.conftest import load_questions


def test_q8_follow_up_uses_the_saved_chat():
    q = next(x for x in load_questions()["questions"] if x["id"] == 8)
    chat_id = str(uuid.uuid4())  # a new chat on every run, so old runs cannot leak in
    results = []
    for text, must_contain in zip(q["conversation"], q["expect"]["answers_contain"]):
        r = answer(text, "agent", [], [], chat_id)
        print(f"\nQ: {text}\nTOOLS: {r['tools_used']}\nANSWER: {r['answer']}")
        for s in r["sources"]:
            print("  ", s)
        assert all(x in r["answer"] for x in must_contain)
        results.append(r)

    # Sources and tools of the 2nd answer come only from the 2nd question, not from the saved 1st one.
    second = results[1]
    assert "sec_financial_data" in second["tools_used"]
    assert second["sources"] and all(s["company"] == "F" and s["year"] == 2025 for s in second["sources"])
