from backend.pipelines import answer, resolve_filters, sidebar_conflict
from tests.conftest import load_questions


def question(qid):
    return next(x for x in load_questions()["questions"] if x["id"] == qid)


def test_resolve_filters_without_llm():
    scope = {"companies": ["GM"], "years": [2024]}
    # sidebar wins, each list on its own
    assert resolve_filters(["F"], [], scope) == (["F"], [2024])
    assert resolve_filters([], [2025], scope) == (["GM"], [2025])
    assert resolve_filters([], [], scope) == (["GM"], [2024])
    companies, years = resolve_filters([], [], {"companies": [], "years": []})
    assert len(companies) == 10 and years == [2024, 2025]


def test_sidebar_conflict_without_llm():
    scope = {"companies": ["GM", "F"], "years": [2024]}
    assert sidebar_conflict([], [], scope) is None
    assert sidebar_conflict(["GM", "F", "TSLA"], [2024], scope) is None
    msg = sidebar_conflict(["TSLA"], [2025], scope)
    assert msg == ("This question asks about something that is not selected in the sidebar: "
                   "General Motors, Ford, FY2024. Select it in the sidebar, or change the question.")


def test_q2_all_sources_are_gm_2024():
    result = answer(question(2)["question"], "filters", [], [], "test-chat")
    print("\nANSWER:", result["answer"])
    for s in result["sources"]:
        print(" ", s["company"], s["year"], "Item", s["item"])
    assert len(result["sources"]) == 5
    assert all(s["company"] == "GM" and s["year"] == 2024 for s in result["sources"])


def test_q9_sidebar_message():
    q = question(9)
    result = answer(q["question"], "filters", q["sidebar"]["companies"], q["sidebar"]["years"], "test-chat")
    print("\nANSWER:", result["answer"])
    assert result["answer"].startswith(q["expect"]["answer_starts_with"])
    assert "General Motors" in result["answer"]
    assert result["sources"] == [] and result["tools_used"] == []


def test_sidebar_year_conflict():
    result = answer("What was Ford's revenue in 2024?", "filters", [], [2025], "test-chat")
    assert "FY2024" in result["answer"] and result["sources"] == []


def test_sidebar_narrows_a_question_with_no_company():
    result = answer("What risks are reported about battery supply?", "filters", ["TSLA"], [2025], "test-chat")
    print("\nANSWER:", result["answer"])
    assert len(result["sources"]) == 5
    assert all(s["company"] == "TSLA" and s["year"] == 2025 for s in result["sources"])
