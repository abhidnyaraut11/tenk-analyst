import uuid

from backend.agent import format_money, limit_to_sidebar, search_lookup, sec_lookup
from backend.pipelines import answer
from tests.conftest import load_questions


def question(qid):
    return next(x for x in load_questions()["questions"] if x["id"] == qid)


def ask(qid):
    q = question(qid)
    sidebar = q.get("sidebar", {})
    # A new chat ID every time. The agent has memory now, so a reused ID would remember old runs.
    result = answer(q["question"], "agent", sidebar.get("companies", []), sidebar.get("years", []), str(uuid.uuid4()))
    print(f"\nQ{qid} TOOLS:", result["tools_used"])
    print("ANSWER:", result["answer"])
    for s in result["sources"]:
        print("  ", {k: s[k] for k in ("type", "company", "year") + (("item",) if s["type"] == "10-K" else ("metric", "value"))})
    return result


# ---------- direct tool tests (no agent, no chat LLM) ----------

def test_sec_ford_revenue_2024_is_185_billion():
    text, sources = sec_lookup("F", "revenue", [2024], [], [])
    print("\n", text)
    assert len(sources) == 1 and sources[0]["year"] == 2024
    assert round(sources[0]["value"] / 1e9, 1) == 185.0
    assert "$185.0 billion" in text


def test_sec_does_not_return_gm_rd_2025():
    text, sources = sec_lookup("GM", "rd_expense", [2025], [], [])
    print("\n", text)
    assert sources == [] and "not in the SEC financial data" in text


def test_sec_rejects_unsupported_things():
    for args in [("TM", "revenue", [2025]), ("F", "profit", [2025]), ("F", "revenue", [2023])]:
        text, sources = sec_lookup(*args, [], [])
        assert text.startswith("Not supported.") and sources == []


def test_tools_respect_the_sidebar():
    text, sources = sec_lookup("F", "revenue", [2025], ["TSLA"], [])       # company not in sidebar
    assert "sidebar does not include Ford" in text and sources == []
    text, sources = sec_lookup("F", "revenue", [2025], [], [2024])         # year not in sidebar
    assert "sidebar does not include" in text and sources == []
    text, sources = sec_lookup("F", "revenue", [], [], [2024])             # no years asked: use the sidebar's
    assert [s["year"] for s in sources] == [2024]
    text, sources = search_lookup("battery supply", ["F"], [2025], [], ["TSLA"], [])   # nothing left after the sidebar
    assert "sidebar does not include" in text and sources == []


def test_limit_to_sidebar_without_llm():
    assert limit_to_sidebar(["F", "GM"], ["F"], ["F", "GM"]) == ["F"]    # kept only what the sidebar allows
    assert limit_to_sidebar([], ["F"], ["F", "GM"]) == ["F"]             # asked nothing: use the sidebar
    assert limit_to_sidebar(["GM"], ["F"], ["F", "GM"]) is None          # nothing left
    assert limit_to_sidebar([], [], ["F", "GM"]) == []                   # no limit at all


def test_format_money():
    assert format_money(9_400_000_000) == "$9.4 billion"
    assert format_money(-3_700_000_000) == "-$3.7 billion"
    assert format_money(12_000_000) == "$12 million"


def test_search_tool_returns_sources_without_llm():
    text, sources = search_lookup("cybersecurity", ["GM"], [2024], [], [], [])
    assert len(sources) == 5 and all(s["company"] == "GM" and s["year"] == 2024 for s in sources)
    assert "<chunk" in text


# ---------- the agent, with the test questions ----------

def test_q4_ford_rd_2025():
    r = ask(4)
    assert "9.4" in r["answer"] and "sec_financial_data" in r["tools_used"]
    assert any(s["type"] == "sec" and s["company"] == "F" and s["year"] == 2025 for s in r["sources"])


def test_q5_gm_rd_2025_from_the_10k_text():
    r = ask(5)
    assert "8.5" in r["answer"]
    assert any(s["type"] == "10-K" and s["company"] == "GM" and s["year"] == 2025 for s in r["sources"])


def test_q6_carvana_rd_not_reported():
    r = ask(6)
    assert "This number is not reported in the SEC financial data or the 10-K." in r["answer"]
    assert "sec_financial_data" in r["tools_used"]


def test_q7_toyota_out_of_scope():
    r = ask(7)
    assert r["answer"] == question(7)["expect"]["answer_equals"]
    assert r["sources"] == [] and r["tools_used"] == []


def test_q10_sidebar_conflict():
    r = ask(10)
    assert r["answer"].startswith(question(10)["expect"]["answer_starts_with"])
    assert "General Motors" in r["answer"]
    assert r["sources"] == [] and r["tools_used"] == []
