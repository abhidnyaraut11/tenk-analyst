"""The Streamlit page. It talks to the back end over HTTP and never imports the backend folder."""
import os
import secrets
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads .env on your computer. On Render, the values come from the dashboard.

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
ACCESS_CODE = os.environ.get("ACCESS_CODE", "")

COMPANIES = {
    "TSLA": "Tesla", "GM": "General Motors", "F": "Ford", "RIVN": "Rivian", "LCID": "Lucid",
    "APTV": "Aptiv", "BWA": "BorgWarner", "ALV": "Autoliv", "HOG": "Harley-Davidson", "CVNA": "Carvana",
}
YEARS = [2024, 2025]
PIPELINES = ["agent", "decompose", "rerank", "filters", "simple"]
EXAMPLES = [
    "What risks does Tesla report about battery supply?",
    "Compare GM and Ford R&D in 2025",
    "How did Carvana's revenue change from 2024 to 2025?",
]
MSG_WAKING = "The server is waking up. This takes about 30 seconds."
MSG_DOWN = "The answer service is not available right now. Try again in a minute."

st.set_page_config(page_title="10-K Analyst", page_icon="📈")


# ---------- the access code screen ----------

def access_screen():
    """Show a form and stop the script until the code is right."""
    if not ACCESS_CODE:
        st.error("ACCESS_CODE is not set. Add it to your .env file and restart the page.")
        st.stop()
    if st.session_state.get("authed"):
        return
    st.title("10-K Analyst")
    with st.form("access"):  # a form, so pressing Enter submits
        code = st.text_input("Access code", type="password")
        go = st.form_submit_button("Enter")
    if go:
        if secrets.compare_digest(code, ACCESS_CODE):
            st.session_state["authed"] = True
            st.rerun()
        st.error("Wrong access code.")
    st.stop()


# ---------- talking to the back end ----------

def ask_backend(payload: dict) -> tuple[int | None, dict]:
    """Send one question. Wait up to 90 seconds. If that fails, say the server is waking up and try once more."""
    for attempt in (1, 2):
        try:
            r = httpx.post(f"{API_URL}/ask", json=payload, headers={"X-Access-Code": ACCESS_CODE}, timeout=90)
            try:
                return r.status_code, r.json()
            except ValueError:  # the reply was not JSON
                return r.status_code, {"error": MSG_DOWN}
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == 1:
                st.info(MSG_WAKING)
    return None, {"error": MSG_DOWN}


# ---------- drawing a message ----------

def show_source(s: dict):
    if s["type"] == "sec":
        st.markdown(f"**{s['company']} FY{s['year']}, SEC data:** `{s['metric']}` = {s['value']:,} (tag `{s['tag']}`)")
    else:
        st.markdown(f"**{s['company']} FY{s['year']}, Item {s['item']}** – {s['item_title']}  \n"
                    f"{s['excerpt']}…  \n[Open the filing]({s['url']})")


def show_message(m: dict, show_sources: bool):
    with st.chat_message(m["role"]):
        if m.get("is_error"):
            st.error(m["content"])
            return
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("tools_used"):
                st.caption("Tools used: " + ", ".join(f"`{t}`" for t in m["tools_used"]))
            if show_sources and m.get("sources"):
                with st.expander(f"Sources ({len(m['sources'])})"):
                    for s in m["sources"]:
                        show_source(s)


# ---------- the page ----------

access_screen()

if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(uuid.uuid4())  # a random ID; the agent's memory is saved under it
    st.session_state["messages"] = []

with st.sidebar:
    st.header("Filters")
    companies = st.multiselect("Companies", list(COMPANIES), default=list(COMPANIES),
                               format_func=lambda t: f"{COMPANIES[t]} ({t})")
    years = st.multiselect("Fiscal years", YEARS, default=YEARS)
    show_sources = st.toggle("Show sources", value=True)
    if st.button("New chat"):
        st.session_state["chat_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []
        st.rerun()
    with st.expander("Learning mode"):
        pipeline = st.radio("Pipeline", PIPELINES, index=0,
                            help="agent has tools and memory. The others only search the 10-K text.")

st.title("10-K Analyst")
st.caption("Answers come only from the 10-K filings and SEC financial data of ten US auto companies.")

for m in st.session_state["messages"]:
    show_message(m, show_sources)

prompt = None
examples = st.empty()
if not st.session_state["messages"]:
    with examples.container():
        for q in EXAMPLES:
            if st.button(q, use_container_width=True):
                prompt = q
prompt = prompt or st.chat_input("Ask about the 10-Ks")

if prompt:
    examples.empty()
    user_msg = {"role": "user", "content": prompt}
    st.session_state["messages"].append(user_msg)
    show_message(user_msg, show_sources)
    # An empty list means "not set". Send empty lists when everything is selected (or nothing is).
    payload = {"question": prompt, "chat_id": st.session_state["chat_id"], "pipeline": pipeline,
               "companies": [] if len(companies) == len(COMPANIES) else companies,
               "years": [] if len(years) == len(YEARS) else years}
    with st.spinner("Searching the filings…"):
        status, data = ask_backend(payload)
    if status == 200:
        reply = {"role": "assistant", "content": data["answer"],
                 "tools_used": data["tools_used"], "sources": data["sources"]}
    else:
        reply = {"role": "assistant", "content": data.get("error", MSG_DOWN), "is_error": True}
    st.session_state["messages"].append(reply)
    show_message(reply, show_sources)
