"""Shared RAG code: the models, the Qdrant connection, and the search filter."""
import os

from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from qdrant_client import QdrantClient, models

load_dotenv()  # reads the .env file into environment variables

OPENROUTER = "https://openrouter.ai/api/v1"
COLLECTION = os.environ.get("QDRANT_COLLECTION", "tenk")

# Embeddings turn text into a list of 1536 numbers, so similar texts get similar numbers.
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url=OPENROUTER,
    api_key=os.environ["OPENROUTER_API_KEY"],
    check_embedding_ctx_length=False,  # required for OpenRouter
)

llm = ChatOpenAI(
    model=os.environ.get("CHAT_MODEL", "anthropic/claude-haiku-5.5"),
    base_url=OPENROUTER,
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
    max_tokens=1500,
)

client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])


# The vector store wraps the Qdrant collection so we can search it with plain text.
store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)


@traceable(run_type="retriever", name="qdrant_search")  # shows up as its own step in LangSmith
def search(question: str, k: int = 5, flt=None) -> list:
    """Return the k chunks whose meaning is closest to the question."""
    return store.similarity_search(question, k=k, filter=flt)


# FlashRank is a small model (about 3 MB) that runs on the CPU. Load it once, here, not on every call.
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")


@traceable(name="flashrank_rerank")  # shows up as its own step in LangSmith
def rerank(question: str, docs: list, top_n: int = 5) -> list:
    """Score each chunk against the question and keep the top_n best ones, best first."""
    if not docs:
        return []
    # FlashRank wants plain dicts; the id is the chunk's position so we can find it again.
    passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
    ranked = ranker.rerank(RerankRequest(query=question, passages=passages))
    return [docs[r["id"]] for r in ranked[:top_n]]


def to_source(doc) -> dict:
    """Turn a chunk into the source dict that the API returns."""
    meta = doc.metadata
    return {
        "type": "10-K",
        "company": meta["company"],
        "year": meta["year"],
        "item": meta["item"],
        "item_title": meta["item_title"],
        "excerpt": doc.page_content[:300],
        "url": meta["source"],
    }


def make_filter(companies=None, years=None, items=None):
    """Build a Qdrant filter. An empty or missing list means 'no limit'."""
    must = []
    # LangChain stores metadata under "metadata.", so the keys start with it.
    if companies:
        must.append(models.FieldCondition(key="metadata.company", match=models.MatchAny(any=companies)))
    if years:
        must.append(models.FieldCondition(key="metadata.year", match=models.MatchAny(any=years)))
    if items:
        must.append(models.FieldCondition(key="metadata.item", match=models.MatchAny(any=items)))
    return models.Filter(must=must) if must else None
