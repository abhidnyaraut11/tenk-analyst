"""Stage 0: split the 10-K files into chunks and store them in Qdrant.

Run with:  uv run python -m backend.ingest
Safe to run twice: a filing that is already complete is skipped.
"""
import csv
import re
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from backend.rag import COLLECTION, client, embeddings, make_filter

DATA = Path(__file__).resolve().parent.parent / "data"
ITEM = re.compile(r"^## Item (\w+)\. (.+)$", re.M)
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)


def read_items(path: Path) -> list[dict]:
    """Return one dict per Item: code, title, and text."""
    text = path.read_text(encoding="utf-8")
    heads = list(ITEM.finditer(text))
    items = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        items.append({"code": h.group(1), "title": h.group(2), "text": text[h.end():end].strip()})
    return items


def chunk_id(company: str, year: int, item: str, n: int) -> str:
    """A stable ID: the same chunk always gets the same ID, so re-running overwrites, not duplicates."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{company}/{year}/{item}/{n}"))


def make_chunks(row: dict) -> tuple[list[Document], list[str]]:
    """Split one filing into chunks. Chunks never cross from one Item into the next."""
    company, year = row["ticker"], int(row["fiscal_year"])
    docs, ids = [], []
    for item in read_items(DATA / row["file"]):
        for n, piece in enumerate(splitter.split_text(item["text"])):
            docs.append(Document(
                page_content=piece,
                metadata={"company": company, "year": year, "item": item["code"],
                          "item_title": item["title"], "source": row["source_url"]},
            ))
            ids.append(chunk_id(company, year, item["code"], n))
    return docs, ids


def create_collection():
    """Create the collection and the filter indexes once."""
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(
        size=1536, distance=models.Distance.COSINE))
    client.create_payload_index(COLLECTION, "metadata.company", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.item", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.year", models.PayloadSchemaType.INTEGER)


def ingest() -> dict:
    """Store all filings. Returns how many chunks were embedded per filing."""
    create_collection()
    store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    embedded = {}
    with open(DATA / "filings.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = f"{row['ticker']}_{row['fiscal_year']}"
            docs, ids = make_chunks(row)
            # Skip the filing if Qdrant already holds all of its chunks.
            have = client.count(COLLECTION, exact=True, count_filter=make_filter(
                [row["ticker"]], [int(row["fiscal_year"])])).count
            if have == len(docs):
                embedded[name] = 0
                print(f"{name}: already complete ({have} chunks), skipped")
                continue
            store.add_documents(docs, ids=ids, batch_size=200)
            embedded[name] = len(docs)
            print(f"{name}: embedded {len(docs)} chunks")
    return embedded


if __name__ == "__main__":
    ingest()
    print("Total chunks in Qdrant:", client.count(COLLECTION, exact=True).count)
