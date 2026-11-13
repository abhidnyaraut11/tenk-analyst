import csv
from collections import Counter

from backend.ingest import DATA, ingest, make_chunks
from backend.rag import COLLECTION, client, make_filter


def test_no_repeated_item_codes():
    # A repeated Item code in one filing would make two chunks share an ID.
    with open(DATA / "filings.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            _, ids = make_chunks(row)
            assert len(ids) == len(set(ids)), f"ID collision in {row['file']}"


def test_ingest_gives_10229_chunks_and_second_run_embeds_nothing():
    ingest()  # first run: fills Qdrant if needed
    assert client.count(COLLECTION, exact=True).count == 10229
    second = ingest()
    assert sum(second.values()) == 0  # nothing embedded again
    assert client.count(COLLECTION, exact=True).count == 10229


def test_metadata_and_filter():
    points, _ = client.scroll(COLLECTION, limit=5, with_payload=True,
                              scroll_filter=make_filter(["GM"], [2024]))
    assert len(points) == 5
    for p in points:
        meta = p.payload["metadata"]
        assert meta["company"] == "GM" and meta["year"] == 2024
        assert meta["item"] and meta["item_title"] and meta["source"].startswith("https://www.sec.gov")
    assert Counter(  # all 20 filings are present
        (p.payload["metadata"]["company"], p.payload["metadata"]["year"])
        for p in client.scroll(COLLECTION, limit=10229, with_payload=["metadata.company", "metadata.year"])[0]
    ).__len__() == 20
