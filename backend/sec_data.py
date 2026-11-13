"""Exact numbers from SEC's XBRL company facts (the numbers behind each 10-K)."""
import csv
import functools
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

FILINGS_CSV = Path(__file__).resolve().parent.parent / "data" / "filings.csv"

# metric -> XBRL tags to try, in this order
METRIC_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "rd_expense": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "operating_income": ["OperatingIncomeLoss"],
}


@functools.lru_cache(maxsize=1)
def ciks() -> dict:
    """ticker -> CIK (SEC's company number), read from data/filings.csv."""
    with open(FILINGS_CSV, encoding="utf-8", newline="") as f:
        return {row["ticker"]: row["cik"] for row in csv.DictReader(f)}


@functools.lru_cache(maxsize=32)  # the file is large and does not change, so ask SEC once per company
def company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    # SEC requires a User-Agent with a name and an email. It comes from the environment, never from the code.
    req = urllib.request.Request(url, headers={"User-Agent": os.environ["SEC_USER_AGENT"]})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def annual_value(ticker: str, metric: str, year: int):
    """Return (value in dollars, tag) for a full fiscal year, or (None, None) if SEC does not have it."""
    facts = company_facts(ciks()[ticker])["facts"].get("us-gaap", {})
    for tag in METRIC_TAGS[metric]:
        for values in facts.get(tag, {}).get("units", {}).values():
            for v in values:
                # The frame "CY2025" marks a full calendar year. It skips the older years each 10-K repeats.
                if v.get("frame") == f"CY{year}":
                    return v["val"], tag
    return None, None
