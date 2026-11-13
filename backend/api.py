"""The FastAPI back end: POST /ask and GET /health."""
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

log = logging.getLogger("tenk.api")

REQUIRED = ["OPENROUTER_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "NEON_DATABASE_URL",
            "SEC_USER_AGENT", "ACCESS_CODE"]
OPTIONAL = ["LANGSMITH_API_KEY"]  # tracing is optional, but a line break in it is still a problem


def check_env(env=os.environ) -> str | None:
    """Return a problem message, or None if all is well. It names the variable, never its value.
    A value pasted with an extra line would fail later, and the library's error would print the key."""
    for name in REQUIRED:
        if not env.get(name):
            return f"Missing environment variable: {name}"
    for name in REQUIRED + OPTIONAL:
        if "\n" in env.get(name, "") or "\r" in env.get(name, ""):
            return f"The environment variable {name} contains a line break. Paste it again on one line."
    return None


# Check BEFORE importing the pipelines: they read the keys when they are imported.
_problem = check_env()
if _problem:
    print(_problem)
    raise SystemExit(1)

from backend.agent import close_pool  # noqa: E402
from backend.pipelines import answer  # noqa: E402

PIPELINES = ["simple", "filters", "rerank", "decompose", "agent"]
MAX_QUESTION = 1000

# The fixed error bodies from requirements.md.
ERR_CODE = "Wrong or missing access code."
ERR_EMPTY = "The question is empty."
ERR_LONG = "The question is too long. Use at most 1,000 characters."
ERR_PIPELINE = "Unknown pipeline. Use simple, filters, rerank, decompose, or agent."
ERR_DOWN = "The answer service is not available right now. Try again in a minute."


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    close_pool()  # close the Neon connections when the back end stops


app = FastAPI(lifespan=lifespan)


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    # A body that is not valid JSON or has the wrong types. Do not echo it back.
    return error(400, "The request is not valid.")


class AskRequest(BaseModel):
    question: str = ""
    chat_id: str = ""
    pipeline: str = "agent"
    companies: list[str] = []
    years: list[int] = []


@app.get("/health")
def health():
    return {"status": "ok"}  # calls no paid service


@app.post("/ask")
def ask(req: AskRequest, x_access_code: str | None = Header(default=None)):
    # compare_digest compares in constant time, so the time taken does not leak the code
    if not x_access_code or not secrets.compare_digest(x_access_code, os.environ["ACCESS_CODE"]):
        return error(401, ERR_CODE)
    if not req.question.strip():
        return error(400, ERR_EMPTY)
    if len(req.question) > MAX_QUESTION:
        return error(400, ERR_LONG)
    if req.pipeline not in PIPELINES:
        return error(400, ERR_PIPELINE)
    try:
        return answer(req.question, req.pipeline, req.companies, req.years, req.chat_id or str(uuid.uuid4()))
    except Exception as e:  # OpenRouter, Qdrant, Neon, or SEC failed
        # Log only the error type. The message can contain a key, so it must never be logged.
        log.error("answer failed: %s", type(e).__name__)
        return error(503, ERR_DOWN)
