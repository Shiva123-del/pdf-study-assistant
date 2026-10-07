"""
PDF Study Assistant (RAG) — single-file version.

Upload PDF books, ask questions about them, and take quizzes.

Setup:
    pip install "gradio>=5,<6" langchain-core langchain-openai langchain-chroma \
        langchain-text-splitters pymupdf python-dotenv pydantic
    Put OPENAI_API_KEY=sk-... in a .env file next to this script.

Run:
    python app.py            (add --share for a public HTTPS link)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
import argparse
import hashlib
import html
import os
import random
import re

import gradio as gr
import pymupdf
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()


# ======================================================================
# SETTINGS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

# --- OpenAI -------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Vector store -------------------------------------------------------
# Same folder and collection name as the original script, so books you
# already processed keep working without re-uploading.
DB_DIRECTORY = os.getenv("DB_DIRECTORY", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_knowledge_base")

# --- Chunking -----------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))

# --- Retrieval / chat ---------------------------------------------------
ANSWER_TOP_K = int(os.getenv("ANSWER_TOP_K", "10"))
MAX_HISTORY_MESSAGES = 8  # last N chat messages sent to the model

# --- Quiz ---------------------------------------------------------------
QUIZ_SIZES = [5, 10, 15, 20]
MAX_QUIZ_QUESTIONS = max(QUIZ_SIZES)
QUIZ_DIFFICULTIES = ["Easy", "Medium", "Hard"]

# Dropdown value meaning "search every stored book".
ALL_DOCUMENTS = "__all__"
ALL_DOCUMENTS_LABEL = "📚 All books"


def has_api_key() -> bool:
    return bool(OPENAI_API_KEY)


# ======================================================================
# VECTOR DATABASE
# ======================================================================

_DELETE_BATCH_SIZE = 1000


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    name: str
    pages: int
    chunks: int


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Create the vector store once and reuse it for every request."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIRECTORY,
    )


def document_filter(document_id: str | None) -> dict | None:
    """Chroma metadata filter for one book, or None for all books."""
    if not document_id or document_id == ALL_DOCUMENTS:
        return None
    return {"document_id": document_id}


# ----------------------------------------------------------------------
# Library
# ----------------------------------------------------------------------

def list_documents() -> list[StoredDocument]:
    """Every stored book with its page and chunk counts."""
    data = get_vector_store().get(include=["metadatas"])
    books: dict[str, dict] = {}

    for metadata in data.get("metadatas") or []:
        if not metadata or not metadata.get("document_id"):
            continue
        entry = books.setdefault(
            metadata["document_id"],
            {"name": metadata.get("source", "Unknown"), "pages": set(), "chunks": 0},
        )
        entry["pages"].add(metadata.get("page_number"))
        entry["chunks"] += 1

    documents = [
        StoredDocument(doc_id, info["name"], len(info["pages"]), info["chunks"])
        for doc_id, info in books.items()
    ]
    return sorted(documents, key=lambda doc: doc.name.lower())


def document_choices(include_all: bool = True) -> list[tuple[str, str]]:
    """(label, document_id) pairs for dropdowns.

    Books are selected by their content hash, so two different files that
    share a filename never get mixed up.
    """
    documents = list_documents()
    name_counts = Counter(doc.name for doc in documents)

    choices = [(ALL_DOCUMENTS_LABEL, ALL_DOCUMENTS)] if include_all else []
    for doc in documents:
        label = doc.name if name_counts[doc.name] == 1 else f"{doc.name} · {doc.document_id[:8]}"
        choices.append((label, doc.document_id))
    return choices


def has_documents() -> bool:
    return bool(get_vector_store().get(limit=1, include=["metadatas"])["ids"])


def delete_document(document_id: str) -> int:
    """Remove every chunk of a book. Returns the number of chunks deleted."""
    store = get_vector_store()
    ids = store.get(where={"document_id": document_id}, include=["metadatas"])["ids"]
    for start in range(0, len(ids), _DELETE_BATCH_SIZE):
        store.delete(ids=ids[start:start + _DELETE_BATCH_SIZE])
    return len(ids)


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------

def search(query: str, document_id: str | None, k: int = 5, fetch_k: int | None = None) -> list[Document]:
    """Relevant *and* diverse passages (max marginal relevance search).

    Plain similarity search often returns several near-identical chunks from
    the overlap between neighbours; MMR spreads the results out.
    """
    return get_vector_store().max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=max(fetch_k or k * 4, k),
        filter=document_filter(document_id),
    )


def sample_chunks(document_id: str | None, count: int) -> list[Document]:
    """Random passages from a book (used for whole-book quizzes)."""
    data = get_vector_store().get(
        where=document_filter(document_id),
        include=["documents", "metadatas"],
    )
    pairs = [
        (text, metadata or {})
        for text, metadata in zip(data.get("documents") or [], data.get("metadatas") or [])
        if text
    ]
    picked = random.sample(pairs, min(count, len(pairs)))
    picked.sort(key=lambda pair: (pair[1].get("source", ""), pair[1].get("page_number", 0)))
    return [Document(page_content=text, metadata=metadata) for text, metadata in picked]


def source_of(document: Document) -> tuple[str, object]:
    metadata = document.metadata or {}
    return metadata.get("source", "Unknown"), metadata.get("page_number", "?")


def format_context(documents: list[Document]) -> str:
    """Join passages into one prompt-ready block with source headers."""
    parts = []
    for document in documents:
        source, page = source_of(document)
        parts.append(f"[Source: {source}, page {page}]\n{document.page_content}")
    return "\n\n---\n\n".join(parts)


# ======================================================================
# PDF PROCESSING
# ======================================================================

_LOOKUP_BATCH_SIZE = 500
_ADD_BATCH_SIZE = 100


@dataclass
class IngestResult:
    name: str
    document_id: str = ""
    total_pages: int = 0
    pages_with_text: int = 0
    total_chunks: int = 0
    new_chunks: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(path: str) -> tuple[list[tuple[int, str]], int]:
    """Return ([(page_number, text), ...], total_page_count)."""
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("This PDF is password-protected.")
        total = pdf.page_count
        pages = []
        for page_number, page in enumerate(pdf, start=1):
            text = _clean_text(page.get_text())
            if text:
                pages.append((page_number, text))
    return pages, total


def build_chunks(pages: list[tuple[int, str]], source_name: str, document_id: str) -> tuple[list[Document], list[str]]:
    """Split each page into chunks while keeping page numbers for citations."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents, ids = [], []
    for page_number, text in pages:
        for chunk_number, chunk_text in enumerate(splitter.split_text(text), start=1):
            # Same ID format as the original script -> existing data is reused.
            ids.append(f"{document_id}_page_{page_number}_chunk_{chunk_number}")
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": source_name,
                        "document_id": document_id,
                        "page_number": page_number,
                        "chunk_number": chunk_number,
                    },
                )
            )
    return documents, ids


def ingest_pdf(path: str) -> IngestResult:
    """Process one PDF. Never raises: problems are reported in the result."""
    result = IngestResult(name=os.path.basename(path))
    try:
        result.document_id = file_sha256(path)
        pages, result.total_pages = extract_pages(path)
        result.pages_with_text = len(pages)
        if not pages:
            result.error = "No selectable text found (it may be a scanned PDF that needs OCR)."
            return result

        documents, ids = build_chunks(pages, result.name, result.document_id)
        result.total_chunks = len(documents)

        store = get_vector_store()
        existing: set[str] = set()
        for start in range(0, len(ids), _LOOKUP_BATCH_SIZE):
            existing.update(store.get(ids=ids[start:start + _LOOKUP_BATCH_SIZE], include=["metadatas"])["ids"])

        new_items = [(doc, chunk_id) for doc, chunk_id in zip(documents, ids) if chunk_id not in existing]
        for start in range(0, len(new_items), _ADD_BATCH_SIZE):
            batch = new_items[start:start + _ADD_BATCH_SIZE]
            store.add_documents(documents=[doc for doc, _ in batch], ids=[chunk_id for _, chunk_id in batch])
        result.new_chunks = len(new_items)

    except Exception as error:  # noqa: BLE001 - shown to the user
        result.error = str(error) or error.__class__.__name__
    return result


# ======================================================================
# QUESTION ANSWERING
# ======================================================================

NO_RESULTS_MESSAGE = (
    "I couldn't find anything related to that in the selected book. "
    "Try rephrasing, or switch to **📚 All books**."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a friendly study assistant answering questions about the user's books.

Rules:
- Use ONLY the context below. Never invent facts.
- If the context doesn't contain the answer, say so plainly and suggest a related question the books can answer.
- Cite sources inline like (book.pdf, p. 12).
- Answer in the same language the user asked in.
- Be clear and well structured: short paragraphs, bullet points or steps where they help.

Context:
{context}""",
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user's latest message as a standalone search query for finding passages "
        "in their books. Resolve words like 'it', 'that' or 'the second one' using the "
        "conversation. Keep the user's language. Return only the query.",
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


@lru_cache(maxsize=4)
def get_chat_model(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=temperature)


def _text(content) -> str:
    """Model output content as plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def to_langchain_history(history: list[dict]) -> list[BaseMessage]:
    """Convert [{'role', 'content'}] chat messages into LangChain messages."""
    messages: list[BaseMessage] = []
    for message in history[-MAX_HISTORY_MESSAGES:]:
        content = _text(message.get("content"))
        if message.get("role") == "user":
            messages.append(HumanMessage(content))
        elif message.get("role") == "assistant":
            messages.append(AIMessage(content))
    return messages


def condense_question(question: str, history: list[BaseMessage]) -> str:
    """Turn a follow-up ("explain it simpler") into a self-contained search query."""
    if not history:
        return question
    response = (CONDENSE_PROMPT | get_chat_model(0.0)).invoke({"history": history, "question": question})
    return _text(response.content).strip() or question


def stream_answer(question: str, history: list[dict], document_id: str | None) -> Iterator[tuple[str, list[Document]]]:
    """Yield (answer_so_far, source_passages) as the answer streams in.

    The first yield has an empty answer, so the UI can show sources early.
    """
    lc_history = to_langchain_history(history)
    query = condense_question(question, lc_history)
    documents = search(query, document_id, k=ANSWER_TOP_K)

    if not documents:
        yield NO_RESULTS_MESSAGE, []
        return

    yield "", documents

    messages = ANSWER_PROMPT.format_messages(
        context=format_context(documents),
        history=lc_history,
        question=question,
    )
    answer = ""
    for chunk in get_chat_model().stream(messages):
        answer += _text(chunk.content)
        yield answer, documents


def format_sources(documents: list[Document], excerpt_chars: int = 320) -> str:
    """Markdown list of the passages an answer was based on."""
    if not documents:
        return "_No sources for this answer._"

    lines = []
    for index, document in enumerate(documents, start=1):
        source, page = source_of(document)
        excerpt = " ".join(document.page_content.split())
        if len(excerpt) > excerpt_chars:
            excerpt = excerpt[:excerpt_chars].rsplit(" ", 1)[0] + " …"
        lines.append(f"**{index}. {source}** — page {page}\n\n> {excerpt}\n")
    return "\n".join(lines)


# ======================================================================
# QUIZ
# ======================================================================

LETTERS = ["A", "B", "C", "D"]

DIFFICULTY_GUIDE = {
    "Easy": "test recall of key facts, terms and definitions.",
    "Medium": "test understanding and application of ideas, not just recall.",
    "Hard": "test analysis, comparison and multi-step reasoning, with very plausible distractors.",
}


class QuizError(Exception):
    """A problem worth showing to the user as-is."""


# --- Schema the model must follow (no JSON parsing by hand) --------------

class QuizOptions(BaseModel):
    A: str
    B: str
    C: str
    D: str


class QuizQuestion(BaseModel):
    question: str
    options: QuizOptions
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: str = Field(description="1-2 sentences on why the answer is correct, based on the excerpts.")
    source: str = Field(description="Filename from the excerpt header.")
    page: int = Field(description="Page number from the excerpt header.")


class Quiz(BaseModel):
    questions: list[QuizQuestion]


QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an experienced teacher writing multiple-choice questions.

Use ONLY the document excerpts provided. Never invent facts.
- Write {count} questions, or fewer if the excerpts cannot support that many good ones.
- Four options (A-D) per question, exactly one correct. Distractors must be plausible but clearly wrong according to the excerpts.
- No duplicate or near-duplicate questions; cover different parts of the excerpts.
- Difficulty: {difficulty} - {difficulty_guide}
- Explanations must not refer to option letters (the options are shuffled afterwards).
- Copy "source" and "page" from the header of the excerpt each question is based on.
- Write in the same language as the excerpts.""",
    ),
    ("human", "Topic: {topic}\n\nDocument excerpts:\n{context}"),
])


def _shuffle_options(question: dict, rng: random.Random) -> dict:
    """Randomise option order so the answer isn't usually 'A' or 'B'."""
    order = LETTERS[:]
    rng.shuffle(order)  # order[i] = original letter shown at position i
    shuffled = dict(question)
    shuffled["options"] = {LETTERS[i]: question["options"][order[i]] for i in range(4)}
    shuffled["correct_answer"] = LETTERS[order.index(question["correct_answer"])]
    return shuffled


def clean_questions(questions: list[dict], count: int, rng: random.Random | None = None) -> list[dict]:
    """Drop duplicates/incomplete questions, shuffle options, cap the count."""
    rng = rng or random.Random()
    seen: set[str] = set()
    cleaned = []
    for question in questions:
        key = " ".join(str(question.get("question", "")).lower().split())
        options = question.get("options") or {}
        if (
            not key
            or key in seen
            or question.get("correct_answer") not in LETTERS
            or not all(str(options.get(letter, "")).strip() for letter in LETTERS)
        ):
            continue
        seen.add(key)
        cleaned.append(_shuffle_options(question, rng))
    return cleaned[:count]


def generate_quiz(topic: str, document_id: str | None, count: int, difficulty: str = "Medium") -> list[dict]:
    """Create a quiz from the selected book. Empty topic = whole book."""
    count = max(1, min(int(count), MAX_QUIZ_QUESTIONS))
    topic = (topic or "").strip()
    passages_needed = min(max(count * 2, 8), 30)

    if topic:
        documents = search(topic, document_id, k=passages_needed, fetch_k=passages_needed * 3)
    else:
        documents = sample_chunks(document_id, passages_needed)

    if not documents:
        raise QuizError("No matching content found in the selected book. Try another topic or book.")

    model = ChatOpenAI(model=CHAT_MODEL, temperature=0.4).with_structured_output(
        Quiz, method="json_schema"
    )
    quiz = (QUIZ_PROMPT | model).invoke({
        "count": count,
        "difficulty": difficulty,
        "difficulty_guide": DIFFICULTY_GUIDE.get(difficulty, DIFFICULTY_GUIDE["Medium"]),
        "topic": topic or "General review of the whole book",
        "context": format_context(documents),
    })

    if quiz is None:
        raise QuizError("The model didn't return a quiz. Please try again.")

    questions = clean_questions([q.model_dump() for q in quiz.questions], count)
    if not questions:
        raise QuizError("No valid questions were generated. Please try again.")
    return questions


def grade_quiz(questions: list[dict], answers: list[str | None]) -> tuple[int, str]:
    """Return (score, Markdown report)."""
    score = 0
    sections = []
    for index, question in enumerate(questions):
        user_answer = answers[index] if index < len(answers) else None
        correct = question["correct_answer"]
        is_correct = user_answer == correct
        score += is_correct

        if is_correct:
            verdict = "✅ Correct"
            your_line = f"{user_answer}. {question['options'][user_answer]}"
        elif user_answer:
            verdict = "❌ Incorrect"
            your_line = f"{user_answer}. {question['options'][user_answer]}"
        else:
            verdict = "⚪ Not answered"
            your_line = "—"

        sections.append(
            f"#### Q{index + 1}. {verdict}\n\n"
            f"**{question['question']}**\n\n"
            f"- Your answer: {your_line}\n"
            f"- Correct answer: **{correct}. {question['options'][correct]}**\n\n"
            f"💡 {question.get('explanation') or 'No explanation provided.'}\n\n"
            f"<sub>📄 {question.get('source', 'Unknown')}, page {question.get('page', '?')}</sub>\n"
        )

    total = len(questions)
    percent = score / total * 100 if total else 0
    if percent >= 80:
        badge = "🏆 Excellent work!"
    elif percent >= 50:
        badge = "👍 Good effort — review the misses below."
    else:
        badge = "📖 Keep going — the explanations below will help."

    header = f"## Score: {score} / {total} ({percent:.0f}%)\n\n{badge}\n\n---\n\n"
    return score, header + "\n---\n\n".join(sections)


# ======================================================================
# GRADIO INTERFACE
# ======================================================================

APP_TITLE = "PDF Study Assistant"
GRADIO_MAJOR = int(gr.__version__.split(".")[0])

# Calm, tinted palette: soft blue-grey page, white cards, one indigo accent.
THEME = gr.themes.Default(
    primary_hue="indigo",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#eef1f7",
    body_background_fill_dark="#0b1120",
    block_background_fill="#ffffff",
    block_background_fill_dark="#141c2f",
    block_border_color="#e2e7f0",
    block_border_color_dark="#243049",
    block_shadow="0 1px 3px rgba(15, 23, 42, 0.06)",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="#ffffff",
)

CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ---------- header ---------- */
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    color: #e0e7ff; border-radius: 18px; padding: 22px 26px; margin-bottom: 4px;
}
.hero h1 { color: #ffffff; font-size: 1.55rem; font-weight: 700; margin: 0 0 4px 0; }
.hero p { margin: 0; color: #c7d2fe; }
.hero .warn {
    margin-top: 12px; padding: 8px 12px; border-radius: 10px;
    background: #fef3c7; color: #92400e; font-size: .92rem;
}

/* ---------- quiz game ---------- */
.game-intro { text-align: center; padding: 6px 0 2px; }
.game-intro .big { font-size: 2.6rem; line-height: 1; }
.game-intro h2 { margin: 8px 0 4px; }
.game-intro p { margin: 0; opacity: .75; }

.hud { display: flex; gap: 10px; flex-wrap: wrap; }
.hud-item {
    flex: 1; min-width: 120px; text-align: center; padding: 10px 12px; border-radius: 14px;
    background: #1e1b4b; color: #e0e7ff; font-size: .95rem;
}
.hud-item b { color: #ffffff; font-size: 1.15rem; }
.progress { height: 10px; background: #dfe3ee; border-radius: 99px; margin: 12px 0 16px; overflow: hidden; }
.progress-fill { height: 100%; background: linear-gradient(90deg, #6366f1, #22c55e); border-radius: 99px;
                 transition: width .4s ease; }

.q-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    color: #ffffff; border-radius: 20px; padding: 26px 28px;
    box-shadow: 0 10px 24px rgba(30, 27, 75, .18);
}
.q-badge {
    display: inline-block; font-size: .8rem; letter-spacing: .04em; text-transform: uppercase;
    background: rgba(255,255,255,.14); color: #c7d2fe; padding: 4px 10px; border-radius: 99px;
}
.q-text { font-size: 1.3rem; font-weight: 600; line-height: 1.45; margin-top: 12px; }

.answer-btn, .answer-btn button {
    min-height: 70px !important; padding: 14px 20px !important;
    font-size: 1.02rem !important; font-weight: 500 !important;
    white-space: normal !important; text-align: left !important; justify-content: flex-start !important;
    background: #ffffff !important; color: #1e293b !important;
    border: 2px solid #e2e7f0 !important; border-radius: 16px !important;
    transition: transform .12s ease, border-color .12s ease, box-shadow .12s ease;
}
.answer-btn:hover, .answer-btn button:hover {
    transform: translateY(-2px); border-color: #6366f1 !important;
    box-shadow: 0 6px 14px rgba(79, 70, 229, .15);
}
.dark .answer-btn, .dark .answer-btn button {
    background: #1b2438 !important; color: #e2e8f0 !important; border-color: #2c3a57 !important;
}

.reveal { display: grid; gap: 10px; margin-top: 18px; }
.opt { padding: 12px 16px; border-radius: 14px; background: rgba(255,255,255,.08); color: #e0e7ff; }
.opt.correct { background: #16a34a; color: #ffffff; font-weight: 600; }
.opt.wrong { background: #dc2626; color: #ffffff; }
.feedback { margin-top: 18px; padding: 14px 16px; border-radius: 14px; background: rgba(255,255,255,.1); }
.feedback .title { font-size: 1.15rem; font-weight: 700; margin-bottom: 4px; }
.feedback .src { font-size: .82rem; opacity: .7; margin-top: 6px; }

.result-card {
    text-align: center; padding: 34px 24px; border-radius: 22px; color: #ffffff;
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    box-shadow: 0 10px 24px rgba(30, 27, 75, .18);
}
.result-emoji { font-size: 3.6rem; line-height: 1; }
.result-title { font-size: 1.6rem; font-weight: 800; margin-top: 10px; }
.result-score { font-size: 1.2rem; margin-top: 6px; color: #c7d2fe; }
.stars { font-size: 2.1rem; letter-spacing: 6px; margin-top: 12px; color: #fbbf24; }
.stars .off { color: rgba(255,255,255,.22); }
.result-stats { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin-top: 16px; }
.result-stats span { background: rgba(255,255,255,.12); padding: 8px 14px; border-radius: 99px; }
.result-msg { margin-top: 16px; color: #e0e7ff; }
"""

EMPTY_LIBRARY_MESSAGE = (
    "You haven't uploaded any books yet. Go to **📚 Upload books**, add a PDF, "
    "then come back and ask me anything about it."
)
NO_KEY_MESSAGE = "⚠️ OPENAI_API_KEY is missing. Add it to your .env file and restart the app."
SOURCES_PLACEHOLDER = "_Ask a question to see the pages the answer came from._"

POINTS_PER_CORRECT = 10
STREAK_BONUS = 5  # extra points for every correct answer once you're on a 3+ streak


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _skip():
    """'Leave this output unchanged' across Gradio versions."""
    return gr.skip() if hasattr(gr, "skip") else gr.update()


def _esc(value) -> str:
    return html.escape(str(value))


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _safe_choices(include_all: bool) -> list[tuple[str, str]]:
    try:
        return document_choices(include_all=include_all)
    except Exception:  # noqa: BLE001 - e.g. missing API key on first launch
        return [(ALL_DOCUMENTS_LABEL, ALL_DOCUMENTS)] if include_all else []


def _library_markdown() -> str:
    try:
        documents = list_documents()
    except Exception:  # noqa: BLE001
        documents = []
    if not documents:
        return "_No books yet. Upload a PDF above to get started._"
    return "\n".join(f"- 📘 **{d.name}** — {d.pages} pages" for d in documents)


def refresh_library(selected: str | None = None):
    """Outputs: book list, ask dropdown, quiz dropdown, remove dropdown."""
    choices = _safe_choices(include_all=True)
    valid_ids = {value for _, value in choices}
    value = selected if selected in valid_ids else ALL_DOCUMENTS
    return (
        _library_markdown(),
        gr.update(choices=choices, value=value),
        gr.update(choices=choices, value=value),
        gr.update(choices=_safe_choices(include_all=False), value=None),
    )


# ----------------------------------------------------------------------
# Upload tab
# ----------------------------------------------------------------------

def _ingest_report(results: list[IngestResult]) -> str:
    lines = []
    for r in results:
        if not r.ok:
            lines.append(f"- ⚠️ **{r.name}** — {r.error}")
        elif r.new_chunks == 0:
            lines.append(f"- ♻️ **{r.name}** — already uploaded")
        else:
            lines.append(f"- ✅ **{r.name}** — ready ({r.pages_with_text} pages)")

    if any(r.ok for r in results):
        lines.append("\nAll set! Go to **💬 Ask questions** or **🎮 Quiz game**.")
    return "\n".join(lines)


def handle_ingest(files, progress=gr.Progress()):
    if not files:
        raise gr.Error("Please choose a PDF first.")
    if not has_api_key():
        raise gr.Error(NO_KEY_MESSAGE)

    paths = [os.fspath(f) for f in (files if isinstance(files, list) else [files])]
    results = []
    for index, path in enumerate(paths):
        progress((index, len(paths)), desc=f"Reading {os.path.basename(path)}…")
        results.append(ingest_pdf(path))
    progress((len(paths), len(paths)), desc="Done")

    last_ok = next((r.document_id for r in reversed(results) if r.ok), None)
    return (_ingest_report(results), None, *refresh_library(last_ok))


def handle_delete(document_id: str | None):
    if not document_id:
        raise gr.Error("Pick a book to remove first.")
    delete_document(document_id)
    gr.Info("Book removed.")
    return refresh_library()


# ----------------------------------------------------------------------
# Ask tab
# ----------------------------------------------------------------------

def chat_turn(message: str, history: list[dict], document_id: str):
    """Stream one question/answer exchange.

    Yields: (textbox, chatbot, history_state, sources).
    """
    question = (message or "").strip()
    if not question:
        gr.Warning("Please type a question first.")
        yield message, _skip(), _skip(), _skip()
        return

    history = list(history or [])
    user_message = _user(question)
    yield "", history + [user_message, _assistant("⏳ Looking through your book…")], history, _skip()

    answer, documents = "", []
    try:
        if not has_api_key():
            answer = NO_KEY_MESSAGE
        elif not has_documents():
            answer = EMPTY_LIBRARY_MESSAGE
        else:
            for answer, documents in stream_answer(question, history, document_id):
                shown = answer + " ▌" if answer else "✍️ Writing the answer…"
                yield "", history + [user_message, _assistant(shown)], history, _skip()
    except Exception as error:  # noqa: BLE001 - shown in the chat
        answer, documents = f"⚠️ Something went wrong: {error}", []

    new_history = history + [user_message, _assistant(answer)]
    sources = format_sources(documents) if documents else SOURCES_PLACEHOLDER
    yield "", new_history, new_history, sources


def clear_chat():
    return [], [], SOURCES_PLACEHOLDER, ""


# ----------------------------------------------------------------------
# Quiz game
# ----------------------------------------------------------------------
# The game state is a plain dict kept in gr.State:
#   questions, difficulty, index, answers, correct, points, streak, best_streak

def _new_game(questions: list[dict], difficulty: str) -> dict:
    return {
        "questions": questions, "difficulty": difficulty, "index": 0, "answers": [],
        "correct": 0, "points": 0, "streak": 0, "best_streak": 0,
    }


def _is_answered(game: dict) -> bool:
    return len(game["answers"]) > game["index"]


def _hud_html(game: dict) -> str:
    total = len(game["questions"])
    shown = min(game["index"] + 1, total)
    progress = len(game["answers"]) / total * 100 if total else 0
    return f"""
<div class="hud">
  <div class="hud-item">Question<br><b>{shown} / {total}</b></div>
  <div class="hud-item">⭐ Points<br><b>{game['points']}</b></div>
  <div class="hud-item">🔥 Streak<br><b>{game['streak']}</b></div>
</div>
<div class="progress"><div class="progress-fill" style="width:{progress:.0f}%"></div></div>
"""


def _question_html(game: dict) -> str:
    question = game["questions"][game["index"]]
    parts = [
        '<div class="q-card">',
        f'<span class="q-badge">{_esc(game["difficulty"])} · Question {game["index"] + 1}</span>',
        f'<div class="q-text">{_esc(question["question"])}</div>',
    ]

    if _is_answered(game):
        chosen = game["answers"][game["index"]]
        correct = question["correct_answer"]
        parts.append('<div class="reveal">')
        for letter in LETTERS:
            css = "correct" if letter == correct else "wrong" if letter == chosen else ""
            mark = " ✓" if letter == correct else " ✗" if letter == chosen else ""
            parts.append(f'<div class="opt {css}"><b>{letter}.</b> {_esc(question["options"][letter])}{mark}</div>')
        parts.append("</div>")

        if chosen == correct:
            bonus = " (+5 streak bonus!)" if game["streak"] >= 3 else ""
            title = f"🎉 Correct! +{POINTS_PER_CORRECT}{bonus}"
        else:
            title = f"😅 Not quite — the answer is {correct}."
        parts.append(
            f'<div class="feedback"><div class="title">{title}</div>'
            f'<div>💡 {_esc(question.get("explanation") or "")}</div>'
            f'<div class="src">📄 {_esc(question.get("source", "Unknown"))}, page {_esc(question.get("page", "?"))}</div>'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _result_html(game: dict) -> str:
    total = len(game["questions"])
    percent = game["correct"] / total * 100 if total else 0
    if percent >= 90:
        emoji, stars, message = "🏆", 3, "Outstanding! You really know this material."
    elif percent >= 70:
        emoji, stars, message = "🥈", 2, "Great job! Just a few to review."
    elif percent >= 40:
        emoji, stars, message = "🥉", 1, "Nice try! Check the review below and play again."
    else:
        emoji, stars, message = "📚", 0, "Keep practising — the review below will help."

    star_html = "★" * stars + f'<span class="off">{"★" * (3 - stars)}</span>'
    return f"""
<div class="result-card">
  <div class="result-emoji">{emoji}</div>
  <div class="result-title">Quiz complete!</div>
  <div class="result-score">{game['correct']} of {total} correct · {percent:.0f}%</div>
  <div class="stars">{star_html}</div>
  <div class="result-stats">
    <span>⭐ {game['points']} points</span>
    <span>🔥 Best streak {game['best_streak']}</span>
  </div>
  <div class="result-msg">{message}</div>
</div>
"""


def _screen(game: dict | None, mode: str):
    """Everything the quiz tab shows, for one of: 'setup', 'play', 'result'.

    Outputs: game_state, setup_panel, play_panel, result_panel, hud,
             question_card, answer buttons x4, next_button, result_card, review.
    """
    hidden_button = gr.update(visible=False)

    if mode == "setup":
        return (
            game, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            "", "", *[hidden_button] * 4, hidden_button, "", "",
        )

    if mode == "result":
        _, review = grade_quiz(game["questions"], game["answers"])
        return (
            game, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
            "", "", *[hidden_button] * 4, hidden_button, _result_html(game), review,
        )

    question = game["questions"][game["index"]]
    answered = _is_answered(game)
    buttons = [
        gr.update(value=f"{letter}.  {question['options'][letter]}", visible=not answered, interactive=True)
        for letter in LETTERS
    ]
    is_last = game["index"] == len(game["questions"]) - 1
    next_button = gr.update(visible=answered, value="See my score 🏁" if is_last else "Next question ➜")
    return (
        game, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
        _hud_html(game), _question_html(game), *buttons, next_button, "", "",
    )


def start_game(topic, document_id, count, difficulty):
    if not has_api_key():
        raise gr.Error(NO_KEY_MESSAGE)
    try:
        questions = generate_quiz(topic, document_id, int(count), difficulty)
    except QuizError as error:
        raise gr.Error(str(error)) from error
    except Exception as error:  # noqa: BLE001
        raise gr.Error(f"Couldn't create the quiz: {error}") from error
    return _screen(_new_game(questions, difficulty), "play")


def choose_answer(letter: str, game: dict | None):
    if not game:
        return _screen(None, "setup")
    if _is_answered(game):  # ignore double clicks
        return _screen(game, "play")

    game = {**game, "answers": [*game["answers"], letter]}
    if letter == game["questions"][game["index"]]["correct_answer"]:
        game["correct"] += 1
        game["streak"] += 1
        game["best_streak"] = max(game["best_streak"], game["streak"])
        game["points"] += POINTS_PER_CORRECT + (STREAK_BONUS if game["streak"] >= 3 else 0)
    else:
        game["streak"] = 0
    return _screen(game, "play")


def next_question(game: dict | None):
    if not game:
        return _screen(None, "setup")
    if not _is_answered(game):
        return _screen(game, "play")
    game = {**game, "index": game["index"] + 1}
    if game["index"] >= len(game["questions"]):
        return _screen(game, "result")
    return _screen(game, "play")


def play_again(game: dict | None):
    if not game:
        return _screen(None, "setup")
    return _screen(_new_game(game["questions"], game["difficulty"]), "play")


def back_to_setup(game: dict | None):
    return _screen(game, "setup")


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------

def _chatbot() -> gr.Chatbot:
    kwargs = dict(show_label=False, height=440)
    if GRADIO_MAJOR < 6:
        kwargs.update(
            type="messages",
            placeholder="### 👋 Hi! Ask me anything about your book.",
        )
    return gr.Chatbot(**kwargs)


def _header_html() -> str:
    warning = "" if has_api_key() else f'<div class="warn">{_esc(NO_KEY_MESSAGE)}</div>'
    return f"""
<div class="hero">
  <h1>📖 {APP_TITLE}</h1>
  <p>Upload a book, ask questions about it, and challenge yourself with a quiz game.</p>
  {warning}
</div>
"""


def build_demo() -> gr.Blocks:
    blocks_kwargs = {"title": APP_TITLE}
    if GRADIO_MAJOR < 6:
        blocks_kwargs.update(theme=THEME, css=CSS)

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(_header_html())

        chat_history = gr.State([])
        game_state = gr.State(None)
        initial_choices = _safe_choices(include_all=True)

        with gr.Tabs():
            # ------------------------------------------------- Upload
            with gr.Tab("📚 Upload books"):
                pdf_files = gr.File(
                    label="Choose PDF books",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath",
                )
                upload_button = gr.Button("Upload", variant="primary")
                upload_status = gr.Markdown()

                gr.Markdown("### Your books")
                library_list = gr.Markdown()

                with gr.Accordion("Remove a book", open=False):
                    with gr.Row():
                        remove_selector = gr.Dropdown(label="Book", choices=[], value=None, scale=3)
                        remove_button = gr.Button("Remove", variant="stop", scale=1)

            # ---------------------------------------------------- Ask
            with gr.Tab("💬 Ask questions"):
                ask_book = gr.Dropdown(
                    label="Which book?",
                    choices=initial_choices,
                    value=ALL_DOCUMENTS,
                )
                chatbot = _chatbot()
                with gr.Row():
                    question_box = gr.Textbox(
                        show_label=False,
                        placeholder="Type your question and press Enter",
                        lines=1,
                        max_lines=4,
                        scale=5,
                        autofocus=True,
                    )
                    send_button = gr.Button("Ask", variant="primary", scale=1, min_width=80)
                clear_button = gr.Button("Start a new chat", size="sm", variant="secondary")

                with gr.Accordion("📄 Where did this answer come from?", open=False):
                    sources_panel = gr.Markdown(SOURCES_PLACEHOLDER)

            # ---------------------------------------------- Quiz game
            with gr.Tab("🎮 Quiz game"):
                # Screen 1: setup
                with gr.Column(visible=True) as setup_panel:
                    gr.HTML(
                        '<div class="game-intro"><div class="big">🎮</div>'
                        "<h2>Ready to play?</h2>"
                        "<p>Answer one question at a time. Build a streak for bonus points!</p></div>"
                    )
                    with gr.Row():
                        quiz_book = gr.Dropdown(
                            label="Book", choices=initial_choices, value=ALL_DOCUMENTS, scale=1
                        )
                        quiz_topic = gr.Textbox(
                            label="Topic (optional)",
                            placeholder="Leave empty for the whole book",
                            scale=1,
                        )
                    with gr.Row():
                        quiz_count = gr.Radio(
                            label="How many questions?", choices=QUIZ_SIZES, value=QUIZ_SIZES[0]
                        )
                        quiz_difficulty = gr.Radio(
                            label="Difficulty", choices=QUIZ_DIFFICULTIES, value="Medium"
                        )
                    start_button = gr.Button("▶  Start game", variant="primary", size="lg")

                # Screen 2: playing
                with gr.Column(visible=False) as play_panel:
                    hud = gr.HTML()
                    question_card = gr.HTML()
                    answer_buttons = []
                    for row_letters in (LETTERS[:2], LETTERS[2:]):
                        with gr.Row(equal_height=True):
                            for letter in row_letters:
                                answer_buttons.append(gr.Button(letter, elem_classes="answer-btn"))
                    next_button = gr.Button("Next question ➜", variant="primary", size="lg", visible=False)
                    quit_button = gr.Button("Quit game", size="sm", variant="secondary")

                # Screen 3: results
                with gr.Column(visible=False) as result_panel:
                    result_card = gr.HTML()
                    with gr.Row():
                        replay_button = gr.Button("🔁  Play again", variant="primary", size="lg")
                        new_game_button = gr.Button("🎮  New game", size="lg")
                    with gr.Accordion("📋 Review your answers", open=False):
                        review = gr.Markdown()

        # --------------------------------------------------------- Events
        library_outputs = [library_list, ask_book, quiz_book, remove_selector]

        demo.load(refresh_library, outputs=library_outputs)
        upload_button.click(
            handle_ingest,
            inputs=[pdf_files],
            outputs=[upload_status, pdf_files, *library_outputs],
            show_progress="full",
        )
        remove_button.click(handle_delete, inputs=[remove_selector], outputs=library_outputs)

        chat_inputs = [question_box, chat_history, ask_book]
        chat_outputs = [question_box, chatbot, chat_history, sources_panel]
        send_button.click(chat_turn, inputs=chat_inputs, outputs=chat_outputs)
        question_box.submit(chat_turn, inputs=chat_inputs, outputs=chat_outputs)
        clear_button.click(clear_chat, outputs=[chatbot, chat_history, sources_panel, question_box])

        game_outputs = [
            game_state, setup_panel, play_panel, result_panel, hud, question_card,
            *answer_buttons, next_button, result_card, review,
        ]
        start_button.click(
            start_game,
            inputs=[quiz_topic, quiz_book, quiz_count, quiz_difficulty],
            outputs=game_outputs,
            show_progress="full",
        )
        for letter, button in zip(LETTERS, answer_buttons):
            button.click(
                lambda game, chosen=letter: choose_answer(chosen, game),
                inputs=[game_state],
                outputs=game_outputs,
            )
        next_button.click(next_question, inputs=[game_state], outputs=game_outputs)
        replay_button.click(play_again, inputs=[game_state], outputs=game_outputs)
        new_game_button.click(back_to_setup, inputs=[game_state], outputs=game_outputs)
        quit_button.click(back_to_setup, inputs=[game_state], outputs=game_outputs)

    return demo


def launch_kwargs() -> dict:
    """Gradio 6 moved theme/css from Blocks() to launch()."""
    return {"theme": THEME, "css": CSS} if GRADIO_MAJOR >= 6 else {}


# ======================================================================
# RUN THE APP
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG PDF Knowledge Assistant")
    parser.add_argument("--host", default="0.0.0.0", help="Host address for deployment.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "7860")),
        help="Port to use. By default Gradio takes the first free port from 7860 upwards.",
    )
    parser.add_argument("--share", action="store_true", help="Create a temporary public HTTPS link.")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue(default_concurrency_limit=4).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        **launch_kwargs(),
    )


if __name__ == "__main__":
    main()
