# PDF Study Assistant

Upload a PDF book, ask questions about it, and get answers grounded in the actual
pages — cited down to the page number. Then test yourself with quizzes generated
from the same material.

Built with Python, LangChain, Chroma and the OpenAI API, in a single Gradio app.

---

## Why this is not just another RAG demo

Most "chat with your PDF" projects stop at *embed, retrieve, prompt*. This one deals
with the things that break once you actually use it.

### Retrieval returns diverse passages, not three copies of the same one

Chunks overlap by 200 characters, so plain similarity search keeps returning the same
passage in slightly different forms and wastes the context window. This uses
**Maximal Marginal Relevance** (`k=5`, `fetch_k=20`), which trades a little relevance
for diversity:

```python
return get_vector_store().max_marginal_relevance_search(
    query, k=k, fetch_k=max(fetch_k or k * 4, k), filter=document_filter(document_id),
)
```

### Follow-up questions get rewritten before retrieval

"Explain that more simply" is a useless search query. A condensation step turns each
follow-up into a standalone query using the conversation history, so retrieval still
works on the second and third question.

### Ingestion is idempotent

Documents are identified by **SHA-256 of their contents**, not by filename. Two
different files called `notes.pdf` never collide, and re-uploading the same book
costs zero embedding calls. Chunk IDs are deterministic:
`{document_hash}_page_{n}_chunk_{n}`.

### Every answer is traceable

Text is extracted page by page and the page number travels with each chunk through
to the citation. The system prompt requires inline citations like `(book.pdf, p. 12)`,
and the UI shows the passages the answer was built from.

### Quiz options are shuffled after generation

LLMs have a measurable bias toward putting the correct answer at A or B. Options are
shuffled post-generation — and explanations are written without referencing option
letters, because the letters move:

```python
order = LETTERS[:]
rng.shuffle(order)
shuffled["options"] = {LETTERS[i]: question["options"][order[i]] for i in range(4)}
shuffled["correct_answer"] = LETTERS[order.index(question["correct_answer"])]
```

### Temperature is set per task

`0.0` for query rewriting, `0.1` for grounded answering, `0.4` for quiz generation —
deterministic where correctness matters, varied where it does not.

---

## Pipeline

| Stage | Implementation |
|---|---|
| Extraction | PyMuPDF, page by page, page numbers preserved |
| Chunking | `RecursiveCharacterTextSplitter`, 1000 chars / 200 overlap, paragraph → sentence → word fallback |
| Embedding | OpenAI `text-embedding-3-small` |
| Storage | Chroma, persistent on disk |
| Retrieval | Maximal Marginal Relevance, `k=5`, `fetch_k=20`, metadata-filtered per book |
| Generation | `gpt-4.1-mini`, streamed, grounded in retrieved context |
| Quiz | Pydantic structured output (JSON schema mode), de-duplicated, options shuffled |

Also handled: password-protected PDFs, scanned PDFs with no selectable text (reports
that OCR is needed rather than silently returning nothing), batched Chroma
reads/writes/deletes, and `lru_cache` singletons for the vector store and chat model.

---

## Features

**Ask questions** — multi-turn chat over one book or the whole library, streamed
token by token, with the source passages shown for every answer.

**Quiz mode** — 5 to 20 questions, Easy / Medium / Hard, scoped to a topic or drawn
from the whole book. Points, streak bonuses and a scored review at the end.

**Library** — upload several PDFs, scope questions to one or search across all,
remove a book and its chunks.

---

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your OpenAI API key
python app.py               # add --share for a public HTTPS link
```

Options: `--host` (use `0.0.0.0` for other devices on your network), `--port`,
`--share`.

### Configuration

Everything below is an environment variable with a sensible default:

| Variable | Default |
|---|---|
| `CHAT_MODEL` | `gpt-4.1-mini` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` |
| `CHUNK_SIZE` | `1000` |
| `CHUNK_OVERLAP` | `200` |
| `ANSWER_TOP_K` | `5` |
| `DB_DIRECTORY` | `./chroma_db` |

---

## Cost

Built deliberately on `gpt-4.1-mini` and `text-embedding-3-small`. A system that
works but costs too much to run is not finished. Re-uploading a book you already
ingested costs nothing, because of the content-hash check.

---

## Known limitations

- `sample_chunks()` loads the whole collection into memory before sampling. Fine for
  a handful of books, heavy once the library grows.
- `QuizQuestion.page` is typed `int`, but `format_context` can emit `"?"` if
  `page_number` is ever missing — that would fail validation for the whole quiz
  rather than one question. Ingestion always sets it, so this only affects
  externally-written data.
- Scanned PDFs are detected but not OCR'd.

---

## Licence

MIT
