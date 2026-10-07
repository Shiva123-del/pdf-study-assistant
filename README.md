# PDF Study Assistant

Upload a PDF book, ask questions about it, and get answers grounded in the actual
pages — cited down to the page number. Then test yourself with quizzes generated
from the same material.

Built with Python, LangChain, Chroma and the OpenAI API, in a single Gradio app —
and **evaluated**, not just demoed (see [Evaluation](#evaluation)).

---

## Why this is not just another RAG demo

Most "chat with your PDF" projects stop at *embed, retrieve, prompt*. This one deals
with the things that break once you actually use it, and measures whether its
settings are any good.

### Retrieval returns diverse passages, not copies of the same one

Chunks overlap by 300 characters, so plain similarity search can return the same
passage in slightly different forms and waste the context window. This uses
**Maximal Marginal Relevance** (`k=10`, `fetch_k=40`), which trades a little
relevance for diversity:

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
| Chunking | `RecursiveCharacterTextSplitter`, 1500 chars / 300 overlap, paragraph → sentence → word fallback |
| Embedding | OpenAI `text-embedding-3-small` |
| Storage | Chroma, persistent on disk |
| Retrieval | Maximal Marginal Relevance, `k=10`, `fetch_k=40`, metadata-filtered per book |
| Generation | `gpt-4.1-mini`, streamed, grounded in retrieved context |
| Quiz | Pydantic structured output (JSON schema mode), de-duplicated, options shuffled |

Also handled: password-protected PDFs, scanned PDFs with no selectable text (reports
that OCR is needed rather than silently returning nothing), batched Chroma
reads/writes/deletes, and `lru_cache` singletons for the vector store and chat model.

---

## Evaluation

The retrieval settings above were **chosen from measurements**, not guessed. The
benchmark is 30 questions written from the OpenStax *Introduction to Computer
Science* textbook, each labelled with the page that holds the answer. All scripts and
raw results are in [`evaluation/`](evaluation/). **Full write-up: [evaluation/REPORT.pdf](evaluation/REPORT.pdf)** (3 pages, plain-language).

### Choosing chunk size and K

Nine settings were compared (3 chunk sizes × K = 3, 5, 10). At K = 10:

| Chunk / overlap | Recall@10 | Precision@10 | MRR | Chunks in book |
|---|---|---|---|---|
| 500 / 100 | 93.33% | 23.33% | 82.81% | 6,340 |
| 1000 / 200 | 93.33% | 15.00% | 80.83% | 3,397 |
| **1500 / 300** | **93.33%** | **15.33%** | **84.17%** | **2,349** |

- **1500 / 300 ranks the right page highest at every K** (best MRR) and produces 63%
  fewer chunks than 500 / 100, so a book indexes roughly twice as fast.
- Recall rises with K (86.67% at K = 3 → 93.33% at K = 10) while page-based precision
  falls (38.89% → 15.33%). That trade-off is accepted on purpose: the model can
  ignore a passage it does not need, but cannot use one that was never retrieved.
- 500 / 100 has the best precision at every K, so it is the better choice if keeping
  irrelevant text out matters more than ranking.

### Answer quality, citations, robustness

An independent, stronger LLM judged generated answers on a 1–5 scale:

| Metric | Full run (1000/200, K=5, 30/30) | Selected setup (1500/300, K=10, 16/30)* |
|---|---|---|
| Faithfulness | 4.73 | 4.75 |
| Answer relevancy | 4.97 | 5.00 |
| Correctness | 4.87 | 4.81 |

- **Citations:** 26 of 30 answers had fully supported citations. The main failure is
  *over-extension* — a correct answer that adds a detail the cited page does not state.
- **Robustness:** 8 of 8 tests passed (rephrasing, typos, short and long queries,
  out-of-document questions, prompt injection, time-sensitive questions).
- **Latency:** 1.93 s average end to end (measured at K = 5; not re-measured at K = 10).

\* The run for the selected setup stopped at 16 of 30 questions when API credit ran
out, so it is shown for reference and is not treated as a finished benchmark.

### What these numbers do not show

- **30 questions is small**: one question is worth 3.33 percentage points, so gaps of a
  few points between settings are suggestive, not conclusive.
- **One textbook.** Results may differ on scanned or table-heavy PDFs.
- **Page-based precision is approximate** — it counts a result as relevant if it comes
  from the right page, and some page labels were imperfect.
- **An LLM judged the answers**, not a human audit.

### Reproducing it

The scripts expect the textbook at `~/Desktop/Introduction_To_Computer_Science_-_WEB.pdf`
(free from OpenStax) and `OPENAI_API_KEY` in `.env`. Each script writes its results
to `evaluation/`, for example:

```bash
python evaluation/evaluate_configurations.py   # retrieval: recall / precision / MRR across K
python evaluation/evaluate_robustness.py       # the 8 robustness tests
python evaluation/evaluate_latency.py          # retrieval + generation latency
```

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

Options: `--host`, `--port` (also read from the `PORT` environment variable), `--share`.

### Configuration

Everything below is an environment variable with a sensible default:

| Variable | Default |
|---|---|
| `CHAT_MODEL` | `gpt-4.1-mini` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` |
| `CHUNK_SIZE` | `1500` |
| `CHUNK_OVERLAP` | `300` |
| `ANSWER_TOP_K` | `10` |
| `DB_DIRECTORY` | `./chroma_db` |

Changing `CHUNK_SIZE` or `CHUNK_OVERLAP` requires re-uploading your books, because the
stored chunks were cut with the old values.

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
- Next steps from the evaluation: add a reranker after retrieval, verify citations
  claim by claim, and let the system abstain when the evidence is weak.

---

## Licence

MIT — see [LICENSE](LICENSE).
