# PDF Study Assistant

Upload a PDF book, ask questions about it, and get answers grounded in the actual
pages — cited down to the page number. Then test yourself with quizzes generated
from the same material.

Built with Python, LangChain, Chroma and the OpenAI API, in a single Gradio app —
and **evaluated**, not just demoed (see [Evaluation](#evaluation)).

**Live demo:** https://pdf-study-assistant-5rgb.onrender.com
*(Free hosting: the first load can take about 30 seconds to wake up, and uploaded books are cleared when the service restarts, so upload a PDF each time.)*

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

The settings above were **chosen from measurements**, not guessed. Everything below comes from the scripts and raw result files in [`evaluation/`](evaluation/), and the full write-up is in **[evaluation/REPORT.pdf](evaluation/REPORT.pdf)** (6 pages).

**The benchmark.** 30 questions written from the OpenStax *Introduction to Computer Science* textbook, each labelled with the page(s) that hold the answer (39 page labels, 8 questions with more than one correct page). An independent, stronger LLM (GPT-5.6-sol) judges answers and citations, so the system never grades its own work.

### Headline results

| | Result |
|---|---|
| Chosen setup | chunk 1500 / overlap 300 / K=10 |
| Retrieval | Recall@10 93.33%, MRR 84.17% (best at every K), 63% fewer chunks than 500/100 |
| Answer quality (independent judge, 1-5) | faithfulness 4.73, relevancy 4.97, correctness 4.87 |
| Citation test | 26 of 30 answers fully supported; 23 of 30 perfect (5/5/5) |
| Robustness | 8 of 8 passed (average 4.88 / 5) |
| Speed (chosen setup, 16 questions) | about 3.0 s per question, 2.7x the input tokens of K=5 |

### 1. Retrieval: coverage versus precision

Retrieving more passages finds the answer more often but returns more irrelevant text. First benchmark run (chunk 1000 / overlap 200):

| K | Recall | Precision | MRR |
|---|---|---|---|
| 1 | 73.33% | 73.33% | 73.33% |
| 3 | 86.67% | 41.11% | 78.89% |
| 5 | 86.67% | 25.33% | 78.33% |
| 10 | 96.67% | 15.33% | 81.78% |

The trade-off is accepted on purpose: the model can ignore a passage it does not need, but cannot use one that was never retrieved.

### 2. Choosing chunk size and K (9 settings, one run)

| Chunk / overlap | K | Recall | Precision | MRR |
|---|---|---|---|---|
| 500 / 100 | 3 | 80.00% | 46.67% | 78.33% |
| 1000 / 200 | 3 | 83.33% | 40.00% | 77.22% |
| 1500 / 300 | 3 | 86.67% | 38.89% | 82.78% |
| 500 / 100 | 5 | 86.67% | 33.33% | 80.83% |
| 1000 / 200 | 5 | 86.67% | 25.33% | 78.89% |
| 1500 / 300 | 5 | 86.67% | 25.33% | 82.50% |
| 500 / 100 | 10 | 93.33% | 23.33% | 82.81% |
| 1000 / 200 | 10 | 93.33% | 15.00% | 80.83% |
| **1500 / 300** | **10** | **93.33%** | **15.33%** | **84.17%** |

| Chunk / overlap | Chunks in the book | Indexing time |
|---|---|---|
| 500 / 100 | 6,340 | ~66 s |
| 1000 / 200 | 3,397 | ~38 s |
| **1500 / 300** | **2,349** | **~28 s** |

1500/300 ranks the right page best at every K and indexes about twice as fast. 500/100 has the best precision at every K, so it is the better choice if keeping irrelevant text out matters most. The first run above and this comparison disagree by exactly one question for the same 1000/200 setting (96.67% vs 93.33% recall at K=10); choices are made inside one run, which is internally consistent.

### 3. Answer quality

| Run | Judge | Faithfulness | Relevancy | Correctness |
|---|---|---|---|---|
| 1000/200, K=5 (30/30), first run | GPT-4.1-mini (same model that wrote the answers) | 5.00 | 5.00 | 5.00 |
| 1000/200, K=5 (30/30) | GPT-5.6-sol | 4.73 | 4.97 | 4.87 |
| 1500/300, K=10 (16/30) | GPT-5.6-sol | 4.75 | 5.00 | 4.81 |

The first run let the generating model grade its own answers and gave a perfect score on all 30, which is why it was replaced by an independent judge. On the same first 16 questions the two setups are within a few hundredths of each other (faithfulness 4.81 vs 4.75, correctness 4.94 vs 4.81), so there is no sign that K=10 hurts quality. The run for the chosen setup stopped at 16 of 30 questions when API credit ran out, so it is **not** treated as a finished benchmark. Every point lost was the same kind of error: a broadly correct answer adding a reasonable detail the retrieved text does not state.

### 4. Citation test

For each of the 30 answers, the judge read the question, the answer, the extracted citations and the retrieved text, and decided whether the cited pages really support the claims (it was told not to compare against the labelled pages).

| Metric | Mean (1-5) | Score 5 | Score 4 | Score 3 | Score 2 or below |
|---|---|---|---|---|---|
| Coverage | 4.80 | 27 | 1 | 1 | 1 |
| Correctness | 4.77 | 24 | 5 | 1 | 0 |
| Completeness | 4.63 | 24 | 2 | 3 | 1 |

- **26 of 30 answers (86.67%)** had fully supported citations; **23 of 30** scored 5 on all three metrics. The 30 answers averaged 98 words and cited 22 distinct pages.
- **The 4 answers that were not fully supported** (q6 hardware components, q10 data science, q17 recursion, q29 sequential vs binary search) share one failure: *evidence over-extension*, a correct answer that includes a detail its cited page does not state. Correct is not the same as backed by the source.
- **A limitation of the test itself:** the script only recognises citations written as `(Source: file, p. N)`. Five answers cited several pages in one bracket (e.g. `p. 20; p. 127`), so the script extracted zero citations for them, although the judge still saw the full answer. The extracted-citation count is therefore an undercount.

### 5. Robustness (8 of 8 passed, average 4.88 / 5)

Rephrased, misspelled, one-word and long multi-part questions all scored 5/5. Both out-of-document questions ("capital of France", "CEO of Microsoft") scored 5/5 because the system said the book does not contain the answer instead of inventing one. The prompt-injection attempt to reveal an API key was refused (5/5). "Latest version of Python" scored 4/5: the book mentions Python 3.9.4, and a static document cannot say what is latest today.

### 6. Speed and cost

| | 1000/200, K=5 (30 questions) | 1500/300, K=10 (16 questions) |
|---|---|---|
| Retrieval (avg) | 0.48 s | 0.73 s |
| Generation (avg) | 1.46 s | 2.28 s |
| Total per question (avg) | 1.93 s | 3.01 s |
| Input tokens (avg) | 1,053 | 2,873 |

The chosen setup uses about 2.7x the input tokens and 1.6x the time: the price of higher recall, still fast enough for interactive use.

### What these numbers do not show

- **30 questions is small.** One question is worth 3.33 percentage points, so gaps of a few points are suggestive, not conclusive.
- **The questions cluster in one area.** All labelled pages fall between pages 20 and 119 of a much longer book.
- **One textbook.** Results may differ on scanned or table-heavy PDFs.
- **Page-based precision is approximate** (a result counts as relevant if it comes from a labelled page, and some labels were imperfect), so compare settings rather than reading absolute values.
- **An LLM judged the answers**, not a human audit.
- **The robustness suite is thin:** one test per category, four of eight on binary search.

### Reproducing it

The scripts expect the textbook at `~/Desktop/Introduction_To_Computer_Science_-_WEB.pdf` (free from OpenStax) and `OPENAI_API_KEY` in `.env`. Each script writes its results into `evaluation/`:

| Test | Script |
|---|---|
| Retrieval, chunk size x K | `evaluate_configurations.py` |
| Answer quality (independent judge) | `evaluate_answers_independent.py` |
| Answer quality, other setups | `evaluate_answer_configurations.py` |
| Citation test | `evaluate_citations_v2.py` |
| Robustness | `evaluate_robustness.py` |
| Latency and tokens | `evaluate_latency.py` |

```bash
python evaluation/evaluate_configurations.py
python evaluation/evaluate_citations_v2.py
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
  claim by claim, widen the citation-extraction pattern, finish the full 30-question run
  for the chosen setup, and expand the benchmark beyond pages 20-119.

---

## Licence

MIT — see [LICENSE](LICENSE).
