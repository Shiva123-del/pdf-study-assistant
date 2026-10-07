import os
import json
import re
import time
import statistics
from pathlib import Path

import fitz
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

PDF_PATH = (
    BASE_DIR.parent
    / "Introduction_To_Computer_Science_-_WEB.pdf"
)

DATASET_PATH = (
    BASE_DIR
    / "evaluation"
    / "test_dataset.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "evaluation"
    / "answer_configuration_evaluation_results.json"
)

CHECKPOINT_PATH = (
    BASE_DIR
    / "evaluation"
    / "answer_configuration_checkpoint.json"
)


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-small"

GENERATOR_MODEL = "gpt-4.1-mini"

JUDGE_MODEL = "gpt-5.6-sol"


# ============================================================
# RETRY SETTINGS
# ============================================================

MAX_RETRIES = 5

RETRY_DELAY_SECONDS = 5


# ============================================================
# CONFIGURATIONS
# ============================================================

CONFIGURATIONS = [
    {
        "name": "500_100_K3",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 3,
    },
    {
        "name": "500_100_K5",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 5,
    },
    {
        "name": "1500_300_K3",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "top_k": 3,
    },
    {
        "name": "1500_300_K5",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "top_k": 5,
    },
    {
        "name": "1500_300_K10",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "top_k": 10,
    },
]


# ============================================================
# MODELS
# ============================================================

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)

generator = ChatOpenAI(
    model=GENERATOR_MODEL,
    temperature=0,
)

judge = ChatOpenAI(
    model=JUDGE_MODEL,
    temperature=0,
)


# ============================================================
# LOAD DATASET
# ============================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8",
) as f:

    QUESTIONS = json.load(f)


# ============================================================
# FILE CHECK
# ============================================================

def check_required_files():

    print("\nChecking required files...")

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"\nPDF not found:\n{PDF_PATH}"
        )

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATASET_PATH}"
        )

    print(
        f"PDF found: {PDF_PATH}"
    )

    print(
        f"Dataset found: {DATASET_PATH}"
    )


# ============================================================
# RETRY HELPER
# ============================================================

def invoke_with_retry(
    model,
    prompt,
    operation_name,
):

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            return model.invoke(
                prompt
            )

        except Exception as error:

            last_error = error

            print(
                f"\n  {operation_name} failed "
                f"(attempt "
                f"{attempt}/{MAX_RETRIES})"
            )

            print(
                f"  Error: {error}"
            )

            if attempt < MAX_RETRIES:

                wait_time = (
                    RETRY_DELAY_SECONDS
                    * attempt
                )

                print(
                    f"  Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

            else:

                print(
                    f"\n  {operation_name} "
                    f"failed after "
                    f"{MAX_RETRIES} attempts."
                )

    raise last_error


# ============================================================
# LOAD PDF
# ============================================================

def load_pdf_pages():

    print("\nLoading PDF...")

    print(
        f"PDF: {PDF_PATH}"
    )

    pdf = fitz.open(
        str(PDF_PATH)
    )

    page_documents = []

    for page_index in range(
        len(pdf)
    ):

        page = pdf[page_index]

        text = page.get_text(
            "text"
        ).strip()

        if not text:
            continue

        page_documents.append(
            {
                "page_number":
                    page_index + 1,

                "text":
                    text,
            }
        )

    pdf.close()

    print(
        f"Loaded "
        f"{len(page_documents)} "
        f"pages containing text."
    )

    return page_documents


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    page_documents,
    chunk_size,
    chunk_overlap,
):

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )

    chunks = []

    for page in page_documents:

        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk_number, chunk_text in enumerate(
            page_chunks
        ):

            chunks.append(
                {
                    "text":
                        chunk_text,

                    "page_number":
                        page["page_number"],

                    "chunk_number":
                        chunk_number,
                }
            )

    return chunks


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vector_store(
    chunks,
    config_name,
):

    collection_name = (
        f"eval_{config_name}"
    )

    print(
        f"\nCreating vector store: "
        f"{collection_name}"
    )

    print(
        f"Embedding "
        f"{len(chunks)} chunks..."
    )

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source":
                PDF_PATH.name,

            "page_number":
                chunk["page_number"],

            "chunk_number":
                chunk["chunk_number"],
        }
        for chunk in chunks
    ]

    ids = [
        f"{config_name}_{i}"
        for i in range(
            len(chunks)
        )
    ]

    start_time = (
        time.perf_counter()
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    batch_size = 100

    for start in range(
        0,
        len(documents),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(documents),
        )

        success = False

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            try:

                vectorstore.add_texts(
                    texts=documents[
                        start:end
                    ],

                    metadatas=metadatas[
                        start:end
                    ],

                    ids=ids[
                        start:end
                    ],
                )

                success = True

                break

            except Exception as error:

                print(
                    f"\nEmbedding batch "
                    f"{start}-{end} failed "
                    f"(attempt "
                    f"{attempt}/{MAX_RETRIES})"
                )

                print(
                    f"Error: {error}"
                )

                if attempt < MAX_RETRIES:

                    wait_time = (
                        RETRY_DELAY_SECONDS
                        * attempt
                    )

                    print(
                        f"Retrying in "
                        f"{wait_time} seconds..."
                    )

                    time.sleep(
                        wait_time
                    )

        if not success:

            raise RuntimeError(
                "Embedding failed after "
                f"{MAX_RETRIES} attempts."
            )

        print(
            f"  Embedded "
            f"{end}/{len(documents)} chunks",
            end="\r",
        )

    embedding_time = (
        time.perf_counter()
        - start_time
    )

    print()

    print(
        f"Embedding completed in "
        f"{embedding_time:.2f} seconds."
    )

    return (
        vectorstore,
        embedding_time,
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_context(
    vectorstore,
    question,
    top_k,
):

    fetch_k = max(
        top_k * 4,
        top_k,
    )

    start_time = (
        time.perf_counter()
    )

    documents = (
        vectorstore
        .max_marginal_relevance_search(
            question,
            k=top_k,
            fetch_k=fetch_k,
        )
    )

    retrieval_time = (
        time.perf_counter()
        - start_time
    )

    return (
        documents,
        retrieval_time,
    )


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(
    documents
):

    if not documents:

        return ""

    formatted = []

    for i, doc in enumerate(
        documents,
        start=1,
    ):

        page_number = (
            doc.metadata.get(
                "page_number",
                "unknown",
            )
        )

        source = (
            doc.metadata.get(
                "source",
                PDF_PATH.name,
            )
        )

        formatted.append(
            f"""
--- Retrieved Source {i} ---
Source: {source}
Page: {page_number}

{doc.page_content}
"""
        )

    return "\n".join(
        formatted
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    documents,
):

    context = format_context(
        documents
    )

    if not context:

        return (
            "",
            0,
            0,
            0,
            0,
        )

    prompt = f"""
You are an academic study assistant.

Answer the user's question using ONLY the retrieved
textbook context provided below.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain enough information,
   clearly say that the retrieved textbook context is
   insufficient.
4. Answer the question directly and clearly.
5. Keep the answer appropriately concise.
6. Include page citations using this format:
   [Page X]
7. Every factual claim should be supported by the
   retrieved context.
8. If multiple pages support the answer, cite the
   relevant pages.

USER QUESTION:

{question}

RETRIEVED TEXTBOOK CONTEXT:

{context}

ANSWER:
"""

    start_time = (
        time.perf_counter()
    )

    response = invoke_with_retry(
        generator,
        prompt,
        "Answer generation",
    )

    generation_time = (
        time.perf_counter()
        - start_time
    )

    answer = response.content

    answer = response_to_text(
        answer
    )

    usage = (
        getattr(
            response,
            "usage_metadata",
            {},
        )
        or {}
    )

    input_tokens = usage.get(
        "input_tokens",
        0,
    )

    output_tokens = usage.get(
        "output_tokens",
        0,
    )

    total_tokens = usage.get(
        "total_tokens",
        0,
    )

    return (
        answer,
        generation_time,
        input_tokens,
        output_tokens,
        total_tokens,
    )


# ============================================================
# RESPONSE TO TEXT
# ============================================================

def response_to_text(
    content
):

    if isinstance(
        content,
        str,
    ):

        return content

    if isinstance(
        content,
        list,
    ):

        parts = []

        for item in content:

            if isinstance(
                item,
                str,
            ):

                parts.append(
                    item
                )

            elif isinstance(
                item,
                dict,
            ):

                text_value = (
                    item.get(
                        "text"
                    )
                )

                if text_value:

                    parts.append(
                        text_value
                    )

            else:

                text_value = getattr(
                    item,
                    "text",
                    None,
                )

                if text_value:

                    parts.append(
                        text_value
                    )

        return "\n".join(
            parts
        )

    return str(content)


# ============================================================
# PARSE JSON
# ============================================================

def parse_json_response(
    content
):

    text = response_to_text(
        content
    ).strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL,
        )

        if match:

            return json.loads(
                match.group(0)
            )

        print(
            "\nCould not parse judge response:"
        )

        print(text)

        raise


# ============================================================
# JUDGE
# ============================================================

def judge_answer(
    question,
    answer,
    documents,
):

    context = format_context(
        documents
    )

    prompt = f"""
You are an independent evaluator of a
Retrieval-Augmented Generation (RAG) system.

Evaluate the answer ONLY against the retrieved
textbook context below.

Do not reward the answer simply because it sounds
plausible.

The answer must be supported by the provided context.

Evaluate three dimensions on a 1-5 scale.

FAITHFULNESS:

Are the factual claims in the answer supported by
the retrieved context?

1 = mostly unsupported
2 = many unsupported claims
3 = partially supported
4 = mostly supported
5 = fully supported


ANSWER RELEVANCY:

Does the answer directly and appropriately answer
the user's question?

1 = does not answer the question
2 = mostly irrelevant
3 = partially relevant
4 = mostly relevant
5 = directly and completely relevant


CORRECTNESS:

Is the answer factually correct according to the
retrieved textbook context?

1 = incorrect
2 = mostly incorrect
3 = partially correct
4 = mostly correct
5 = fully correct


Important:

- Do not use outside knowledge.
- Do not penalize an answer merely because it is concise.
- Penalize unsupported claims.
- If the answer says that the context is insufficient,
  evaluate whether that judgment is appropriate.

Return ONLY valid JSON:

{{
  "faithfulness": <1-5>,
  "answer_relevancy": <1-5>,
  "correctness": <1-5>,
  "reason": "<brief explanation>"
}}

QUESTION:

{question}

ANSWER:

{answer}

RETRIEVED TEXTBOOK CONTEXT:

{context}
"""

    response = invoke_with_retry(
        judge,
        prompt,
        "Independent judge",
    )

    return parse_json_response(
        response.content
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    config_name,
    results,
):

    checkpoint = {
        "configuration":
            config_name,

        "completed_questions":
            len(results),

        "results":
            results,

        "timestamp":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
    }

    with open(
        CHECKPOINT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            checkpoint,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# EVALUATE CONFIGURATION
# ============================================================

def evaluate_configuration(
    page_documents,
    config,
):

    name = config["name"]

    chunk_size = (
        config["chunk_size"]
    )

    chunk_overlap = (
        config["chunk_overlap"]
    )

    top_k = config["top_k"]

    print("\n")
    print("=" * 70)

    print(
        f"CONFIGURATION: {name}"
    )

    print(
        f"Chunk size = {chunk_size}"
    )

    print(
        f"Chunk overlap = {chunk_overlap}"
    )

    print(
        f"Top K = {top_k}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # CHUNKS
    # --------------------------------------------------------

    chunks = create_chunks(
        page_documents,
        chunk_size,
        chunk_overlap,
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # VECTOR STORE
    # --------------------------------------------------------

    vectorstore, embedding_time = (
        build_vector_store(
            chunks,
            name,
        )
    )

    results = []

    retrieval_times = []

    generation_times = []

    input_tokens = []

    output_tokens = []

    total_tokens = []

    faithfulness_scores = []

    relevancy_scores = []

    correctness_scores = []

    # --------------------------------------------------------
    # QUESTIONS
    # --------------------------------------------------------

    for index, item in enumerate(
        QUESTIONS,
        start=1,
    ):

        question_id = item["id"]

        question = item["question"]

        print(
            f"\n[{index}/{len(QUESTIONS)}] "
            f"{question_id}: {question}"
        )

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        documents, retrieval_time = (
            retrieve_context(
                vectorstore,
                question,
                top_k,
            )
        )

        retrieval_times.append(
            retrieval_time
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        (
            answer,
            generation_time,
            in_tokens,
            out_tokens,
            tot_tokens,
        ) = generate_answer(
            question,
            documents,
        )

        generation_times.append(
            generation_time
        )

        input_tokens.append(
            in_tokens
        )

        output_tokens.append(
            out_tokens
        )

        total_tokens.append(
            tot_tokens
        )

        # ----------------------------------------------------
        # JUDGE
        # ----------------------------------------------------

        evaluation = judge_answer(
            question,
            answer,
            documents,
        )

        faithfulness = int(
            evaluation[
                "faithfulness"
            ]
        )

        relevancy = int(
            evaluation[
                "answer_relevancy"
            ]
        )

        correctness = int(
            evaluation[
                "correctness"
            ]
        )

        faithfulness_scores.append(
            faithfulness
        )

        relevancy_scores.append(
            relevancy
        )

        correctness_scores.append(
            correctness
        )

        retrieved_pages = [
            doc.metadata.get(
                "page_number"
            )
            for doc in documents
        ]

        results.append(
            {
                "id":
                    question_id,

                "question":
                    question,

                "answer":
                    answer,

                "retrieved_pages":
                    retrieved_pages,

                "faithfulness":
                    faithfulness,

                "answer_relevancy":
                    relevancy,

                "correctness":
                    correctness,

                "judge_reason":
                    evaluation.get(
                        "reason",
                        "",
                    ),

                "retrieval_latency_seconds":
                    retrieval_time,

                "generation_latency_seconds":
                    generation_time,

                "input_tokens":
                    in_tokens,

                "output_tokens":
                    out_tokens,

                "total_tokens":
                    tot_tokens,
            }
        )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            name,
            results,
        )

        print(
            f"  Faithfulness: "
            f"{faithfulness}/5 | "
            f"Relevancy: "
            f"{relevancy}/5 | "
            f"Correctness: "
            f"{correctness}/5"
        )

        print(
            f"  Checkpoint saved "
            f"({len(results)}/"
            f"{len(QUESTIONS)})"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    avg_retrieval = statistics.mean(
        retrieval_times
    )

    median_retrieval = statistics.median(
        retrieval_times
    )

    avg_generation = statistics.mean(
        generation_times
    )

    median_generation = statistics.median(
        generation_times
    )

    total_latencies = [
        r[
            "retrieval_latency_seconds"
        ]
        +
        r[
            "generation_latency_seconds"
        ]
        for r in results
    ]

    avg_total_latency = (
        statistics.mean(
            total_latencies
        )
    )

    median_total_latency = (
        statistics.median(
            total_latencies
        )
    )

    summary = {

        "configuration":
            name,

        "chunk_size":
            chunk_size,

        "chunk_overlap":
            chunk_overlap,

        "top_k":
            top_k,

        "num_chunks":
            len(chunks),

        "embedding_time_seconds":
            embedding_time,

        "faithfulness":
            statistics.mean(
                faithfulness_scores
            ),

        "answer_relevancy":
            statistics.mean(
                relevancy_scores
            ),

        "correctness":
            statistics.mean(
                correctness_scores
            ),

        "average_retrieval_latency_seconds":
            avg_retrieval,

        "median_retrieval_latency_seconds":
            median_retrieval,

        "average_generation_latency_seconds":
            avg_generation,

        "median_generation_latency_seconds":
            median_generation,

        "average_total_latency_seconds":
            avg_total_latency,

        "median_total_latency_seconds":
            median_total_latency,

        "average_input_tokens":
            statistics.mean(
                input_tokens
            ),

        "average_output_tokens":
            statistics.mean(
                output_tokens
            ),

        "average_total_tokens":
            statistics.mean(
                total_tokens
            ),

        "questions_evaluated":
            len(QUESTIONS),

        "details":
            results,
    }

    # --------------------------------------------------------
    # DELETE TEMPORARY COLLECTION
    # --------------------------------------------------------

    try:

        vectorstore.delete_collection()

    except Exception:

        pass

    # --------------------------------------------------------
    # CLEAR CHECKPOINT
    # --------------------------------------------------------

    if CHECKPOINT_PATH.exists():

        CHECKPOINT_PATH.unlink()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\nRESULT:")

    print(
        f"Faithfulness: "
        f"{summary['faithfulness']:.2f}/5"
    )

    print(
        f"Answer Relevancy: "
        f"{summary['answer_relevancy']:.2f}/5"
    )

    print(
        f"Correctness: "
        f"{summary['correctness']:.2f}/5"
    )

    print(
        f"Average Total Latency: "
        f"{summary['average_total_latency_seconds']:.3f}s"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "RAG ANSWER-QUALITY "
        "CONFIGURATION EVALUATION"
    )

    print("=" * 70)

    print(
        f"\nGenerator Model: "
        f"{GENERATOR_MODEL}"
    )

    print(
        f"Independent Judge: "
        f"{JUDGE_MODEL}"
    )

    print(
        f"Embedding Model: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Questions: "
        f"{len(QUESTIONS)}"
    )

    print(
        f"Configurations: "
        f"{len(CONFIGURATIONS)}"
    )

    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    page_documents = (
        load_pdf_pages()
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    all_results = []

    for config in CONFIGURATIONS:

        result = (
            evaluate_configuration(
                page_documents,
                config,
            )
        )

        all_results.append(
            result
        )

    # ========================================================
    # COMPARISON
    # ========================================================

    comparison = []

    for result in all_results:

        comparison.append(
            {
                "configuration":
                    result[
                        "configuration"
                    ],

                "chunk_size":
                    result[
                        "chunk_size"
                    ],

                "chunk_overlap":
                    result[
                        "chunk_overlap"
                    ],

                "top_k":
                    result[
                        "top_k"
                    ],

                "num_chunks":
                    result[
                        "num_chunks"
                    ],

                "faithfulness":
                    round(
                        result[
                            "faithfulness"
                        ],
                        3,
                    ),

                "answer_relevancy":
                    round(
                        result[
                            "answer_relevancy"
                        ],
                        3,
                    ),

                "correctness":
                    round(
                        result[
                            "correctness"
                        ],
                        3,
                    ),

                "avg_total_latency_seconds":
                    round(
                        result[
                            "average_total_latency_seconds"
                        ],
                        3,
                    ),

                "avg_total_tokens":
                    round(
                        result[
                            "average_total_tokens"
                        ],
                        1,
                    ),
            }
        )

    # ========================================================
    # SAVE FINAL RESULTS
    # ========================================================

    output = {

        "experiment":
            "RAG answer-quality "
            "configuration ablation",

        "generator_model":
            GENERATOR_MODEL,

        "independent_judge_model":
            JUDGE_MODEL,

        "embedding_model":
            EMBEDDING_MODEL,

        "dataset_size":
            len(QUESTIONS),

        "configurations":
            all_results,

        "comparison":
            comparison,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print("\n\n")

    print("=" * 100)

    print(
        "FINAL ANSWER-QUALITY COMPARISON"
    )

    print("=" * 100)

    print(
        f"{'Config':<18}"
        f"{'Faith':>10}"
        f"{'Relevancy':>12}"
        f"{'Correct':>10}"
        f"{'Latency':>12}"
    )

    print("-" * 100)

    for row in comparison:

        print(
            f"{row['configuration']:<18}"
            f"{row['faithfulness']:>10.2f}"
            f"{row['answer_relevancy']:>12.2f}"
            f"{row['correctness']:>10.2f}"
            f"{row['avg_total_latency_seconds']:>11.3f}s"
        )

    print("=" * 100)

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()