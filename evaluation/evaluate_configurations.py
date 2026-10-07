import json
import os
import statistics
import time
from pathlib import Path

import fitz
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

PDF_PATH = (
    Path.home()
    / "Desktop"
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
    / "config_k10_evaluation_results.json"
)

EMBEDDING_MODEL = "text-embedding-3-small"

# =========================================================
# IMPORTANT:
# All configurations are tested with Top-K = 10
# =========================================================

TOP_K = 10

CONFIGURATIONS = [
    {
        "name": "500_100",
        "chunk_size": 500,
        "chunk_overlap": 100,
    },
    {
        "name": "1000_200",
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    {
        "name": "1500_300",
        "chunk_size": 1500,
        "chunk_overlap": 300,
    },
]


# =========================================================
# VALIDATION
# =========================================================

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}"
    )

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )


# =========================================================
# LOAD DATASET
# =========================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8",
) as f:
    dataset = json.load(f)


# =========================================================
# LOAD PDF
# =========================================================

print()
print("=" * 75)
print("RAG CHUNKING EVALUATION — TOP-K = 10")
print("=" * 75)

print(f"PDF:       {PDF_PATH}")
print(f"Questions: {len(dataset)}")
print(f"Top-K:     {TOP_K}")
print()

print("Loading PDF...")

pdf = fitz.open(PDF_PATH)

print(f"PDF pages: {len(pdf)}")


# =========================================================
# EXTRACT TEXT
# =========================================================

print()
print("Extracting PDF text...")

pages = []

for page_number, page in enumerate(
    pdf,
    start=1,
):

    text = page.get_text("text").strip()

    if not text:
        continue

    pages.append(
        {
            "text": text,
            "page_number": page_number,
        }
    )

pdf.close()

print(
    f"Pages with extracted text: {len(pages)}"
)


# =========================================================
# METRIC FUNCTIONS
# =========================================================

def normalize_pages(page_list):

    return {
        int(page)
        for page in page_list
        if str(page).isdigit()
    }


def calculate_recall(
    retrieved_documents,
    relevant_pages,
):

    relevant = normalize_pages(
        relevant_pages
    )

    if not relevant:
        return 0.0

    retrieved_pages = set()

    for doc in retrieved_documents:

        page = doc.metadata.get(
            "page_number"
        )

        try:
            retrieved_pages.add(
                int(page)
            )
        except (
            ValueError,
            TypeError,
        ):
            continue

    return (
        1.0
        if retrieved_pages.intersection(
            relevant
        )
        else 0.0
    )


def calculate_precision(
    retrieved_documents,
    relevant_pages,
):

    relevant = normalize_pages(
        relevant_pages
    )

    if not retrieved_documents:
        return 0.0

    relevant_chunks = 0

    for doc in retrieved_documents:

        page = doc.metadata.get(
            "page_number"
        )

        try:
            page = int(page)
        except (
            ValueError,
            TypeError,
        ):
            continue

        if page in relevant:
            relevant_chunks += 1

    return (
        relevant_chunks
        / len(retrieved_documents)
    )


def calculate_mrr(
    retrieved_documents,
    relevant_pages,
):

    relevant = normalize_pages(
        relevant_pages
    )

    for rank, doc in enumerate(
        retrieved_documents,
        start=1,
    ):

        page = doc.metadata.get(
            "page_number"
        )

        try:
            page = int(page)
        except (
            ValueError,
            TypeError,
        ):
            continue

        if page in relevant:
            return 1.0 / rank

    return 0.0


# =========================================================
# EVALUATE ONE CONFIGURATION
# =========================================================

def evaluate_configuration(
    config_name,
    chunk_size,
    chunk_overlap,
):

    print()
    print("=" * 75)
    print(f"CONFIGURATION: {config_name}")
    print("=" * 75)

    print(f"Chunk size:    {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")
    print(f"Top-K:         {TOP_K}")
    print()

    # -----------------------------------------------------
    # CREATE CHUNKS
    # -----------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []

    for page_data in pages:

        page_chunks = splitter.split_text(
            page_data["text"]
        )

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1,
        ):

            chunks.append(
                {
                    "text": chunk_text,
                    "page_number": page_data[
                        "page_number"
                    ],
                    "chunk_number": chunk_number,
                }
            )

    print(
        f"Generated chunks: {len(chunks)}"
    )

    # -----------------------------------------------------
    # CREATE VECTOR STORE
    # -----------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    collection_name = (
        f"rag_eval_k10_{config_name}"
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # -----------------------------------------------------
    # CREATE EMBEDDINGS
    # -----------------------------------------------------

    print()
    print("Creating embeddings...")

    embedding_start = time.perf_counter()

    batch_size = 100

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):

        batch = chunks[
            start:start + batch_size
        ]

        texts = [
            item["text"]
            for item in batch
        ]

        metadatas = [
            {
                "page_number": item[
                    "page_number"
                ],
                "chunk_number": item[
                    "chunk_number"
                ],
            }
            for item in batch
        ]

        ids = [
            f"{config_name}_k10_{start + i}"
            for i in range(len(batch))
        ]

        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        completed = min(
            start + batch_size,
            len(chunks),
        )

        print(
            f"  Embedded "
            f"{completed}/{len(chunks)} chunks"
        )

    embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    # -----------------------------------------------------
    # RETRIEVAL EVALUATION
    # -----------------------------------------------------

    print()
    print("Evaluating retrieval...")

    recalls = []
    precisions = []
    mrr_values = []
    latencies = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        question = item["question"]

        relevant_pages = item.get(
            "relevant_pages",
            [],
        )

        start_time = time.perf_counter()

        retrieved_documents = (
            vectorstore
            .max_marginal_relevance_search(
                question,
                k=TOP_K,
                fetch_k=max(
                    TOP_K * 4,
                    TOP_K,
                ),
            )
        )

        latency = (
            time.perf_counter()
            - start_time
        )

        recall = calculate_recall(
            retrieved_documents,
            relevant_pages,
        )

        precision = calculate_precision(
            retrieved_documents,
            relevant_pages,
        )

        mrr = calculate_mrr(
            retrieved_documents,
            relevant_pages,
        )

        recalls.append(recall)
        precisions.append(precision)
        mrr_values.append(mrr)
        latencies.append(latency)

        print(
            f"[{index:02d}/{len(dataset)}] "
            f"Recall={recall:.0%} "
            f"Precision={precision:.0%} "
            f"MRR={mrr:.3f} "
            f"Latency={latency:.3f}s"
        )

    # -----------------------------------------------------
    # AGGREGATE METRICS
    # -----------------------------------------------------

    average_recall = statistics.mean(
        recalls
    )

    average_precision = statistics.mean(
        precisions
    )

    average_mrr = statistics.mean(
        mrr_values
    )

    average_latency = statistics.mean(
        latencies
    )

    median_latency = statistics.median(
        latencies
    )

    sorted_latencies = sorted(
        latencies
    )

    p95_index = min(
        int(len(sorted_latencies) * 0.95),
        len(sorted_latencies) - 1,
    )

    p95_latency = sorted_latencies[
        p95_index
    ]

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    result = {
        "configuration": config_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": TOP_K,
        "number_of_chunks": len(chunks),
        "dataset_size": len(dataset),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_time_seconds": embedding_time,
        "recall": average_recall,
        "precision": average_precision,
        "mrr": average_mrr,
        "average_latency_seconds": average_latency,
        "median_latency_seconds": median_latency,
        "p95_latency_seconds": p95_latency,
    }

    print()
    print(f"{config_name} RESULTS")
    print(
        f"Chunks:     {len(chunks)}"
    )
    print(
        f"Recall:     {average_recall:.2%}"
    )
    print(
        f"Precision:  {average_precision:.2%}"
    )
    print(
        f"MRR:        {average_mrr:.2%}"
    )
    print(
        f"Latency:    {average_latency:.3f}s"
    )
    print(
        f"P95:        {p95_latency:.3f}s"
    )

    # -----------------------------------------------------
    # CLEAN TEMPORARY COLLECTION
    # -----------------------------------------------------

    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    return result


# =========================================================
# RUN THREE CONFIGURATIONS
# =========================================================

all_results = []

for config in CONFIGURATIONS:

    result = evaluate_configuration(
        config_name=config["name"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )

    all_results.append(result)


# =========================================================
# SAVE RESULTS
# =========================================================

final_output = {
    "experiment": (
        "Chunk size and overlap comparison "
        "at Top-K = 10"
    ),
    "pdf": str(PDF_PATH),
    "embedding_model": EMBEDDING_MODEL,
    "top_k": TOP_K,
    "dataset_size": len(dataset),
    "configurations": all_results,
}

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        final_output,
        f,
        indent=2,
    )


# =========================================================
# FINAL COMPARISON
# =========================================================

print()
print()
print("=" * 90)
print("FINAL CHUNKING COMPARISON — TOP-K = 10")
print("=" * 90)

print(
    f"{'Config':<12}"
    f"{'Chunk':<10}"
    f"{'Overlap':<10}"
    f"{'Chunks':<10}"
    f"{'Recall':<12}"
    f"{'Precision':<12}"
    f"{'MRR':<12}"
    f"{'Latency':<12}"
)

print("-" * 90)

for result in all_results:

    print(
        f"{result['configuration']:<12}"
        f"{result['chunk_size']:<10}"
        f"{result['chunk_overlap']:<10}"
        f"{result['number_of_chunks']:<10}"
        f"{result['recall']:<12.2%}"
        f"{result['precision']:<12.2%}"
        f"{result['mrr']:<12.2%}"
        f"{result['average_latency_seconds']:<12.3f}"
    )

print("=" * 90)

print()
print("Results saved to:")
print(OUTPUT_PATH)