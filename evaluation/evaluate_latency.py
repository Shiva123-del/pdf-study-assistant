import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DATASET_PATH = BASE_DIR / "evaluation" / "test_dataset.json"

OUTPUT_PATH = (
    BASE_DIR
    / "evaluation"
    / "latency_evaluation_results.json"
)

DB_DIRECTORY = BASE_DIR / "chroma_db"

COLLECTION_NAME = "pdf_knowledge_base"

EMBEDDING_MODEL = "text-embedding-3-small"

LLM_MODEL = "gpt-4.1-mini"

TOP_K = 5


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def get_vector_store():

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIRECTORY)
    )

    return vector_store


# ============================================================
# CREATE LLM
# ============================================================

def get_llm():

    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )


# ============================================================
# SEARCH
# ============================================================

def retrieve_documents(
    vector_store,
    question,
    document_id=None
):

    filter_condition = None

    if document_id:

        filter_condition = {
            "document_id": document_id
        }

    start_time = time.perf_counter()

    documents = vector_store.max_marginal_relevance_search(
        question,
        k=TOP_K,
        fetch_k=TOP_K * 4,
        filter=filter_condition
    )

    retrieval_time = time.perf_counter() - start_time

    return documents, retrieval_time


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents):

    context_parts = []

    for document in documents:

        metadata = document.metadata

        source = metadata.get(
            "source",
            "Unknown source"
        )

        page = metadata.get(
            "page_number",
            "?"
        )

        text = document.page_content

        context_parts.append(
            f"Source: {source}, Page: {page}\n"
            f"{text}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    llm,
    question,
    context
):

    prompt = f"""
You are a study assistant answering questions
from a textbook.

Use ONLY the provided context.

Do not invent facts.

If the context does not contain enough information
to answer the question, say so clearly.

Question:
{question}

Context:
{context}

Answer clearly and concisely.
"""

    start_time = time.perf_counter()

    response = llm.invoke(prompt)

    generation_time = time.perf_counter() - start_time

    return response, generation_time


# ============================================================
# EXTRACT TOKEN USAGE
# ============================================================

def get_token_usage(response):

    usage = getattr(
        response,
        "usage_metadata",
        None
    )

    if not usage:

        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

    input_tokens = usage.get(
        "input_tokens",
        usage.get("prompt_tokens", 0)
    )

    output_tokens = usage.get(
        "output_tokens",
        usage.get("completion_tokens", 0)
    )

    total_tokens = usage.get(
        "total_tokens",
        input_tokens + output_tokens
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RAG LATENCY & TOKEN EVALUATION")
    print("=" * 70)

    dataset = load_dataset()

    print(
        f"Evaluation questions: {len(dataset)}"
    )

    print()

    print("Loading vector database...")

    vector_store = get_vector_store()

    print("Loading LLM...")

    llm = get_llm()

    print()

    results = []

    # --------------------------------------------------------
    # RUN EVALUATION
    # --------------------------------------------------------

    for index, case in enumerate(
        dataset,
        start=1
    ):

        question = case["question"]

        print(
            f"[{index}/{len(dataset)}] "
            f"Evaluating {case['id']}..."
        )

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        documents, retrieval_time = retrieve_documents(
            vector_store,
            question
        )

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        context = build_context(
            documents
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        response, generation_time = generate_answer(
            llm,
            question,
            context
        )

        # ----------------------------------------------------
        # TOKEN USAGE
        # ----------------------------------------------------

        token_usage = get_token_usage(
            response
        )

        # ----------------------------------------------------
        # TOTAL LATENCY
        # ----------------------------------------------------

        total_time = (
            retrieval_time
            + generation_time
        )

        results.append(
            {
                "id": case["id"],
                "question": question,
                "retrieval_latency_seconds": retrieval_time,
                "generation_latency_seconds": generation_time,
                "total_latency_seconds": total_time,
                "retrieved_chunks": len(documents),
                "input_tokens": token_usage[
                    "input_tokens"
                ],
                "output_tokens": token_usage[
                    "output_tokens"
                ],
                "total_tokens": token_usage[
                    "total_tokens"
                ]
            }
        )

        print(
            f"    Retrieval: "
            f"{retrieval_time:.3f}s"
        )

        print(
            f"    Generation: "
            f"{generation_time:.3f}s"
        )

        print(
            f"    Total: "
            f"{total_time:.3f}s"
        )

        print(
            f"    Tokens: "
            f"{token_usage['total_tokens']}"
        )

        print()

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    retrieval_times = [
        r["retrieval_latency_seconds"]
        for r in results
    ]

    generation_times = [
        r["generation_latency_seconds"]
        for r in results
    ]

    total_times = [
        r["total_latency_seconds"]
        for r in results
    ]

    input_tokens = [
        r["input_tokens"]
        for r in results
    ]

    output_tokens = [
        r["output_tokens"]
        for r in results
    ]

    total_tokens = [
        r["total_tokens"]
        for r in results
    ]

    # --------------------------------------------------------
    # PERCENTILE FUNCTION
    # --------------------------------------------------------

    def percentile(values, percentile):

        if not values:
            return 0

        values = sorted(values)

        index = (
            percentile / 100
        ) * (len(values) - 1)

        lower = int(index)

        upper = min(
            lower + 1,
            len(values) - 1
        )

        weight = index - lower

        return (
            values[lower]
            * (1 - weight)
            + values[upper]
            * weight
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {

        "questions": len(results),

        "retrieval_latency": {
            "average_seconds": statistics.mean(
                retrieval_times
            ),
            "median_seconds": statistics.median(
                retrieval_times
            ),
            "p95_seconds": percentile(
                retrieval_times,
                95
            )
        },

        "generation_latency": {
            "average_seconds": statistics.mean(
                generation_times
            ),
            "median_seconds": statistics.median(
                generation_times
            ),
            "p95_seconds": percentile(
                generation_times,
                95
            )
        },

        "total_latency": {
            "average_seconds": statistics.mean(
                total_times
            ),
            "median_seconds": statistics.median(
                total_times
            ),
            "p95_seconds": percentile(
                total_times,
                95
            )
        },

        "tokens": {
            "average_input_tokens": statistics.mean(
                input_tokens
            ),
            "average_output_tokens": statistics.mean(
                output_tokens
            ),
            "average_total_tokens": statistics.mean(
                total_tokens
            ),
            "total_input_tokens": sum(
                input_tokens
            ),
            "total_output_tokens": sum(
                output_tokens
            ),
            "total_tokens": sum(
                total_tokens
            )
        }
    }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {

        "configuration": {
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K
        },

        "summary": summary,

        "results": results
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("LATENCY & TOKEN EVALUATION COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Questions evaluated: "
        f"{len(results)}"
    )

    print()

    print("RETRIEVAL LATENCY")

    print(
        f"Average: "
        f"{summary['retrieval_latency']['average_seconds']:.3f}s"
    )

    print(
        f"Median:  "
        f"{summary['retrieval_latency']['median_seconds']:.3f}s"
    )

    print(
        f"P95:     "
        f"{summary['retrieval_latency']['p95_seconds']:.3f}s"
    )

    print()

    print("GENERATION LATENCY")

    print(
        f"Average: "
        f"{summary['generation_latency']['average_seconds']:.3f}s"
    )

    print(
        f"Median:  "
        f"{summary['generation_latency']['median_seconds']:.3f}s"
    )

    print(
        f"P95:     "
        f"{summary['generation_latency']['p95_seconds']:.3f}s"
    )

    print()

    print("TOTAL LATENCY")

    print(
        f"Average: "
        f"{summary['total_latency']['average_seconds']:.3f}s"
    )

    print(
        f"Median:  "
        f"{summary['total_latency']['median_seconds']:.3f}s"
    )

    print(
        f"P95:     "
        f"{summary['total_latency']['p95_seconds']:.3f}s"
    )

    print()

    print("TOKEN USAGE")

    print(
        f"Average input tokens: "
        f"{summary['tokens']['average_input_tokens']:.1f}"
    )

    print(
        f"Average output tokens: "
        f"{summary['tokens']['average_output_tokens']:.1f}"
    )

    print(
        f"Average total tokens: "
        f"{summary['tokens']['average_total_tokens']:.1f}"
    )

    print()

    print("Results saved to:")

    print(OUTPUT_PATH)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()