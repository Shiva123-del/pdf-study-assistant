import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import get_vector_store


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

DATASET_PATH = PROJECT_ROOT / "evaluation" / "test_dataset.json"

BOOK_NAME = "Introduction_To_Computer_Science_-_WEB.pdf"

K_VALUES = [1, 3, 5, 10]


# ---------------------------------------------------------
# LOAD TEST DATASET
# ---------------------------------------------------------

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    test_dataset = json.load(file)


# ---------------------------------------------------------
# LOAD VECTOR DATABASE
# ---------------------------------------------------------

store = get_vector_store()


# Find document_id for our test book
data = store.get(include=["metadatas"])

document_id = None

for metadata in data.get("metadatas", []):
    if metadata and metadata.get("source") == BOOK_NAME:
        document_id = metadata.get("document_id")
        break


if document_id is None:
    print(f"\nERROR: Could not find '{BOOK_NAME}' in ChromaDB.")
    print("Make sure this PDF has been uploaded to your RAG application first.")
    sys.exit(1)


# ---------------------------------------------------------
# MRR EVALUATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RAG MRR EVALUATION")
print("=" * 60)

print(f"Book: {BOOK_NAME}")
print(f"Questions: {len(test_dataset)}")
print(f"K values: {K_VALUES}")
print("\n")


mrr_results = {k: [] for k in K_VALUES}


# ---------------------------------------------------------
# TEST EACH QUESTION
# ---------------------------------------------------------

for item in test_dataset:

    question_id = item["id"]
    question = item["question"]

    relevant_pages = set(item["relevant_pages"])

    print(f"Question: {question_id}")

    # Retrieve enough documents for the largest K
    retrieved_documents = store.max_marginal_relevance_search(
        question,
        k=max(K_VALUES),
        fetch_k=max(K_VALUES) * 4,
        filter={"document_id": document_id},
    )

    retrieved_pages = []

    for document in retrieved_documents:

        page = document.metadata.get("page_number")

        if page is not None:
            retrieved_pages.append(page)


    # -----------------------------------------------------
    # Calculate Reciprocal Rank for each K
    # -----------------------------------------------------

    for k in K_VALUES:

        top_k_pages = retrieved_pages[:k]

        reciprocal_rank = 0.0

        for rank, page in enumerate(top_k_pages, start=1):

            if page in relevant_pages:

                reciprocal_rank = 1 / rank
                break

        mrr_results[k].append(reciprocal_rank)

        print(
            f"  MRR@{k}: {reciprocal_rank:.2f}"
        )

    print()


# ---------------------------------------------------------
# FINAL MRR CALCULATIONS
# ---------------------------------------------------------

print("=" * 60)
print("FINAL MRR COMPARISON")
print("=" * 60)

print()

print(
    f"{'K':<10}"
    f"{'MRR':<15}"
    f"{'Questions':<12}"
)

print("-" * 60)


final_mrr = {}

for k in K_VALUES:

    scores = mrr_results[k]

    mrr = sum(scores) / len(scores)

    final_mrr[k] = mrr

    print(
        f"{k:<10}"
        f"{mrr:.2%}"
        f"{len(scores):<12}"
    )


print("=" * 60)


# ---------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------

print("\n")
print("INTERPRETATION")
print("-" * 60)

for k in K_VALUES:

    print(
        f"MRR@{k}: {final_mrr[k]:.2%}"
    )

print("\nHigher MRR means the first relevant chunk")
print("appears closer to the top of the retrieval results.")