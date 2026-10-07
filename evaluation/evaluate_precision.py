import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import get_vector_store


DATASET_PATH = PROJECT_ROOT / "evaluation" / "test_dataset.json"
BOOK_NAME = "Introduction_To_Computer_Science_-_WEB.pdf"

K_VALUES = [1, 3, 5, 10]


# ---------------------------------------------------------
# Load test dataset
# ---------------------------------------------------------

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    test_dataset = json.load(file)


# ---------------------------------------------------------
# Connect to existing ChromaDB
# ---------------------------------------------------------

store = get_vector_store()

data = store.get(
    include=["metadatas"]
)


# ---------------------------------------------------------
# Find document ID for the selected book
# ---------------------------------------------------------

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
# Header
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RAG PRECISION EVALUATION")
print("=" * 60)

print(f"Book: {BOOK_NAME}")
print(f"Questions: {len(test_dataset)}")
print(f"K values: {K_VALUES}")

print("\n")


# ---------------------------------------------------------
# Store final results
# ---------------------------------------------------------

summary = {}


# ---------------------------------------------------------
# Run evaluation for every K
# ---------------------------------------------------------

for k in K_VALUES:

    print("\n" + "#" * 60)
    print(f"PRECISION@{k}")
    print("#" * 60)

    total_relevant_chunks = 0
    total_retrieved_chunks = 0

    question_precisions = []


    # -----------------------------------------------------
    # Evaluate every question
    # -----------------------------------------------------

    for item in test_dataset:

        question_id = item["id"]
        question = item["question"]

        relevant_pages = set(item["relevant_pages"])


        # Retrieve top-K chunks
        retrieved_documents = store.max_marginal_relevance_search(
            question,
            k=k,
            fetch_k=k * 4,
            filter={"document_id": document_id},
        )


        # Count how many retrieved chunks are relevant
        relevant_count = 0

        retrieved_pages = []


        for document in retrieved_documents:

            page = document.metadata.get("page_number")

            if page is not None:

                retrieved_pages.append(page)

                if page in relevant_pages:
                    relevant_count += 1


        retrieved_count = len(retrieved_documents)

        precision = (
            relevant_count / retrieved_count
            if retrieved_count > 0
            else 0
        )


        total_relevant_chunks += relevant_count
        total_retrieved_chunks += retrieved_count

        question_precisions.append(precision)


        print("-" * 60)
        print(f"Question: {question}")
        print(f"Relevant page(s): {sorted(relevant_pages)}")
        print(f"Retrieved page(s): {retrieved_pages}")
        print(
            f"Relevant chunks: "
            f"{relevant_count}/{retrieved_count}"
        )
        print(
            f"Precision@{k}: "
            f"{precision:.2%}"
        )


    # -----------------------------------------------------
    # Calculate overall Precision@K
    # -----------------------------------------------------

    overall_precision = (
        total_relevant_chunks / total_retrieved_chunks
        if total_retrieved_chunks > 0
        else 0
    )


    summary[k] = {
        "relevant_chunks": total_relevant_chunks,
        "retrieved_chunks": total_retrieved_chunks,
        "precision": overall_precision,
    }


    print("\n" + "=" * 60)
    print(f"PRECISION@{k} RESULT")
    print("=" * 60)

    print(
        f"Precision@{k}: "
        f"{overall_precision:.2%}"
    )

    print(
        f"Relevant chunks: "
        f"{total_relevant_chunks}"
    )

    print(
        f"Retrieved chunks: "
        f"{total_retrieved_chunks}"
    )


# ---------------------------------------------------------
# Final comparison
# ---------------------------------------------------------

print("\n\n" + "=" * 60)
print("FINAL PRECISION COMPARISON")
print("=" * 60)

print("\nK\tPrecision\tRelevant / Retrieved")

print("-" * 60)

for k in K_VALUES:

    result = summary[k]

    print(
        f"{k}\t"
        f"{result['precision']:.2%}\t\t"
        f"{result['relevant_chunks']}/"
        f"{result['retrieved_chunks']}"
    )

print("=" * 60)

print("\nEvaluation completed.")
print("=" * 60)