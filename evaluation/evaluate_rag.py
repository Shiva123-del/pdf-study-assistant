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
print("RAG RETRIEVAL EVALUATION")
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
    print(f"RECALL@{k}")
    print("#" * 60)

    hits = 0

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


        # Extract retrieved page numbers
        retrieved_pages = []

        for document in retrieved_documents:

            page = document.metadata.get("page_number")

            if page is not None:
                retrieved_pages.append(page)


        # Remove duplicate page numbers
        retrieved_pages = list(dict.fromkeys(retrieved_pages))


        # Check whether at least one relevant page was retrieved
        hit = bool(
            relevant_pages.intersection(retrieved_pages)
        )


        if hit:
            hits += 1


        print("-" * 60)
        print(f"Question: {question}")
        print(f"Relevant page(s): {sorted(relevant_pages)}")
        print(f"Retrieved page(s): {retrieved_pages}")
        print(f"Recall@{k}: {'PASS' if hit else 'FAIL'}")


    # Calculate Recall@K
    recall = hits / len(test_dataset)

    summary[k] = {
        "hits": hits,
        "total": len(test_dataset),
        "recall": recall,
    }


    print("\n" + "=" * 60)
    print(f"RECALL@{k} RESULT")
    print("=" * 60)

    print(
        f"Recall@{k}: {recall:.2%}"
    )

    print(
        f"The system retrieved at least one relevant page "
        f"for {hits} out of {len(test_dataset)} questions."
    )


# ---------------------------------------------------------
# Final comparison
# ---------------------------------------------------------

print("\n\n" + "=" * 60)
print("FINAL RECALL COMPARISON")
print("=" * 60)

print("\nK\tRecall\t\tQuestions")

print("-" * 60)

for k in K_VALUES:

    result = summary[k]

    print(
        f"{k}\t"
        f"{result['recall']:.2%}\t\t"
        f"{result['hits']}/{result['total']}"
    )

print("=" * 60)


# ---------------------------------------------------------
# Improvement analysis
# ---------------------------------------------------------

print("\nImprovement as K increases:")

previous_recall = None

for k in K_VALUES:

    current_recall = summary[k]["recall"]

    if previous_recall is None:

        print(
            f"Recall@{k}: {current_recall:.2%}"
        )

    else:

        improvement = (
            current_recall - previous_recall
        ) * 100

        print(
            f"Recall@{k}: {current_recall:.2%} "
            f"({improvement:+.2f} percentage points)"
        )

    previous_recall = current_recall


print("\n" + "=" * 60)
print("Evaluation completed.")
print("=" * 60)