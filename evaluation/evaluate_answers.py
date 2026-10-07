import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import get_vector_store
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

DATASET_PATH = PROJECT_ROOT / "evaluation" / "test_dataset.json"

BOOK_NAME = "Introduction_To_Computer_Science_-_WEB.pdf"

TOP_K = 5

MODEL_NAME = "gpt-4.1-mini"


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    test_dataset = json.load(file)


# ---------------------------------------------------------
# LOAD VECTOR DATABASE
# ---------------------------------------------------------

store = get_vector_store()

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
# LOAD LLM
# ---------------------------------------------------------

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
)


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RAG ANSWER EVALUATION")
print("=" * 60)

print(f"Book: {BOOK_NAME}")
print(f"Model: {MODEL_NAME}")
print(f"Retrieval Top-K: {TOP_K}")
print(f"Questions: {len(test_dataset)}")

print("\n")


results = []


# ---------------------------------------------------------
# PROCESS QUESTIONS
# ---------------------------------------------------------

for index, item in enumerate(test_dataset, start=1):

    question_id = item["id"]
    question = item["question"]

    relevant_pages = set(item["relevant_pages"])


    # -----------------------------------------------------
    # RETRIEVE CONTEXT
    # -----------------------------------------------------

    retrieved_documents = store.max_marginal_relevance_search(
        question,
        k=TOP_K,
        fetch_k=TOP_K * 4,
        filter={"document_id": document_id},
    )


    # -----------------------------------------------------
    # FORMAT CONTEXT
    # -----------------------------------------------------

    context_parts = []

    retrieved_pages = []

    for document in retrieved_documents:

        page = document.metadata.get("page_number")

        if page is not None:
            retrieved_pages.append(page)

        context_parts.append(
            f"[Page {page}]\n{document.page_content}"
        )


    context = "\n\n".join(context_parts)


    # -----------------------------------------------------
    # GENERATE ANSWER
    # -----------------------------------------------------

    prompt = f"""
You are evaluating a Retrieval-Augmented Generation system.

Answer the question using ONLY the provided context.

Do not use outside knowledge.

If the context does not contain enough information to answer the
question, explicitly say that the context does not contain enough
information.

Question:
{question}

Context:
{context}

Give a concise, accurate answer.
"""


    response = llm.invoke(prompt)

    answer = response.content


    # -----------------------------------------------------
    # EVALUATE ANSWER WITH LLM JUDGE
    # -----------------------------------------------------

    judge_prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation system.

Evaluate the generated answer using ONLY the provided context.

Question:
{question}

Context:
{context}

Generated Answer:
{answer}

Give scores from 1 to 5 for:

1. Faithfulness:
Does the answer contain only claims supported by the context?

2. Answer Relevancy:
Does the answer directly address the question?

3. Correctness:
Is the answer factually correct according to the context?

Return ONLY valid JSON in this exact format:

{{
    "faithfulness": 1,
    "answer_relevancy": 1,
    "correctness": 1,
    "reason": "short explanation"
}}

Do not include Markdown.
"""


    judge_response = llm.invoke(judge_prompt)

    judge_text = judge_response.content.strip()


    # -----------------------------------------------------
    # PARSE JUDGE RESULT
    # -----------------------------------------------------

    try:

        evaluation = json.loads(judge_text)

    except json.JSONDecodeError:

        evaluation = {
            "faithfulness": 0,
            "answer_relevancy": 0,
            "correctness": 0,
            "reason": "Could not parse evaluator response.",
        }


    faithfulness = evaluation.get("faithfulness", 0)
    answer_relevancy = evaluation.get("answer_relevancy", 0)
    correctness = evaluation.get("correctness", 0)


    results.append(
        {
            "id": question_id,
            "question": question,
            "answer": answer,
            "retrieved_pages": retrieved_pages,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "correctness": correctness,
            "reason": evaluation.get("reason", ""),
        }
    )


    # -----------------------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------------------

    print(f"Question {index}/{len(test_dataset)} — {question_id}")

    print(f"Faithfulness:    {faithfulness}/5")
    print(f"Answer Relevancy: {answer_relevancy}/5")
    print(f"Correctness:     {correctness}/5")

    print("-" * 60)


# ---------------------------------------------------------
# FINAL RESULTS
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL ANSWER EVALUATION")
print("=" * 60)


total_questions = len(results)


average_faithfulness = (
    sum(r["faithfulness"] for r in results)
    / total_questions
)

average_relevancy = (
    sum(r["answer_relevancy"] for r in results)
    / total_questions
)

average_correctness = (
    sum(r["correctness"] for r in results)
    / total_questions
)


print()
print(f"Faithfulness:     {average_faithfulness:.2f}/5")
print(f"Answer Relevancy: {average_relevancy:.2f}/5")
print(f"Correctness:      {average_correctness:.2f}/5")

print("=" * 60)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "answer_evaluation_results.json"

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

    json.dump(
        results,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print(f"Detailed results saved to:")
print(OUTPUT_PATH)