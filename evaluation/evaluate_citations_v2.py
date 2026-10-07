import json
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import search, get_vector_store


load_dotenv()

client = OpenAI()

GENERATOR_MODEL = "gpt-4.1-mini"
JUDGE_MODEL = "gpt-5.6-sol"

DATASET_PATH = os.path.join(
    BASE_DIR,
    "evaluation",
    "test_dataset.json"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "evaluation",
    "citation_evaluation_v2_results.json"
)

TOP_K = 5


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_document_id():
    vector_store = get_vector_store()

    data = vector_store.get(include=["metadatas"])
    metadatas = data.get("metadatas", [])

    if not metadatas:
        raise RuntimeError(
            "No documents found in the Chroma database."
        )

    for metadata in metadatas:
        document_id = metadata.get("document_id")

        if document_id:
            return document_id

    raise RuntimeError(
        "No document_id found in the Chroma database."
    )


def build_context(results):
    context_parts = []

    for i, doc in enumerate(results, start=1):

        metadata = doc.metadata

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page_number",
            "?"
        )

        context_parts.append(
            f"[Context {i}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def generate_answer(question, document_id):

    results = search(
        question,
        document_id=document_id,
        k=TOP_K
    )

    context = build_context(results)

    prompt = f"""
You are answering a question using a retrieval-augmented
generation system.

Use ONLY the information contained in the retrieved context.

Answer the question clearly and concisely.

Every factual claim based on the retrieved context should
have an inline citation in exactly this format:

(Source: filename, p. X)

Use the actual source filename and page number from the
retrieved context.

Do not invent page numbers.

If the context does not contain enough information to answer
the question, explicitly say that the information is not
available in the provided context.

Question:
{question}

Retrieved Context:
{context}
"""

    response = client.responses.create(
        model=GENERATOR_MODEL,
        input=prompt
    )

    return response.output_text, context


def extract_citations(answer):

    pattern = r"\(Source:\s*([^,]+),\s*p\.\s*(\d+)\)"

    matches = re.findall(
        pattern,
        answer,
        flags=re.IGNORECASE
    )

    citations = []

    for source, page in matches:

        citations.append({
            "source": source.strip(),
            "page": int(page)
        })

    return citations


def judge_citations(
    question,
    answer,
    context,
    citations
):

    citation_text = json.dumps(
        citations,
        indent=2
    )

    prompt = f"""
You are an independent evaluator assessing citations in a
retrieval-augmented generation system.

IMPORTANT:
Do NOT compare the citation against manually assigned
"expected pages."

Instead, determine whether the cited page actually supports
the claim made in the generated answer by examining the
retrieved context.

Question:
{question}

Generated Answer:
{answer}

Extracted Citations:
{citation_text}

Retrieved Context:
{context}

Evaluate the citation quality using these dimensions.

1. Citation Coverage
Did the answer cite the important factual claims?

2. Citation Correctness
Does the cited source/page actually support the claim
associated with the citation?

3. Citation Completeness
Are the important factual claims adequately supported by
citations?

Use a score from 1 to 5:

1 = Very poor
2 = Poor
3 = Partially correct
4 = Good
5 = Excellent

Also provide a binary assessment:

citation_supported = true

ONLY if the cited page in the retrieved context actually
supports the answer's factual claims.

Otherwise:

citation_supported = false

Do not penalize the answer simply because another page might
also contain the same information.

Return ONLY valid JSON in exactly this format:

{{
    "citation_coverage": {{
        "score": 1,
        "reason": "short explanation"
    }},
    "citation_correctness": {{
        "score": 1,
        "reason": "short explanation"
    }},
    "citation_completeness": {{
        "score": 1,
        "reason": "short explanation"
    }},
    "citation_supported": true,
    "reason": "short overall explanation"
}}
"""

    response = client.responses.create(
        model=JUDGE_MODEL,
        input=prompt
    )

    raw = response.output_text.strip()

    try:
        return json.loads(raw)

    except json.JSONDecodeError:

        print("\nWARNING: Judge returned invalid JSON:")
        print(raw)

        return {
            "citation_coverage": {
                "score": None,
                "reason": "Invalid judge response"
            },
            "citation_correctness": {
                "score": None,
                "reason": "Invalid judge response"
            },
            "citation_completeness": {
                "score": None,
                "reason": "Invalid judge response"
            },
            "citation_supported": None,
            "reason": "Invalid judge response"
        }


def main():

    print("=" * 70)
    print("RAG CITATION ACCURACY EVALUATION - V2")
    print("=" * 70)

    dataset = load_dataset()

    print(f"\nQuestions: {len(dataset)}")
    print(f"Generator: {GENERATOR_MODEL}")
    print(f"Independent Judge: {JUDGE_MODEL}")
    print(f"Top-K retrieval: {TOP_K}")

    document_id = get_document_id()

    print("\nDocument ID found.")
    print("Starting evaluation...\n")

    results = []

    for index, item in enumerate(dataset, start=1):

        question_id = item["id"]
        question = item["question"]

        print(
            f"[{index}/{len(dataset)}] "
            f"Evaluating {question_id}: {question}"
        )

        try:

            answer, context = generate_answer(
                question,
                document_id
            )

            citations = extract_citations(
                answer
            )

            evaluation = judge_citations(
                question,
                answer,
                context,
                citations
            )

            result = {
                "id": question_id,
                "question": question,
                "answer": answer,
                "citations": citations,
                "evaluation": evaluation
            }

            results.append(result)

            print(
                f"  Coverage: "
                f"{evaluation['citation_coverage']['score']}/5"
            )

            print(
                f"  Correctness: "
                f"{evaluation['citation_correctness']['score']}/5"
            )

            print(
                f"  Completeness: "
                f"{evaluation['citation_completeness']['score']}/5"
            )

            print(
                f"  Supported: "
                f"{evaluation['citation_supported']}"
            )

        except Exception as e:

            print(f"  ERROR: {e}")

            results.append({
                "id": question_id,
                "question": question,
                "error": str(e)
            })

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    coverage_scores = []
    correctness_scores = []
    completeness_scores = []

    supported_count = 0
    supported_total = 0

    for result in results:

        evaluation = result.get(
            "evaluation"
        )

        if not evaluation:
            continue

        coverage = evaluation[
            "citation_coverage"
        ]["score"]

        correctness = evaluation[
            "citation_correctness"
        ]["score"]

        completeness = evaluation[
            "citation_completeness"
        ]["score"]

        supported = evaluation.get(
            "citation_supported"
        )

        if isinstance(
            coverage,
            (int, float)
        ):
            coverage_scores.append(
                coverage
            )

        if isinstance(
            correctness,
            (int, float)
        ):
            correctness_scores.append(
                correctness
            )

        if isinstance(
            completeness,
            (int, float)
        ):
            completeness_scores.append(
                completeness
            )

        if isinstance(
            supported,
            bool
        ):
            supported_total += 1

            if supported:
                supported_count += 1

    print("\n" + "=" * 70)
    print("FINAL CITATION RESULTS - V2")
    print("=" * 70)

    if coverage_scores:

        print(
            f"Citation Coverage: "
            f"{sum(coverage_scores) / len(coverage_scores):.2f}/5"
        )

    if correctness_scores:

        print(
            f"Citation Correctness: "
            f"{sum(correctness_scores) / len(correctness_scores):.2f}/5"
        )

    if completeness_scores:

        print(
            f"Citation Completeness: "
            f"{sum(completeness_scores) / len(completeness_scores):.2f}/5"
        )

    if supported_total:

        support_percentage = (
            supported_count
            / supported_total
            * 100
        )

        print(
            f"Citation Support Rate: "
            f"{support_percentage:.2f}% "
            f"({supported_count}/{supported_total})"
        )

    print("\nDetailed results saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()