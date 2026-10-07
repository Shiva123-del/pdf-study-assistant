# PDF Study Assistant

Upload a PDF book, ask questions about it, and get answers grounded in the actual
pages — cited down to the page number. Then test yourself with quizzes generated
from the same material.

Built with Python, LangChain, Chroma and the OpenAI API, in a single Gradio app.

---

## Why this is not just another RAG demo

Most "chat with your PDF" projects stop at *embed, retrieve, prompt*. This one deals
with the things that break once you actually use it.

### Retrieval returns diverse passages, not copies of the same one

Chunks overlap by 300 characters, so plain similarity search can return similar
passages repeatedly and waste the context window. This uses
**Maximal Marginal Relevance** (`k=10`, `fetch_k=40`), which trades a little
relevance for diversity:

```python
return get_vector_store().max_marginal_relevance_search(
    query, k=k, fetch_k=max(fetch_k or k * 4, k), filter=document_filter(document_id),
)