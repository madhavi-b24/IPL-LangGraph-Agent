from rag.ingest import load_vectorstore

# Load once at module level (shared across all nodes)
_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


def retrieve(query: str, section: str, k: int = 4):
    """
    Retrieve top-k chunks filtered by section metadata.
    Falls back to unfiltered search if section yields no results.
    """
    vs = get_vectorstore()
    try:
        docs = vs.similarity_search(
            query, k=k, filter={"section": section}
        )
        if docs:
            return docs
    except Exception:
        pass

    # Fallback: no filter
    return vs.similarity_search(query, k=k)