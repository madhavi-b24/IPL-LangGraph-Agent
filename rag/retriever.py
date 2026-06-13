from rag.ingest import load_vectorstore

# Load once at module level (shared across all nodes)
_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


def retrieve(query: str, section: str, k: int = 4):

    print("\n========== RETRIEVAL ==========")
    print("Query:", query)
    print("Section:", section)

    vs = get_vectorstore()

    try:
        docs = vs.similarity_search(
            query,
            k=k,
            filter={"section": section}
        )

        print("Filtered docs:", len(docs))

        if docs:
            print("First doc:")
            print(docs[0].page_content[:300])
            return docs

    except Exception as e:
        print("Filter Error:", e)

    docs = vs.similarity_search(query, k=k)

    print("Fallback docs:", len(docs))

    if docs:
        print("First doc:")
        print(docs[0].page_content[:300])

    print("==============================\n")

    return docs