from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"


def tag_chunk(chunk) -> str:
    """Assign a section tag based on chunk content keywords."""
    text = chunk.page_content.lower()

    if any(k in text for k in ["batting", "runs", "strike rate", "centuries", "fifties", "top-order", "opener", "wk-bat"]):
        return "batting"
    elif any(k in text for k in ["bowling", "wickets", "economy", "bowler", "leg-spin", "off-spin", "yorker", "pace"]):
        return "bowling"
    elif any(k in text for k in ["head-to-head", "h2h", "matchup", "last 5", "team 1 wins", "team 2 wins"]):
        return "h2h"
    elif any(k in text for k in ["venue", "pitch", "stadium", "dew factor", "avg 1st innings", "batting/bowling"]):
        return "venue"
    elif any(k in text for k in ["recent form", "form trend", "match 1", "match 2", "match 3", "match 4", "match 5"]):
        return "form"
    elif any(k in text for k in ["record", "milestone", "highest", "fastest", "most sixes", "highest chase"]):
        return "records"
    elif any(k in text for k in ["2019", "2020", "2021", "2022", "2023", "2024 pos", "season"]):
        return "season"
    elif any(k in text for k in ["team", "captain", "coach", "titles", "home venue"]):
        return "team"
    else:
        return "general"


def ingest_pdf(pdf_path: str):
    """Load PDF, split into chunks, tag metadata, store in ChromaDB."""
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks.")

    # Tag each chunk with a section label
    for chunk in chunks:
        chunk.metadata["section"] = tag_chunk(chunk)

    # Summary of tagging
    from collections import Counter
    tag_counts = Counter(c.metadata["section"] for c in chunks)
    print("Chunk distribution:", dict(tag_counts))

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"Vector store saved to {CHROMA_DIR}")
    return vectorstore


def load_vectorstore():
    """Load an existing ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )