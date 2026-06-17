from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re
from collections import Counter

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

TEAM_ALIASES = {
    "MI": "Mumbai Indians",
    "CSK": "Chennai Super Kings",
    "RCB": "Royal Challengers Bangalore",
    "SRH": "Sunrisers Hyderabad",
    "KKR": "Kolkata Knight Riders",
    "DC": "Delhi Capitals",
    "RR": "Rajasthan Royals",
    "PBKS": "Punjab Kings",
    "LSG": "Lucknow Super Giants",
    "GT": "Gujarat Titans",
}

PLAYER_NAMES = [
    "Virat Kohli", "Rohit Sharma", "Shubman Gill", "Ruturaj Gaikwad",
    "Sanju Samson", "KL Rahul", "Shreyas Iyer", "Rishabh Pant",
    "Hardik Pandya", "Suryakumar Yadav", "David Warner", "Jos Buttler",
    "Faf du Plessis", "Travis Head", "Abhishek Sharma",
    "Yuzvendra Chahal", "DJ Bravo", "Lasith Malinga", "Piyush Chawla",
    "Jasprit Bumrah", "Amit Mishra", "Sunil Narine", "Harbhajan Singh",
    "Sandeep Sharma", "Kagiso Rabada", "Trent Boult", "Mohammed Shami",
    "Pat Cummins", "Varun Chakravarthy", "Rashid Khan", "MS Dhoni",
    "Chris Gayle", "AB de Villiers", "Alzarri Joseph", "Sohail Tanvir",
    "Shikhar Dhawan", "Heinrich Klaasen",
]

VENUE_NAMES = [
    "Wankhede Stadium", "MA Chidambaram Stadium", "M Chinnaswamy Stadium",
    "Wankhede", "Chidambaram", "Chepauk", "Chinnaswamy",
    "Eden Gardens", "Arun Jaitley Stadium", "IS Bindra Stadium",
    "Sawai Mansingh Stadium", "Rajiv Gandhi Intl. Stadium",
    "Rajiv Gandhi International Stadium", "BRSABV Ekana Stadium",
    "Narendra Modi Stadium", "Uppal", "Mohali", "Dubai",
]

SECTION_KEYWORDS = {
    "batting": [
        "batting", "batter", "batters", "runs", "run tally", "strike rate",
        "centuries", "fifties", "average", "avg", "sr", "top-order",
        "middle-order", "opener", "wk-bat", "highest individual score",
    ],
    "bowling": [
        "bowling", "wickets", "wkts", "economy", "econ", "bowler",
        "best figures", "leg-spin", "off-spin", "mystery spin", "yorker",
        "pace", "fast bowling", "swing bowling", "death bowling",
    ],
    "venue": [
        "venue", "pitch", "stadium", "ground", "dew factor", "dew",
        "avg 1st innings", "batting/bowling", "capacity", "boundary",
        "chasing", "wankhede", "chidambaram", "chinnaswamy", "eden gardens",
        "narendra modi", "uppal", "mohali", "jaipur",
    ],
    "h2h": [
        "head-to-head", "h2h", "matchup", "vs", "played each other",
        "total matches", "team 1 wins", "team 2 wins", "last 5",
        "key factor",
    ],
    "team": [
        "team profiles", "team", "captain", "captains", "coach", "titles",
        "home venue", "squad", "2024 pos", "short",
    ],
    "form": [
        "recent form", "form trend", "last 5 ipl matches", "last 5",
        "match 1", "match 2", "match 3", "match 4", "match 5",
        "avg (last 5)", "current form",
    ],
    "records": [
        "record", "records", "milestone", "highest", "fastest", "most",
        "best", "lowest total", "highest chase", "most sixes",
        "most matches", "most runs", "most wickets", "most centuries",
    ],
    "season": [
        "season-wise", "trend", "temporal", "2019", "2020", "2021",
        "2022", "2023", "2024", "season", "consistent",
    ],
}

SECTION_PRIORITY = [
    "batting", "bowling", "h2h", "venue", "team", "form", "records", "season"
]


def _contains_phrase(text: str, phrase: str) -> bool:
    if len(phrase) <= 4 and phrase.isalnum():
        return re.search(rf"\b{re.escape(phrase.lower())}\b", text) is not None
    return phrase.lower() in text


def detect_chunk_entities(text: str):
    """Return comma-separated entity metadata for Chroma filtering/debugging."""
    lowered = text.lower()

    teams = []
    for short_name, full_name in TEAM_ALIASES.items():
        if _contains_phrase(lowered, short_name) or _contains_phrase(lowered, full_name):
            teams.append(full_name)

    players = [
        name for name in PLAYER_NAMES
        if _contains_phrase(lowered, name) or _contains_phrase(lowered, name.split()[-1])
    ]

    venues = [
        name for name in VENUE_NAMES
        if _contains_phrase(lowered, name)
    ]

    return {
        "teams": ", ".join(sorted(set(teams))),
        "players": ", ".join(sorted(set(players))),
        "venues": ", ".join(sorted(set(venues))),
    }


def score_sections(text: str) -> dict:
    lowered = text.lower()
    scores = {}
    for section, keywords in SECTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if _contains_phrase(lowered, keyword))
        if score:
            scores[section] = score
    return scores


def tag_chunk(chunk) -> str:
    """Assign the strongest section tag based on chunk content keywords."""
    scores = score_sections(chunk.page_content)
    if not scores:
        return "general"

    return max(
        scores,
        key=lambda section: (scores[section], -SECTION_PRIORITY.index(section)
                             if section in SECTION_PRIORITY else -99)
    )


def ingest_pdf(pdf_path: str):
    """Load PDF, split into chunks, tag metadata, store in ChromaDB."""
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    filtered_pages = []

    for page in pages:
        text = page.page_content.lower()

        if any(marker in text for marker in [
            "direct stat lookup",
            "exact record retrieval",
            "evaluation",
            "out-of-corpus",
            "out-of-scope",
            "langgraph architecture guide",
            "multi-agent query scenarios",
            "recommended chunking strategy",
        ]):
            continue

        filtered_pages.append(page)

    pages = filtered_pages
    print(f"Loaded {len(pages)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
        separators=[
            "\n\nSection ",
            "\n\n",
            "\n| ",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
        ],
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks.")

    # Tag each chunk with section and entity metadata used by retrieval ranking.
    for chunk in chunks:
        section_scores = score_sections(chunk.page_content)
        chunk.metadata["section"] = tag_chunk(chunk)
        chunk.metadata["tags"] = ", ".join(sorted(section_scores)) or "general"
        chunk.metadata.update(detect_chunk_entities(chunk.page_content))

    # Summary of tagging
    tag_counts = Counter(c.metadata["section"] for c in chunks)
    print("Chunk distribution:", dict(tag_counts))

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    try:
        existing_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        existing_store.delete_collection()
        print("Cleared existing vector collection before re-ingestion.")
    except Exception as e:
        print("No existing vector collection to clear:", e)

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
if __name__ == "__main__":
    ingest_pdf("data/IPL_LangGraph_RAG_Dataset.pdf")
