
# 🏏 IPL Intelligence Assistant
A Multi-Agent Retrieval-Augmented Generation (RAG) system built with LangGraph, Groq Llama 3, ChromaDB, and Streamlit that answers IPL cricket queries using specialized AI agents.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?style=flat)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.1-orange?style=flat)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple?style=flat)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📖 Table of Contents

- Overview
- Features
- Why This Project?
- System Architecture
- Query Routing Examples
- Tech Stack
- Project Structure
- Installation
- Usage
- Sample Queries
- Challenges Solved
- Future Improvements
- License

---
## 📌 Overview

IPL Intelligence Assistant is a Multi-Agent RAG application designed to answer IPL-related cricket questions using LangGraph.

Unlike traditional RAG systems that use a single retrieval chain, this project routes user queries through specialized agents that focus on different IPL knowledge domains such as:

- Batting Statistics
- Bowling Statistics
- Venue Analysis
- Head-to-Head Records
- Recent Form Analysis
- Team Profiles
- IPL Records & Milestones
- Data Validation

The system retrieves relevant IPL information from a custom IPL dataset stored in ChromaDB and generates contextual answers using Groq's Llama 3 model.

---

## 🎯 Project Goal

The goal of this project is to demonstrate how LangGraph can be used to build a Multi-Agent RAG system capable of handling different IPL-related queries through specialized retrieval agents.

Instead of using a single retriever for all questions, the system dynamically routes queries to domain-specific nodes such as batting, bowling, venue, form, head-to-head, and records.
---

## ✨ Features
### Multi-Agent Architecture

- Query routing using LangGraph
- Specialized cricket knowledge agents
- Dynamic workflow execution

### Retrieval-Augmented Generation

- ChromaDB vector database
- HuggingFace embeddings
- Context-aware retrieval

### Validation Layer

- Conflict detection
- Answer verification
- Reliability improvements

### Interactive UI

- Streamlit-based interface
- Query trace visualization
- Retrieved context inspection

---

## 💡 Why This Project?

Traditional RAG systems follow:

User Query
    ↓
Retriever
    ↓
LLM
    ↓
Answer

This works for simple lookups but struggles with complex cricket analysis.

For example:

Suggest a Dream11 XI for MI vs SRH

requires:

- Player form
- Batting stats
- Bowling stats
- Venue conditions
- Team matchups

LangGraph enables these tasks by routing information through multiple specialized agents and combining their outputs before generating a final answer.

---
## 🏗️ System Architecture

``` mermaid
flowchart TD

    A[User Query] --> B[RouterNode]

    B --> C[TeamProfileNode]
    B --> D[BattingStatsNode]
    B --> E[BowlingStatsNode]
    B --> F[VenueNode]
    B --> G[H2HNode]
    B --> H[FormNode]
    B --> I[RecordsNode]

    C --> J[SynthesisNode]
    D --> J
    E --> J
    F --> J
    G --> J
    H --> J
    I --> J

    J --> K[ValidationNode]
    K --> L[Final Answer]
```
---
### Workflow

1. RouterNode classifies the incoming query.
2. Relevant domain nodes retrieve context from ChromaDB.
3. SynthesisNode combines retrieved information.
4. ValidationNode checks for inconsistencies.
5. Final answer is returned to the user.
---

# 🔄 Query Routing Examples

Team Query

Who captains Chennai Super Kings in 2024?

RouterNode
→ TeamProfileNode
→ SynthesisNode
→ ValidationNode

Batting Query

What is Virat Kohli's IPL run tally?

RouterNode
→ BattingStatsNode
→ SynthesisNode
→ ValidationNode

Bowling Query

List bowlers with economy below 7.0

RouterNode
→ BowlingStatsNode
→ SynthesisNode
→ ValidationNode

Dream11 Query

Suggest a Dream11 XI for MI vs SRH

RouterNode
→ FormNode
→ BattingStatsNode
→ BowlingStatsNode
→ VenueNode
→ SynthesisNode
→ ValidationNode

Match Prediction Query

Who will win MI vs CSK?

RouterNode
→ H2HNode
→ FormNode
→ VenueNode
→ BattingStatsNode
→ BowlingStatsNode
→ SynthesisNode
→ ValidationNode

---

# 🛠️ Tech Stack

| Layer | Tool | Reason |
|---|---|---|
| Agent Orchestration | LangGraph 0.2+ | Stateful multi-agent graph |
| LLM | Groq — Llama 3.1 8B Instant | Fast, free inference |
| Vector Store | ChromaDB | Local persistent embeddings |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Lightweight, accurate |
| PDF Loading | PyPDF + LangChain | Structured document ingestion |
| UI | Streamlit | Rapid interactive interface |
| Language | Python 3.10+ | |


---
## 📁 Project Structure

```
ipl-langgraph-rag/
├── .env                        # API keys — never commit this
├── .env.example                # Safe template for reviewers
├── .gitignore
├── requirements.txt
├── setup_venv.sh               # One-command environment setup
├── main.py                     # Terminal test runner
├── app.py                      # Streamlit UI
├── data/
│   └── IPL_LangGraph_RAG_Dataset.pdf
├── graph/
│   ├── __init__.py
│   ├── state.py                # IPLAgentState TypedDict
│   ├── nodes.py                # All core agent nodes
│   ├── team_node.py            # TeamProfileNode
│   ├── validation.py           # ValidationNode
│   └── graph_builder.py        # Full graph wiring
└── rag/
    ├── __init__.py
    ├── ingest.py               # PDF → chunks → ChromaDB
    └── retriever.py            # Metadata-filtered retrieval
```

---


## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or higher
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ipl-langgraph-rag.git
cd ipl-langgraph-rag
```

### Step 2 — Create virtual environment

```bash
bash setup_venv.sh
```

This creates `venv/` and installs all packages from `requirements.txt` automatically.

**Activate manually in future sessions:**

```bash
# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Set your API key

Copy `.env.example` to `.env` and add your Groq key:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
GROQ_API_KEY=your_actual_key_here
```

### Step 4 — Add the dataset

Place `IPL_LangGraph_RAG_Dataset.pdf` inside the `data/` folder.

### Step 5 — Run

```bash
# Terminal test — builds vector store on first run (~2 minutes)
python main.py

# Streamlit UI
streamlit run app.py
```

The first run downloads the embedding model and builds the ChromaDB vector store. Subsequent runs are instant.

---

# 🚀 Usage

Example Queries

Who captains Chennai Super Kings in 2024?

What is Virat Kohli's IPL run tally?

Which opener has the highest strike rate?

List bowlers with economy below 7.0

Suggest a Dream11 XI for MI vs SRH

Who will win MI vs CSK?

---

# 💬 Sample Queries

| Difficulty | Query |
|------------|-------|
| Easy | Who captains Chennai Super Kings in 2024? |
| Easy | What is Virat Kohli's IPL run tally? |
| Easy | What is the highest team total in IPL history? |
| Medium | Which opener has the highest strike rate? |
| Medium | List bowlers with economy below 7.0 |
| Medium | How many times have MI and CSK played each other? |
| Hard | Suggest a Dream11 XI for MI vs SRH at Wankhede |
| Hard | Who will win MI vs CSK? |
| Hard | What bowling strategy should SRH use against CSK? |
| Expert | Detect conflicting IPL statistics in the dataset |
---
## 🏆 Key Achievements

- Built a Multi-Agent RAG system using LangGraph
- Implemented metadata-based retrieval
- Added conflict detection using ValidationNode
- Created a Streamlit interface for interaction
- Developed dynamic query routing across multiple IPL domains
- Built a complete end-to-end AI application from ingestion to answer generation
---
# 🧩 Challenges Solved

Multi-Agent Routing

Implemented query classification and dynamic node activation using LangGraph.

Metadata-Based Retrieval

Added section tagging:

- batting
- bowling
- venue
- h2h
- form
- records
- season
- team

for targeted retrieval.

Validation Layer

Created a ValidationNode that checks for conflicting information before presenting answers.

Explainability

Users can inspect:

- Routing decisions
- Activated nodes
- Retrieved chunks
- Validation results

---

# 🔮 Future Improvements

- Hybrid Search (BM25 + Vector Search)
- Query Expansion and Rewriting
- Cross-Encoder Reranking
- Live IPL Data Integration
- Real-Time Match Prediction
- Advanced Fantasy Team Recommendations

---

# 📄 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project with proper attribution.
