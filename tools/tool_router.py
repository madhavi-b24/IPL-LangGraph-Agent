import os
import math
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from tools.base_tool import BaseTool
from tools.team_tool import team_tool
from tools.player_tool import player_tool
from tools.h2h_tool import h2h_tool
from tools.venue_tool import venue_tool
from tools.dream11_tool import dream11_tool
from tools.prediction_tool import prediction_tool
from rag.ingest import EMBED_MODEL

TOOL_CONFIDENCE_THRESHOLD = 0.55

class ToolRouter:
    def __init__(self):
        self.tools = [
            team_tool,
            player_tool,
            h2h_tool,
            venue_tool,
            dream11_tool,
            prediction_tool,
        ]
        self.embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.tool_embeddings = self._embed_tool_descriptions()

    def _tool_prompt(self, tool: BaseTool) -> str:
        tool_name = tool.name.replace("_", " ")
        return (
            f"{tool_name}: {tool.description}. "
            f"Use this tool for queries about {tool.description}. "
            f"Relevant keywords: {tool.description}, {tool_name}, "
            f"stats, records, analytics, performance, predictions."
        )

    def _embed_tool_descriptions(self):
        descriptions = [self._tool_prompt(tool) for tool in self.tools]
        vectors = self.embedding.embed_documents(descriptions)
        return {tool.name: vector for tool, vector in zip(self.tools, vectors)}

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def select_tool(self, query: str) -> dict:
        query_prompt = f"User query: {query}. Find the best tool by semantic meaning and intent."
        query_vector = self.embedding.embed_query(query_prompt)

        best_tool = None
        best_score = -1.0

        for tool in self.tools:
            tool_vector = self.tool_embeddings[tool.name]
            score = self._cosine_similarity(query_vector, tool_vector)
            if score > best_score:
                best_tool = tool
                best_score = score

        confidence = float(best_score) if best_score >= 0.0 else 0.0
        if best_tool is None or confidence < TOOL_CONFIDENCE_THRESHOLD:
            return {
                "selected_tool": None,
                "confidence": confidence,
            }

        return {
            "selected_tool": best_tool.name,
            "confidence": confidence,
        }
