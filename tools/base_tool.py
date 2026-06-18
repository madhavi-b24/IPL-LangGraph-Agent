from dataclasses import dataclass


@dataclass(frozen=True)
class BaseTool:
    name: str
    description: str
