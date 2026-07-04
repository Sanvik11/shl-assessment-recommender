"""
Loads the SHL Individual Test Solutions catalog into memory.

Catalog JSON is produced by scraper/scrape_catalog.py in a full run (see README).
A verified seed of real, scraped entries ships in data/catalog_seed.json so the
service is runnable and testable without network access to shl.com.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_CATALOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "catalog.json"
)
SEED_CATALOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "catalog_seed.json"
)

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


@dataclass
class Assessment:
    name: str
    url: str
    test_type: str
    description: str = ""
    job_levels: List[str] = field(default_factory=list)
    duration_minutes: Optional[int] = None
    remote_testing: bool = True
    adaptive_irt: bool = False

    def to_recommendation(self) -> dict:
        return {"name": self.name, "url": self.url, "test_type": self.test_type}

    def searchable_text(self) -> str:
        """Text blob used for embedding / retrieval."""
        type_label = TEST_TYPE_LABELS.get(self.test_type, self.test_type)
        parts = [
            self.name,
            type_label,
            self.description,
            " ".join(self.job_levels),
        ]
        return " | ".join(p for p in parts if p)


class Catalog:
    def __init__(self, assessments: List[Assessment]):
        self.assessments = assessments
        self._by_url: Dict[str, Assessment] = {a.url: a for a in assessments}
        self._by_name_lower: Dict[str, Assessment] = {
            a.name.lower(): a for a in assessments
        }

    def __len__(self) -> int:
        return len(self.assessments)

    def get_by_url(self, url: str) -> Optional[Assessment]:
        return self._by_url.get(url)

    def find_by_name(self, name: str) -> Optional[Assessment]:
        """Exact (case-insensitive) then substring match."""
        name_l = name.strip().lower()
        if name_l in self._by_name_lower:
            return self._by_name_lower[name_l]
        for a in self.assessments:
            if name_l in a.name.lower() or a.name.lower() in name_l:
                return a
        return None

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Catalog":
        if path is None:
            path = DEFAULT_CATALOG_PATH if os.path.exists(DEFAULT_CATALOG_PATH) else SEED_CATALOG_PATH
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        assessments = [Assessment(**item) for item in raw]
        return cls(assessments)
