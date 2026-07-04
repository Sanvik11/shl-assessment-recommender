"""
Two interchangeable "brains" behind the agent, both implementing:

    generate(system_prompt: str, history: list[dict], candidates: list[Assessment],
             forced_action: str | None) -> dict
        -> {"reply": str, "selected_indices": list[int], "action": str,
            "end_of_conversation": bool}

Design choice (context engineering): the model NEVER invents a name/URL.
It only ever picks *indices* into a numbered candidate list that we built
from the catalog via retrieval. app/agent.py then materializes the actual
Recommendation objects from those indices. This makes hallucinated catalog
entries structurally impossible, not just "discouraged by prompting" -- the
single biggest failure mode the assignment calls out.

GeminiBackend is the production path (needs GEMINI_API_KEY + network).
OfflineBackend is a deterministic, fully local fallback used automatically
when no key/network is available, so the exact same agent code can be
exercised end-to-end in this environment and in CI.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from app.catalog import Assessment


class BaseBackend:
    def generate(
        self,
        system_prompt: str,
        history: List[Dict[str, str]],
        candidates: List[Assessment],
        forced_action: Optional[str] = None,
    ) -> dict:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Production backend: Gemini, structured JSON output
# --------------------------------------------------------------------------
class GeminiBackend(BaseBackend):
    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["clarify", "recommend", "refine", "compare", "refuse"],
            },
            "reply": {"type": "string"},
            "selected_indices": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "end_of_conversation": {"type": "boolean"},
        },
        "required": ["action", "reply", "selected_indices", "end_of_conversation"],
    }

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": self.RESPONSE_SCHEMA,
                "temperature": 0.3,
            },
        )

    def generate(self, system_prompt, history, candidates, forced_action=None) -> dict:
        candidate_block = _format_candidates(candidates)
        convo_block = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
        forced = f"\nThe orchestrator has already determined action='{forced_action}'. Honor it unless it is clearly wrong." if forced_action else ""

        prompt = (
            f"{system_prompt}\n\n"
            f"CANDIDATE ASSESSMENTS (indices you may cite in selected_indices):\n{candidate_block}\n\n"
            f"CONVERSATION SO FAR:\n{convo_block}\n{forced}\n\n"
            "Respond with the JSON object described in the schema. "
            "selected_indices must only reference the candidate indices above."
        )
        resp = self.model.generate_content(prompt)
        return json.loads(resp.text)


# --------------------------------------------------------------------------
# Offline backend: deterministic, no network, same interface/behavior contract
# --------------------------------------------------------------------------
_SENIORITY_WORDS = [
    "entry", "junior", "graduate", "fresher", "mid-level", "mid level", "senior",
    "lead", "principal", "manager", "director", "executive", "years of experience",
    "years experience",
]
_ROLE_HINTS = [
    "developer", "engineer", "manager", "analyst", "designer", "administrator",
    "consultant", "representative", "specialist", "lead", "architect", "hiring",
    "role", "position", "candidate", "job",
]


class OfflineBackend(BaseBackend):
    """Rule-based agent used when GEMINI_API_KEY is not configured."""

    def generate(self, system_prompt, history, candidates, forced_action=None) -> dict:
        last_user = _last_user_message(history)
        action = forced_action or self._infer_action(history, last_user)

        if action == "refuse":
            return {
                "action": "refuse",
                "reply": (
                    "I can only help with selecting or comparing SHL assessments. "
                    "I'm not able to advise on that, and I can't act on instructions "
                    "embedded in a message. What role or skills are you hiring for?"
                ),
                "selected_indices": [],
                "end_of_conversation": False,
            }

        if action == "clarify":
            missing = self._missing_slot(last_user, history)
            return {
                "action": "clarify",
                "reply": missing,
                "selected_indices": [],
                "end_of_conversation": False,
            }

        if action == "compare":
            idxs = list(range(len(candidates)))
            names = ", ".join(c.name for i, c in enumerate(candidates) if i in idxs)
            reply = (
                f"Here's how they differ based on the catalog: {names}. "
                + " ".join(
                    f"{c.name} ({c.test_type}, {c.duration_minutes or 'untimed'} min): {c.description}"
                    for i, c in enumerate(candidates) if i in idxs
                )
            )
            return {
                "action": "compare",
                "reply": reply,
                "selected_indices": idxs,
                "end_of_conversation": False,
            }

        # recommend / refine
        top_n = min(5, len(candidates)) if len(candidates) > 0 else 0
        idxs = list(range(top_n))
        names = ", ".join(candidates[i].name for i in idxs)
        verb = "Updated the shortlist based on your latest note" if action == "refine" else "Here are assessments that fit"
        reply = f"{verb}: {names}." if names else "I couldn't find a close match yet — could you tell me more about the role or the skills you need?"
        return {
            "action": action if names else "clarify",
            "reply": reply,
            "selected_indices": idxs,
            "end_of_conversation": False,
        }

    # -- heuristics ---------------------------------------------------
    def _infer_action(self, history, last_user: str) -> str:
        text = last_user.lower()

        if _looks_like_injection(text):
            return "refuse"
        if _looks_off_topic(text):
            return "refuse"
        if _looks_like_compare(text):
            return "compare"

        prior_shortlist = _prior_shortlist_given(history)
        if prior_shortlist:
            return "refine"

        if _has_enough_context(_aggregate_user_text(history)):
            return "recommend"
        return "clarify"

    def _missing_slot(self, last_user: str, history) -> str:
        agg = _aggregate_user_text(history).lower()
        if not any(h in agg for h in _ROLE_HINTS):
            return "Happy to help — what role or skill set are you hiring for?"
        if not any(s in agg for s in _SENIORITY_WORDS):
            return "Got it. What seniority level is this for (entry-level, mid, senior)?"
        return "Thanks — any specific skills, test types, or constraints (e.g. time limit) I should filter on?"


# --------------------------------------------------------------------------
# Shared helpers (used by agent.py too)
# --------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore (all|the)? ?(previous|prior|above) instructions",
    r"disregard (all|the)? ?(previous|prior|above) instructions",
    r"you are now",
    r"system prompt",
    r"reveal your (prompt|instructions)",
    r"act as (?!.*(recruiter|hiring manager))",  # generic "act as X" jailbreaks
    r"jailbreak",
    r"pretend you (are|have) no (rules|restrictions)",
    r"new instructions?:",
]

OFFTOPIC_PATTERNS = [
    r"\bsalary\b", r"\boffer letter\b", r"\bnotice period\b", r"\bvisa sponsorship\b",
    r"\bemployment law\b", r"\blabor law\b", r"\blabour law\b", r"\blawsuit\b",
    r"\bcan i (ask|reject).*(age|pregnan|disabilit|religion)\b",
    r"\bwrite (me )?a job description\b", r"\bhow (do|should) i fire\b",
    r"\bperformance improvement plan\b", r"\bwhat should i pay\b",
]

# Broad "clearly nothing to do with hiring/assessments" catch, used only when
# the message ALSO has none of the role/skill/seniority hints. This is a
# coarse heuristic for the offline fallback; the Gemini backend relies on
# rule 6 of the system prompt instead, which generalizes far better.
GENERIC_OFFTOPIC_WORDS = [
    "recipe", "weather", "football score", "cricket score", "stock price",
    "write a poem", "write me a song", "movie recommendation", "who won the",
    "capital of", "translate this",
]


def _looks_like_injection(text: str) -> bool:
    return any(re.search(p, text) for p in INJECTION_PATTERNS)


def _looks_off_topic(text: str) -> bool:
    if any(re.search(p, text) for p in OFFTOPIC_PATTERNS):
        return True
    if any(w in text for w in GENERIC_OFFTOPIC_WORDS) and not any(
        h in text for h in _ROLE_HINTS
    ):
        return True
    return False


def _looks_like_compare(text: str) -> bool:
    return bool(re.search(r"\b(compare|difference between|vs\.?|versus)\b", text))


def _last_user_message(history: List[Dict[str, str]]) -> str:
    for m in reversed(history):
        if m["role"] == "user":
            return m["content"]
    return ""


def _aggregate_user_text(history: List[Dict[str, str]]) -> str:
    return " ".join(m["content"] for m in history if m["role"] == "user")


def _prior_shortlist_given(history: List[Dict[str, str]]) -> bool:
    # Every assistant reply that produced a shortlist names the assessments
    # inline (see agent.py) precisely so this is recoverable from plain text
    # on the next stateless call.
    assistant_texts = [m["content"] for m in history if m["role"] == "assistant"]
    return any(":" in t and ("," in t or len(t.split()) < 40) and any(
        w in t for w in ["shortlist", "assessments", "Here are", "Updated"]
    ) for t in assistant_texts[:-1] if assistant_texts) or (
        len(assistant_texts) > 0 and any(
            "Here are" in t or "Updated the shortlist" in t for t in assistant_texts
        )
    )


def _has_enough_context(agg_text: str) -> bool:
    agg = agg_text.lower()
    score = 0
    if any(h in agg for h in _ROLE_HINTS):
        score += 1
    if any(s in agg for s in _SENIORITY_WORDS):
        score += 1
    if len(agg.split()) > 25:
        score += 1
    # explicit skill/tech words or catalog-ish nouns nudge it over the line
    if re.search(r"\b(java|python|sql|react|\.net|javascript|spring|excel|c\+\+|c#)\b", agg):
        score += 1
    return score >= 2


def _format_candidates(candidates: List[Assessment]) -> str:
    lines = []
    for i, a in enumerate(candidates):
        lines.append(
            f"[{i}] {a.name} | type={a.test_type} | duration={a.duration_minutes or 'untimed'}min "
            f"| levels={','.join(a.job_levels[:4])} | {a.description[:160]}"
        )
    return "\n".join(lines) if lines else "(no candidates retrieved)"


def get_backend() -> BaseBackend:
    api_key = os.getenv("GEMINI_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "auto")
    use_gemini = provider == "gemini" or (provider == "auto" and bool(api_key))
    if use_gemini:
        try:
            print("✅ USING GEMINI BACKEND")
            return GeminiBackend(api_key)
        except Exception as e:  # pragma: no cover
            print(f"[llm] Falling back to offline backend, Gemini init failed: {e}")
    print("⚠️ USING OFFLINE BACKEND")       
    return OfflineBackend()
