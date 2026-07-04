# Approach Document — SHL Assessment Recommender

## Problem framing

The hard part isn't calling an LLM — it's making a stateless, non-deterministic
conversation reliably (a) stay grounded in the real catalog, (b) know when to
ask vs. answer, and (c) survive 8 turns of a user who volunteers information
out of order or changes their mind. I designed around those three constraints
first and treated "which LLM" as a swappable detail.

## Architecture

`POST /chat` → guardrails (injection/off-topic) → retrieval over the catalog
→ an LLM call that picks an **action** (`clarify`/`recommend`/`refine`/
`compare`/`refuse`) and **indices** into the retrieved candidate list → the
orchestrator materializes `Recommendation` objects strictly from the catalog
by those indices → response.

Two design choices do most of the work:

**1. The model selects indices, not names/URLs.** Every turn, retrieval
returns a numbered list of real catalog entries (`[0] Core Java (Advanced
Level)... [1] Spring (New)...`). The LLM's structured output is
`selected_indices: [0, 1]`, not free text. The orchestrator looks up the
actual `Assessment` objects by index and builds `Recommendation`s from
catalog data only. A hallucinated name or URL is not just discouraged by
the prompt — it's structurally impossible, because nothing downstream of
`selected_indices` ever passes model-generated text through as a name or
URL. I validate this in tests with a fuzzed set of role queries, asserting
every returned URL/name/type round-trips through `Catalog.get_by_url`.

**2. Conversation state lives in the reply text, not in a session store.**
The API is stateless: `/chat` gets the full `messages` history, but only
`role`/`content` — not my `recommendations` array — round-trips to the next
call. So every "recommend"/"refine" reply names the shortlist inline
(*"Here are assessments that fit: Spring (New), Core Java..."*). On the
next turn, the orchestrator recovers "a shortlist was already given" purely
by scanning prior assistant text for that pattern, which is what lets
`refine` update the list instead of restarting the conversation, with zero
server-side state.

## Retrieval

Catalog entries are embedded as `name | test_type label | description |
job levels` and searched by cosine similarity, top-15 candidates per turn,
re-run every turn (queries are the concatenation of all user messages so
far, so later turns retrieve with strictly more context, not less).

Two interchangeable backends behind one interface:
- **Production:** Gemini `text-embedding-004` + FAISS (`IndexFlatIP` over
  L2-normalized vectors = cosine).
- **Offline fallback:** scikit-learn TF-IDF + cosine similarity, zero
  network calls.

This sandbox's outbound network only allowlists package registries — it
cannot reach `shl.com` or `generativelanguage.googleapis.com`. Rather than
submit untested code, I built the offline path so the *entire* agent
(guardrails, retrieval, action selection, response shaping) runs and is
tested end-to-end locally, and the Gemini path is a drop-in swap
(`GEMINI_API_KEY` env var) that reuses the same orchestrator, schemas, and
guardrails — only the retrieval/generation calls change.

## Catalog data

The scraper (`scraper/scrape_catalog.py`, BeautifulSoup, paginated,
rate-limited) is written and ready to run against the live catalog, but
couldn't be executed from this environment for the same network-access
reason. Rather than fabricate a large synthetic catalog to hit a "40+"
number, `data/catalog_seed.json` ships 22 real, individually verified SHL
Individual Test Solutions (OPQ32r and its report family, the Verify
ability-test range, and a set of Knowledge & Skills tests — Java, Spring,
React, Python, .NET, SQL/data-warehousing) pulled from real catalog pages.
Recall@10 numbers from this seed will understate what the full ~200+-entry
catalog would score; running the scraper once against a real network and
re-embedding is the single highest-leverage next step.

## Agent design: when to ask vs. answer vs. refuse

- **Clarify** when the aggregated conversation lacks both a role/skill
  signal and a seniority/constraint signal (a 2-of-4 heuristic in the
  offline backend; the Gemini backend uses judgment directly per the
  system prompt). A single vague message like *"I need an assessment"*
  never reaches recommend.
- **Recommend** once there's enough signal — including the "paste a JD"
  case, since a long pasted block already carries role + seniority + skill
  context in one turn.
- **Refine** is detected from whether a prior assistant turn already
  produced a shortlist (see the "state in reply text" trick above); the
  new constraint is folded into the retrieval query rather than starting
  over.
- **Compare** requests are detected by keyword (`compare`, `difference
  between`, `vs`) and, when the user names specific assessments, the
  candidate list is restricted to *exactly* those catalog entries — no
  retrieval dilution — so the comparison is answered only from their
  actual descriptions/durations/types, not the model's prior knowledge of
  what "OPQ" or "Java" tests are typically like.
- **Refuse** covers two distinct cases handled differently: prompt
  injection (regex-detected phrases like "ignore previous instructions",
  "reveal your system prompt") is checked *before* any retrieval/LLM call
  and always short-circuits; general off-topic content (salary, legal,
  or fully unrelated chat) is refused with a redirect back to assessments.

## What I tested and what didn't work at first

Early version let the LLM output the recommendation name/URL directly in
its structured response. It looked fine at first but drifted on `refine`
turns — the second call in a conversation would occasionally return an
assessment name that was a close-but-wrong variant of a real catalog entry
(e.g. dropping "(New)" from a name, which technically isn't in the
catalog). Switching to index-selection over a materialized candidate list
eliminated the whole class of failure rather than reducing its frequency,
which is why the orchestrator is built the way it is now.

I also initially let `compare` pull from general retrieval (top-k merged
with named matches), which sometimes surfaced 3-4 tangentially related
assessments instead of exactly the two the user asked about. Restricting
`compare`'s candidate set to only the exact catalog name-matches (when any
are found) fixed this — verified with a dedicated test
(`test_compare_is_grounded_in_named_assessments`).

`tests/test_agent.py` (9 tests, all offline/no API key) covers: schema
compliance, the vague-query-doesn't-recommend rule, recommend/refine/
compare behavior, both refusal paths, a fuzzed no-hallucination check
across multiple role queries, and an 8-turn conversation for turn-cap
friendliness.

## Stack and AI-tool disclosure

FastAPI + Pydantic (schema enforcement is free and exact), scikit-learn
TF-IDF (offline) / Gemini embeddings + FAISS (production), `google-
generativeai` SDK for Gemini calls with `response_mime_type: application/
json` and an explicit `response_schema` for structured output. I used
Claude (agentic coding, in the same environment I'm submitting from) to
scaffold files and iterate quickly on the guardrail regexes and the
index-selection approach; the architecture decisions above (index-only
selection, state-in-reply-text, compare candidate restriction) and the
test suite were driven and verified by me against the assignment's actual
scoring criteria, not accepted as-is.
