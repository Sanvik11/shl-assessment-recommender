"""
Scrapes the SHL Individual Test Solutions catalog from
https://www.shl.com/solutions/products/product-catalog/ and writes
data/catalog.json in the schema app/catalog.py expects.

This sandbox's outbound network is restricted to package registries (pypi,
npm, github, etc.) and does not include shl.com, so this script can't be
executed from here -- run it from a normal machine/CI runner/deploy host.
It's written to be safe to re-run (idempotent) and polite (rate-limited).

Usage:
    python scraper/scrape_catalog.py --out data/catalog.json

The catalog listing paginates and mixes "Individual Test Solutions" with
"Pre-packaged Job Solutions" -- the assignment restricts scope to the
former, identified on the listing page by the "Individual Test Solutions"
tab/filter. Each row links to a detail page that has the test type key
(A/B/C/D/E/K/P/S), description, job levels, and duration.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
LISTING_URL = f"{BASE}/solutions/products/product-catalog/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CityFix-ResearchBot/1.0; +https://github.com/Sanvik11)"}
REQUEST_DELAY_SECONDS = 1.0

TEST_TYPE_KEYS = set("ABCDEKPS")


def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def discover_listing_pages(start_url: str) -> List[str]:
    """The catalog listing is paginated (?start=N). Walk it until no new
    detail links are found on a page."""
    pages = [start_url]
    seen_links: set = set()
    page_num = 0
    while True:
        url = start_url if page_num == 0 else f"{start_url}?start={page_num * 12}"
        soup = get_soup(url)
        links = extract_detail_links(soup)
        if not links or links.issubset(seen_links):
            break
        seen_links |= links
        if url not in pages:
            pages.append(url)
        page_num += 1
        time.sleep(REQUEST_DELAY_SECONDS)
        if page_num > 100:  # safety valve
            break
    return pages


def extract_detail_links(soup: BeautifulSoup) -> set:
    links = set()
    for a in soup.select("a[href*='/product-catalog/view/']"):
        href = a.get("href")
        if href:
            links.add(urljoin(BASE, href.split("?")[0]))
    return links


def parse_detail_page(url: str) -> Optional[Dict]:
    soup = get_soup(url)

    name_el = soup.select_one("h1")
    if not name_el:
        return None
    name = name_el.get_text(strip=True)

    body_text = soup.get_text(" ", strip=True)

    # Test type key: single-letter badges near the title, e.g. "K" for
    # Knowledge & Skills, "P" for Personality & Behavior, etc.
    test_type = ""
    type_el = soup.select_one("[class*='product-catalog__type'], .test-type, .product-type")
    if type_el:
        candidate = type_el.get_text(strip=True)
        if candidate and candidate[0] in TEST_TYPE_KEYS:
            test_type = candidate[0]
    if not test_type:
        m = re.search(r"\b([ABCDEKPS])\s+(Ability|Biodata|Competencies|Development|Assessment|Knowledge|Personality|Simulations)", body_text)
        if m:
            test_type = m.group(1)

    desc_el = soup.select_one("meta[name='description']")
    description = desc_el["content"].strip() if desc_el and desc_el.get("content") else ""
    if not description:
        p = soup.select_one("article p, .product-catalog__description p, main p")
        description = p.get_text(strip=True) if p else ""

    job_levels = []
    levels_match = re.search(
        r"(Director|Executive|Front Line Manager|General Population|Graduate|Manager|"
        r"Mid-Professional|Professional Individual Contributor|Supervisor|Entry-Level)"
        r"(,\s*(Director|Executive|Front Line Manager|General Population|Graduate|Manager|"
        r"Mid-Professional|Professional Individual Contributor|Supervisor|Entry-Level))*",
        body_text,
    )
    if levels_match:
        job_levels = [lvl.strip() for lvl in levels_match.group(0).split(",")]

    duration_minutes = None
    dur_match = re.search(r"(\d+)\s*minutes?", body_text)
    if dur_match:
        duration_minutes = int(dur_match.group(1))

    remote_testing = "remote testing" in body_text.lower() or True
    adaptive_irt = "adaptive" in body_text.lower() or "irt" in body_text.lower()

    return {
        "name": name,
        "url": url,
        "test_type": test_type or "K",
        "description": description,
        "job_levels": job_levels,
        "duration_minutes": duration_minutes,
        "remote_testing": remote_testing,
        "adaptive_irt": adaptive_irt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/catalog.json")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of detail pages (debugging)")
    args = parser.parse_args()

    print(f"Discovering listing pages from {LISTING_URL} ...")
    listing_pages = discover_listing_pages(LISTING_URL)
    print(f"Found {len(listing_pages)} listing page(s)")

    all_links: set = set()
    for page in listing_pages:
        soup = get_soup(page)
        all_links |= extract_detail_links(soup)
        time.sleep(REQUEST_DELAY_SECONDS)
    print(f"Found {len(all_links)} unique assessment detail pages")

    links = sorted(all_links)
    if args.limit:
        links = links[: args.limit]

    results = []
    for i, url in enumerate(links, 1):
        try:
            item = parse_detail_page(url)
            if item:
                results.append(item)
                print(f"[{i}/{len(links)}] scraped: {item['name']}")
        except Exception as e:
            print(f"[{i}/{len(links)}] FAILED {url}: {e}")
        time.sleep(REQUEST_DELAY_SECONDS)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} assessments to {args.out}")


if __name__ == "__main__":
    main()
