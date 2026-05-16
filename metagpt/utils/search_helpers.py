"""
web_chunker.py — Paste a URL, get structured markdown chunks.
 
Chunking priority:
  1. Header boundaries (H1–H4)
  2. Paragraph boundaries (blank lines / \n\n)
  3. Raw word-count splits (max_words fallback)
 
Usage:
    python3 web_chunker.py <url> [--max-words 200] [--out chunks.json]
"""
 
import asyncio
import re
import json
import sys
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig
 
# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────
 
@dataclass
class Chunk:
    index: int
    method: str          # "header" | "paragraph" | "raw"
    heading: str         # nearest ancestor heading (empty if none)
    heading_level: int   # 1-4, 0 if none
    text: str
    word_count: int
 
 
# ─────────────────────────────────────────────
# Step 1 — fetch markdown via crawl4ai
# ─────────────────────────────────────────────
 
async def fetch_markdown(url: str) -> str:
    config = CrawlerRunConfig(word_count_threshold=0)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
    if not result.success:
        raise RuntimeError(f"Crawl failed: {result.error_message}")
    md = result.markdown or ""
    return md
 
 
# ─────────────────────────────────────────────
# Step 2 — clean / normalise markdown
# ─────────────────────────────────────────────
 
def clean_markdown(md: str) -> str:
    # Collapse 3+ blank lines → 2
    md = re.sub(r'\n{3,}', '\n\n', md)
    # Strip trailing whitespace per line
    md = "\n".join(line.rstrip() for line in md.splitlines())
    return md.strip()
 
 
# ─────────────────────────────────────────────
# Step 3 — split into header sections
# ─────────────────────────────────────────────
 
HEADING_RE = re.compile(r'^(#{1,4})\s+(.+)', re.MULTILINE)
 
def split_by_headers(md: str):
    """
    Returns list of (heading_text, heading_level, body_text).
    Content before the first heading gets heading '' / level 0.
    """
    sections = []
    pos = 0
    prev_heading = ""
    prev_level = 0
    for m in HEADING_RE.finditer(md):
        body = md[pos:m.start()].strip()
        if body or sections == []:
            sections.append((prev_heading, prev_level, body))
        prev_heading = m.group(2).strip()
        prev_level = len(m.group(1))
        pos = m.end()
    # tail after last heading
    tail = md[pos:].strip()
    sections.append((prev_heading, prev_level, tail))
    # drop empty preamble if truly empty
    return [(h, lv, b) for h, lv, b in sections if h or b]
 
 
# ─────────────────────────────────────────────
# Step 4 — split a block by paragraphs
# ─────────────────────────────────────────────
 
def split_by_paragraphs(text: str) -> List[str]:
    """Split on one or more blank lines."""
    paras = re.split(r'\n\n+', text)
    return [p.strip() for p in paras if p.strip()]
 
 
# ─────────────────────────────────────────────
# Step 5 — raw word-count split
# ─────────────────────────────────────────────
 
def split_by_words(text: str, max_words: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks or [text]
 
 
# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────
 
def chunk_markdown(md: str, max_words: int = 200) -> List[Chunk]:
    md = clean_markdown(md)
    chunks: List[Chunk] = []
    idx = 0
 
    for heading, level, body in split_by_headers(md):
        if not body:
            continue
 
        paragraphs = split_by_paragraphs(body)
 
        for para in paragraphs:
            wc = len(para.split())
            if wc == 0:
                continue
 
            if wc <= max_words:
                # fits → paragraph chunk
                method = "header" if heading else "paragraph"
                chunks.append(Chunk(
                    index=idx,
                    method=method,
                    heading=heading,
                    heading_level=level,
                    text=para,
                    word_count=wc,
                ))
                idx += 1
            else:
                # too big → raw split
                for piece in split_by_words(para, max_words):
                    pwc = len(piece.split())
                    if pwc == 0:
                        continue
                    chunks.append(Chunk(
                        index=idx,
                        method="raw",
                        heading=heading,
                        heading_level=level,
                        text=piece,
                        word_count=pwc,
                    ))
                    idx += 1
 
    return chunks
 
 
# ─────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────
 
COLOURS = {
    "header":    "\033[36m",   # cyan
    "paragraph": "\033[33m",   # yellow
    "raw":       "\033[35m",   # magenta
    "reset":     "\033[0m",
    "bold":      "\033[1m",
}
 
def print_chunks(chunks: List[Chunk], show_text: bool = True):
    method_counts = {}
    for c in chunks:
        method_counts[c.method] = method_counts.get(c.method, 0) + 1
 
    print(f"\n{COLOURS['bold']}{'─'*60}{COLOURS['reset']}")
    print(f"{COLOURS['bold']}  Total chunks : {len(chunks)}{COLOURS['reset']}")
    for m, n in method_counts.items():
        col = COLOURS.get(m, "")
        print(f"  {col}{m:12s}{COLOURS['reset']} → {n}")
    print(f"{COLOURS['bold']}{'─'*60}{COLOURS['reset']}\n")
 
    if not show_text:
        return
 
    for c in chunks:
        col = COLOURS.get(c.method, "")
        h_label = f"[H{c.heading_level}] {c.heading}" if c.heading else "(no heading)"
        print(f"{col}{'━'*60}{COLOURS['reset']}")
        print(f"{col}Chunk #{c.index:03d}  method={c.method}  words={c.word_count}{COLOURS['reset']}")
        print(f"  {COLOURS['bold']}{h_label}{COLOURS['reset']}")
        preview = c.text[:300] + ("…" if len(c.text) > 300 else "")
        print(f"  {preview}")
        print()

"""
deep_crawl.py — Search-seeded, filtered, content-cleaned deep crawler

Usage:
    python deep_crawl.py "IMDB top rated sci-fi movies 2024"
    python deep_crawl.py "Python asyncio tutorial"

Pipeline:
    1. DuckDuckGo  → seed URLs for the query
    2. BestFirstCrawlingStrategy crawls seeds with:
         • URLPatternFilter      — blocks query-string / junk URLs
         • ContentTypeFilter     — HTML pages only
         • SEOFilter             — pages with relevant meta/headers
         • ContentRelevanceFilter— pages semantically close to query (BM25)
         • KeywordRelevanceScorer— prioritises highest-scoring pages first
    3. PruningContentFilter     — strips boilerplate, keeps article body
    4. Results saved to crawl_results.json
"""

import asyncio
import json
import re
import sys
from urllib.parse import urlparse, urlunparse

try:
    from ddgs import DDGS  # new package name
except ImportError:
    from duckduckgo_search import DDGS  # fallback

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, LLMConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    ContentTypeFilter,
    FilterChain,
    URLPatternFilter,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from pydantic import BaseModel, Field

# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════

_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "of", "to", "in",
    "is", "on", "with", "how", "what", "best", "top",
}

def extract_keywords(query: str) -> list[str]:
    """Split query into meaningful keywords for scorer/filters."""
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def normalize(url: str) -> str:
    p = urlparse(url)
    clean = urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
    return clean.rstrip("/") or clean


# ══════════════════════════════════════════════
#  STRUCTURED EXTRACTION MODEL
#  DeepSeek fills this schema per page via LLMExtractionStrategy.
#  force_json_response=True forces strict JSON output.
# ══════════════════════════════════════════════

class PageResult(BaseModel):
    title:   str = Field(description="The main title of the page")
    snippet: str = Field(description="A concise 1-2 sentence summary of the page content relevant to the search goal")


def make_extraction_strategy(
        query: str,
        LLM_API_KEY,
        LLM_BASE_URL = "https://api.deepseek.com",
        LLM_MODEL    = "deepseek/deepseek-chat") -> LLMExtractionStrategy:
    llm_cfg = LLMConfig(
        provider=LLM_MODEL,
        api_token=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )
    return LLMExtractionStrategy(
        llm_config=llm_cfg,
        schema=PageResult.model_json_schema(),
        extraction_type="schema",
        instruction=(
            f"Extract the page title and a concise snippet relevant to: '{query}'. "
            f"Snippet must be 1-2 sentences, focused only on content matching the goal."
        ),
        force_json_response=True,   # forces DeepSeek to return strict JSON
        input_format="markdown",    # feed cleaned markdown, not raw HTML
        verbose=False,
    )


def clean_markdown(result) -> str:
    """Return fit_markdown (pruned) if available, else raw."""
    md = getattr(result, "markdown", None) or ""
    if hasattr(md, "fit_markdown"):
        return md.fit_markdown or md.raw_markdown or ""
    return str(md)


# ══════════════════════════════════════════════
#  STEP 1 — SEARCH ENGINE
# ══════════════════════════════════════════════

def search_seeds(query: str, n: int) -> list[str]:
    print(f"\n🔍  Searching: {query!r}")
    results = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query, max_results=n, backend="auto",
                region="wt-wt", safesearch="moderate",
            ))
    except Exception as e:
        print(f"  ⚠️  Search failed: {e}")

    urls = [r.get("href") or r.get("link") for r in results]
    urls = [u for u in urls if u]

    if not urls:
        print("  ❌ No results found.")
    else:
        for u in urls:
            print(f"    seed → {u}")
    return urls


# ══════════════════════════════════════════════
#  STEP 2 — BUILD CONFIGS FROM QUERY
# ══════════════════════════════════════════════

def make_browser_cfg() -> BrowserConfig:
    return BrowserConfig(
        browser_type="chromium",
        headless=True,
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        headers={"Accept-Language": "en-US,en;q=0.9"},
        ignore_https_errors=True,
        verbose=False,
    )


def make_run_cfg(
        query: str, 
        keywords: list[str],
        SCORE_THRESHOLD,
        MAX_DEPTH,
        MAX_PAGES,
        SEMAPHORE,
        LLM_API_KEY,
        LLM_BASE_URL,
        LLM_MODEL) -> CrawlerRunConfig:

    filter_chain = FilterChain([
        # 1. Block query-string URLs — cheap URL-only check, no fetch needed
        URLPatternFilter(patterns=["*?*=*"], reverse=True),

        # 2. HTML only — checked via Content-Type header before body download
        ContentTypeFilter(allowed_types=["text/html"]),

        # NOTE: ContentRelevanceFilter and SEOFilter are post-fetch (need full page body).
        # Removed — they blocked the queue waiting for each page to fully load first.
        # Relevance is handled instead by KeywordRelevanceScorer (URL-signal based, free).
    ])

    # Scorer: prioritises pages whose URLs/titles contain query keywords
    # BestFirstCrawlingStrategy visits highest-scoring pages first
    scorer = KeywordRelevanceScorer(keywords=keywords, weight=0.8)

    return CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=MAX_DEPTH,
            include_external=False,
            max_pages=MAX_PAGES,
            filter_chain=filter_chain,
            url_scorer=scorer,
            score_threshold=SCORE_THRESHOLD,
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),

        # LLM extracts structured title+snippet per page (DeepSeek, forced JSON)
        extraction_strategy=make_extraction_strategy(
            query,
            LLM_API_KEY,
            LLM_BASE_URL,
            LLM_MODEL),

        # Content cleaning: prune nav/footer/ads, keep article body
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.4,
                threshold_type="fixed",
                min_word_threshold=10,
            )
        ),

        # Proven browser timing (from working base code)
        wait_until="domcontentloaded",
        page_timeout=8000,
        delay_before_return_html=0,

        semaphore_count=SEMAPHORE,
        cache_mode=CacheMode.BYPASS,
        stream=True,      # pages arrive as they complete (proven pattern)
        verbose=True,
    )


# ══════════════════════════════════════════════
#  STEP 3 — CRAWL  (proven async-for pattern)
# ══════════════════════════════════════════════

async def crawl(
        query: str,
        seeds: list[str],
        LLM_API_KEY,
        MAX_DEPTH = 2,
        MAX_PAGES = 8,
        SEMAPHORE = 3,     # parallel tabs — safe for most sites
        # Filter / scorer thresholds
        SCORE_THRESHOLD = 0.1,   # minimum scorer value to follow a link

        # ── DeepSeek LLM config (used by LLMExtractionStrategy) ──────────────
        LLM_BASE_URL = "https://api.deepseek.com",
        LLM_MODEL    = "deepseek/deepseek-chat"   # litellm provider/model format
        ) -> list[dict]:
    keywords = extract_keywords(query)
    print(f"\n🔑  Keywords: {keywords}")

    run_cfg = make_run_cfg(query, 
        keywords,
        SCORE_THRESHOLD,
        MAX_DEPTH,
        MAX_PAGES,
        SEMAPHORE,
        LLM_API_KEY,
        LLM_BASE_URL,
        LLM_MODEL)
    
    seen_pages  : set[str]   = set()
    seen_urls   : set[str]   = set()   # dedup final results
    results_out : list[dict] = []

    async with AsyncWebCrawler(config=make_browser_cfg()) as crawler:
        try:
            async for result in await crawler.arun_many(seeds, config=run_cfg):
                norm  = normalize(result.url)
                depth = result.metadata.get("depth", 0)
                score = result.metadata.get("score", 0.0)

                # Track all discovered internal links (mirrors proven code)
                for lk in result.links.get("internal", []):
                    href = lk.get("href", "")
                    if href:
                        seen_pages.add(href)

                if not result.success:
                    print(f"[FAIL] {result.url[:70]}")
                    continue

                md = clean_markdown(result)

                # ── LLM structured extraction (title + snippet) ──────
                # extracted_content is set by LLMExtractionStrategy
                title   = None
                snippet = None
                raw_extracted = result.extracted_content
                if raw_extracted:
                    try:
                        extracted = json.loads(raw_extracted)
                        # LLMExtractionStrategy may return a list or a dict
                        if isinstance(extracted, list) and extracted:
                            extracted = extracted[0]
                        title   = extracted.get("title", "").strip() or None
                        snippet = extracted.get("snippet", "").strip() or None
                    except (json.JSONDecodeError, AttributeError):
                        pass

                # Fallback: metadata title + first paragraph
                if not title:
                    title = (result.metadata or {}).get("title", "") or urlparse(result.url).path
                if not snippet:
                    paragraphs = [p.strip() for p in md.split("\n\n") if p.strip()]
                    snippet = paragraphs[0][:300] if paragraphs else ""

                print(f"\n[{depth}] score={score:.2f}  {result.url[:70]}")

                norm_url = normalize(result.url)
                if norm_url in seen_urls:
                    continue   # skip duplicate
                seen_urls.add(norm_url)

                results_out.append({
                    "title":   title.strip(),
                    "link":    result.url,
                    "snippet": snippet,
                })

        except Exception as e:
            print(f"\n[CRAWLER ERROR] {e}")

        finally:
            print(f"\nDone")
            print(f"Pages visited : {len(seen_pages)}")
            print(f"Pages stored  : {len(results_out)}")

    return results_out

async def deep_collect(query, 
        LLM_API_KEY,
        MAX_SEARCH_SEEDS = 5,
        MAX_DEPTH        = 2,
        MAX_PAGES        = 50,
        SEMAPHORE        = 3,     # parallel tabs — safe for most sites
        # Filter / scorer thresholds
        SCORE_THRESHOLD     = 0.1,   # minimum scorer value to follow a link

        # ── DeepSeek LLM config (used by LLMExtractionStrategy) ──────────────
        LLM_BASE_URL = "https://api.deepseek.com",
        LLM_MODEL    = "deepseek/deepseek-chat"   # litellm provider/model format
    ):
    print(f"\n🎯  Goal: {query}")

    seeds = search_seeds(query, MAX_SEARCH_SEEDS)
    if not seeds:
        print("No seeds found.")
        sys.exit(1)

    pages = await crawl(
        query, 
        seeds,
        LLM_API_KEY,
        LLM_BASE_URL=LLM_BASE_URL,
        LLM_MODEL=LLM_MODEL,   # litellm provider/model format
        MAX_DEPTH=MAX_DEPTH,
        MAX_PAGES=MAX_PAGES,
        SEMAPHORE=SEMAPHORE,     # parallel tabs — safe for most sites
        # Filter / scorer thresholds
        SCORE_THRESHOLD=SCORE_THRESHOLD,   # minimum scorer value to follow a link
    )

    # Summary
    print(f"\n{'═'*60}")
    print(f"  DONE — {len(pages)} pages")
    print(f"{'═'*60}")
    for p in pages:
        print(f"  {p['link'][:70]}  →  {p['title'][:50]}")

    # Save
    with open("crawl_results.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"\n💾  Saved → crawl_results.json")

    # Preview top result
    if pages:
        top = pages[0]
        print(f"\n{'─'*60}\n  Top result: {top['link']}\n{'─'*60}")
        print(top["snippet"])