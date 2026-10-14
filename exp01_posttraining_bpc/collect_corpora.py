"""Build the fresh-text corpora (Aug-Oct 2026) plus the enwik8 control slice.

Usage:  python collect_corpora.py [--target-bytes 100000] [--out data]

Outputs, per corpus NAME:
  data/NAME.txt        UTF-8 text that gets scored
  data/NAME.prov.json  provenance: one record per source document (url, date, bytes) + totals

Corpora:
  wiki2026      English Wikipedia articles created in mainspace in the last 30 days (MediaWiki
                recentchanges rctype=new) whose title contains "2026"; creation date re-verified via
                prop=revisions&rvdir=newer&rvlimit=1 and required >= 2026-08-01.
  arxiv2026     "Title. Abstract" blocks from arXiv cs.LG and math.AP, published >= 2026-08-01.
  gutenberg2026 Fallback for news (en.wikinews.org has had no new mainspace article since May 2026):
                English Project Gutenberg books released >= 2026-08-01, header/footer stripped.
  enwik8_ctrl   Bytes 1,000,000-1,100,000 of enwik8 (old text, seen by every model; sanity control).

All functions are pure except the ones named fetch_* / write_* (network / disk I/O); no globals
besides immutable constants.
"""
from __future__ import annotations

import argparse
import html as htmllib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

UA = "compression-intelligence-pilot/0.1 (research; contact via github)"
MIN_DATE = "2026-08-01"
WIKI_API = "https://en.wikipedia.org/w/api.php"
ARXIV_API = "https://export.arxiv.org/api/query"
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
ATOM = "{http://www.w3.org/2005/Atom}"
DROP_SECTIONS = ("References", "See also", "External links", "Notes", "Further reading",
                 "Citations", "Sources", "Bibliography", "Footnotes")


# ----------------------------------------------------------------------------- I/O helpers
def fetch_bytes(url: str, retries: int = 8, pause: float = 1.0) -> bytes:
    """GET with retries. Wikimedia allows ~3 quick anonymous API calls, then 429 + Retry-After (~17 s)."""
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            time.sleep(pause)  # be polite
            return data
        except Exception as e:
            last = e
            retry_after = getattr(e, "headers", None) and e.headers.get("retry-after")
            wait = float(retry_after) + 1 if retry_after else pause * (2 ** i)
            print(f"retry {i} in {wait:.0f}s {url[:100]}: {e}", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def fetch_json(url: str) -> dict:
    # Wikimedia throttles anonymous API clients hard (429 after ~3 quick calls): pace at 1 call / 6 s
    return json.loads(fetch_bytes(url, pause=6.0).decode("utf-8"))


def api_url(base: str, params: dict) -> str:
    return base + "?" + urllib.parse.urlencode(params)


def write_corpus(out_dir: Path, name: str, text: str, docs: list[dict], notes: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
    prov = {
        "corpus": name,
        "bytes": len(text.encode("utf-8")),
        "chars": len(text),
        "n_docs": len(docs),
        "min_date_required": MIN_DATE if name != "enwik8_ctrl" else None,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": notes,
        "docs": docs,
    }
    (out_dir / f"{name}.prov.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False), encoding="utf-8")
    return {k: prov[k] for k in ("corpus", "bytes", "chars", "n_docs")}


# ----------------------------------------------------------------------------- pure text helpers
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def clean_wiki_extract(text: str) -> str:
    """Cut trailing reference-type sections, drop empty headings, collapse blank lines."""
    lines = text.split("\n")
    out: list[str] = []
    for ln in lines:
        m = re.match(r"^(=+)\s*(.*?)\s*=+$", ln.strip())
        if m and m.group(2) in DROP_SECTIONS:
            break
        out.append(ln.rstrip())
    # drop headings immediately followed by another heading or end (empty sections)
    kept: list[str] = []
    nonblank = list(out)
    for i, ln in enumerate(nonblank):
        if re.match(r"^=+.*=+$", ln.strip()):
            nxt = next((x for x in nonblank[i + 1:] if x.strip()), None)
            if nxt is None or re.match(r"^=+.*=+$", nxt.strip()):
                continue
        kept.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def take_until(blocks: list[tuple[str, dict]], target_bytes: int, sep: str = "\n\n") -> tuple[str, list[dict]]:
    """Concatenate blocks in order until target_bytes is reached; return text + provenance of used blocks."""
    parts: list[str] = []
    used: list[dict] = []
    size = 0
    for text, meta in blocks:
        if size >= target_bytes:
            break
        parts.append(text)
        size += len(text.encode("utf-8")) + len(sep)
        used.append({**meta, "bytes": len(text.encode("utf-8"))})
    return sep.join(parts) + "\n", used


def strip_gutenberg(raw: str) -> str:
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", raw)
    body = raw[start.end() if start else 0: end.start() if end else len(raw)]
    body = body.replace("\r\n", "\n")
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def gutenberg_slice(body: str, skip_chars: int, max_bytes: int) -> str:
    """Skip front matter (title page/contents), start and end at paragraph boundaries."""
    start = body.find("\n\n", max(skip_chars, len(body) // 8))  # past title page / contents / preface
    start = 0 if start < 0 else start + 2
    chunk = body[start:]
    enc = chunk.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
    cut = enc.rfind("\n\n")
    return (enc[:cut] if cut > 0 else enc).strip()


def parse_arxiv_feed(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        out.append({
            "id": e.findtext(f"{ATOM}id"),
            "title": normalize_ws(e.findtext(f"{ATOM}title") or ""),
            "abstract": normalize_ws(e.findtext(f"{ATOM}summary") or ""),
            "published": (e.findtext(f"{ATOM}published") or "")[:10],
        })
    return out


def interleave(*lists: list) -> list:
    out = []
    for i in range(max(len(lst) for lst in lists)):
        for lst in lists:
            if i < len(lst):
                out.append(lst[i])
    return out


# ----------------------------------------------------------------------------- wiki2026
def fetch_new_wiki_pages(max_batches: int = 10) -> list[dict]:
    """Newest mainspace page creations from recentchanges (10x500; deeper paging hits HTTP 429)."""
    pages, cont = [], {}
    for _ in range(max_batches):
        params = {"action": "query", "list": "recentchanges", "rctype": "new", "rcnamespace": 0,
                  "rcshow": "!redirect", "rclimit": 500, "rcprop": "title|timestamp|sizes",
                  "format": "json", **cont}
        d = fetch_json(api_url(WIKI_API, params))
        pages += d["query"]["recentchanges"]
        print(f"recentchanges: {len(pages)} new pages so far", file=sys.stderr, flush=True)
        if "continue" not in d:
            break
        cont = d["continue"]
    return pages


def select_wiki_candidates(pages: list[dict], event_min_len: int = 3000, other_min_len: int = 6000) -> list[dict]:
    """Newest first. Event pages ('2026' in title) first, then any other new article (still text first
    published in Sept-Oct 2026, just not about a 2026 event). Skip sub-event/result/squad/season/list pages."""
    skip = (" – ", " - ", "squad", "qualification", "season", "results", "List of", "list of", "team",
            "discography", "filmography", "(disambiguation)")
    newest = sorted(pages, key=lambda p: p["timestamp"], reverse=True)
    ok = lambda p: not any(k in p["title"] for k in skip)  # noqa: E731
    event = [dict(p, kind="event") for p in newest if "2026" in p["title"] and p.get("newlen", 0) >= event_min_len and ok(p)]
    other = [dict(p, kind="other_new_article") for p in newest
             if "2026" not in p["title"] and p.get("newlen", 0) >= other_min_len and ok(p)]
    seen, out = set(), []
    for p in event + other:
        if p["title"] not in seen:
            seen.add(p["title"])
            out.append(p)
    return out


def fetch_intro_lengths(titles: list[str]) -> dict[str, int]:
    """Length of the plain-text lead section for up to 20 titles per call (cheap prose pre-screen)."""
    out: dict[str, int] = {}
    for i in range(0, len(titles), 20):
        d = fetch_json(api_url(WIKI_API, {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                          "exlimit": 20, "titles": "|".join(titles[i:i + 20]), "format": "json"}))
        for pg in d["query"].get("pages", {}).values():
            out[pg.get("title", "")] = len(pg.get("extract", "") or "")
    return out


def fetch_creation_date(title: str) -> str:
    d = fetch_json(api_url(WIKI_API, {"action": "query", "prop": "revisions", "titles": title,
                                      "rvdir": "newer", "rvlimit": 1, "rvprop": "timestamp",
                                      "format": "json"}))
    page = next(iter(d["query"]["pages"].values()))
    revs = page.get("revisions") or []  # page deleted since creation -> no revisions
    return revs[0]["timestamp"] if revs else ""


def fetch_wiki_extract(title: str) -> tuple[str, int]:
    d = fetch_json(api_url(WIKI_API, {"action": "query", "prop": "extracts|info", "explaintext": 1,
                                      "exsectionformat": "wiki", "titles": title, "format": "json"}))
    page = next(iter(d["query"]["pages"].values()))
    return page.get("extract", ""), page.get("lastrevid", 0)


def build_wiki2026(target: int) -> tuple[str, list[dict], str]:
    all_cands = select_wiki_candidates(fetch_new_wiki_pages())
    screened = all_cands[:400]
    intro = fetch_intro_lengths([p["title"] for p in screened])
    cands = [p for p in screened if intro.get(p["title"], 0) >= 600]  # prose-rich leads only
    print(f"wiki: {len(all_cands)} candidates, {len(cands)} with lead >= 600 chars", file=sys.stderr, flush=True)
    blocks: list[tuple[str, dict]] = []
    size = 0
    for p in cands:
        if size >= target:
            break
        print(f"wiki: {p['title']} ({size} B so far)", file=sys.stderr, flush=True)
        raw, revid = fetch_wiki_extract(p["title"])
        text = clean_wiki_extract(raw)
        if len(text.encode("utf-8")) < 1500:  # mostly tables/infobox -> too little prose
            continue
        created = fetch_creation_date(p["title"])  # verify only pages we keep (API is rate-limited)
        if not created or created[:10] < MIN_DATE:  # deleted page or older creation
            continue
        text = text[:12000].rsplit("\n", 1)[0] if len(text) > 12000 else text  # cap any single article
        body = f"{p['title']}\n\n{text}"
        url = "https://en.wikipedia.org/w/index.php?" + urllib.parse.urlencode({"title": p["title"], "oldid": revid})
        blocks.append((body, {"title": p["title"], "url": url, "date": created, "created": created,
                              "kind": p["kind"]}))
        size += len(body.encode("utf-8"))
    text, used = take_until(blocks, target)
    notes = (f"{len(all_cands)} candidate new mainspace pages from the newest 5000 creations in recentchanges: "
             "event pages ('2026' in title, >=3000 B) first, then other new articles (>=6000 B), newest first, "
             "excluding sub-event/result/squad/season/team/list pages; "
             f"of the first {len(screened)}, {len(cands)} have a plain-text lead >= 600 chars (prose pre-screen); "
             "creation date verified with rvdir=newer; extracts via prop=extracts&explaintext=1; "
             "References/See also/External links sections dropped; each article capped at 12k chars.")
    return text, used, notes


# ----------------------------------------------------------------------------- arxiv2026
def build_arxiv2026(target: int) -> tuple[str, list[dict], str]:
    per_cat = {}
    for cat in ("cs.LG", "math.AP"):
        url = api_url(ARXIV_API, {"search_query": f"cat:{cat}", "sortBy": "submittedDate",
                                  "sortOrder": "descending", "max_results": 200})
        entries = [e for e in parse_arxiv_feed(fetch_bytes(url, pause=3.0)) if e["published"] >= MIN_DATE]
        per_cat[cat] = [(f"{e['title']}. {e['abstract']}",
                         {"url": e["id"], "date": e["published"], "cat": cat, "title": e["title"]})
                        for e in entries]
    text, used = take_until(interleave(per_cat["cs.LG"], per_cat["math.AP"]), target)
    notes = (f"arXiv API, 200 newest per category; kept published>={MIN_DATE} "
             f"(cs.LG {len(per_cat['cs.LG'])}, math.AP {len(per_cat['math.AP'])}); interleaved; "
             "whitespace normalised; LaTeX left as-is.")
    return text, used, notes


# ----------------------------------------------------------------------------- gutenberg2026
def fetch_gutenberg_recent_ids(pages: int = 3) -> list[int]:
    ids: list[int] = []
    for start in range(1, 25 * pages + 1, 25):
        html = fetch_bytes(f"https://www.gutenberg.org/ebooks/search/?sort_order=release_date&start_index={start}",
                           pause=2.0).decode("utf-8", "ignore")
        ids += [int(x) for x in re.findall(r'href="/ebooks/(\d+)"', html)]
    return list(dict.fromkeys(ids))


def parse_gutenberg_meta(html: str) -> dict:
    def field(name: str) -> str:
        m = re.search(rf"<th>{name}</th>\s*<td[^>]*>(.*?)</td>", html, re.S)
        return htmllib.unescape(normalize_ws(re.sub(r"<[^>]+>", " ", m.group(1)))) if m else ""
    return {"title": field("Title"), "language": field("Language"), "release": field("Release Date"),
            "author": field("Author")}


def parse_release_date(s: str) -> str:
    m = re.match(r"(\w+) (\d+), (\d{4})", s)
    if not m:
        return ""
    fmt = "%B %d %Y" if len(m.group(1)) > 3 else "%b %d %Y"  # Gutenberg prints "Oct 13, 2026" or "October 13, 2026"
    return time.strftime("%Y-%m-%d", time.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", fmt))


def build_gutenberg2026(target: int, n_books: int = 4) -> tuple[str, list[dict], str]:
    per_book = target // n_books + 2000
    blocks: list[tuple[str, dict]] = []
    for gid in fetch_gutenberg_recent_ids():
        if len(blocks) >= n_books:
            break
        meta = parse_gutenberg_meta(fetch_bytes(f"https://www.gutenberg.org/ebooks/{gid}", pause=2.0).decode("utf-8", "ignore"))
        date = parse_release_date(meta["release"])
        if meta["language"] != "English" or date < MIN_DATE:
            continue
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        body = strip_gutenberg(fetch_bytes(url, pause=2.0).decode("utf-8", "ignore"))
        if len(body) < 40000:
            continue
        sl = gutenberg_slice(body, skip_chars=15000, max_bytes=per_book)
        blocks.append((sl, {"url": url, "date": date, "title": meta["title"], "author": meta["author"],
                            "gutenberg_id": gid}))
    text, used = take_until(blocks, target)
    notes = ("FALLBACK for news2026: en.wikinews.org has no new mainspace article since 2026-05 (logevents). "
             f"English Gutenberg books released >= {MIN_DATE}, header/footer stripped, ~{per_book} B slice per book "
             "starting at max(15000 chars, 1/8 of the book) to skip front matter. CAVEAT: release date is the transcription date; the underlying "
             "texts are old (public domain) and may exist in pre-training data via other scans.")
    return text, used, notes


# ----------------------------------------------------------------------------- enwik8 control
def build_enwik8_ctrl(cache: Path, lo: int = 1_000_000, hi: int = 1_100_000) -> tuple[str, list[dict], str]:
    zpath = cache / "enwik8.zip"
    if not zpath.exists():
        zpath.write_bytes(fetch_bytes(ENWIK8_URL))
    with zipfile.ZipFile(zpath) as z, z.open("enwik8") as f:
        raw = f.read(hi)[lo:hi]
    text = raw.decode("utf-8", errors="ignore")  # slice may cut a multi-byte char at the edges
    docs = [{"url": ENWIK8_URL, "date": "2006 (Wikipedia dump, March 2006)", "byte_range": [lo, hi],
             "dropped_edge_bytes": len(raw) - len(text.encode("utf-8"))}]
    return text, docs, "enwik8 bytes [1,000,000, 1,100,000); raw XML/wiki markup kept; invalid edge bytes dropped."


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-bytes", type=int, default=100_000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    builders = {
        "wiki2026": lambda: build_wiki2026(args.target_bytes),
        "arxiv2026": lambda: build_arxiv2026(args.target_bytes),
        "gutenberg2026": lambda: build_gutenberg2026(args.target_bytes),
        "enwik8_ctrl": lambda: build_enwik8_ctrl(out),
    }
    for name, fn in builders.items():
        if args.only and name not in args.only:
            continue
        text, docs, notes = fn()
        print(json.dumps(write_corpus(out, name, text, docs, notes)), flush=True)


if __name__ == "__main__":
    main()
