"""AG-Fresh-2026: a small, double-annotated news test set published after Jev's snapshot.

Pipeline (run on your own machine, which can reach the feeds):
  1. scripts/collect_headlines.py  - run once a day; appends new items to data/fresh/candidates.csv
  2. scripts/make_annotation_sheets.py - samples items, writes one blind sheet per annotator
  3. both annotators label independently (docs/annotation_guide.md)
  4. scripts/agreement.py           - Cohen's kappa + adjudication sheet, then the final test file
"""

from __future__ import annotations

import csv
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

FRESH = Path("data/fresh")
CANDIDATES = FRESH / "candidates.csv"
FINAL = FRESH / "ag_fresh_2026.csv"
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
EXTRA_CODES = ["none"]  # fits no AG News class -> excluded from the test set

# Several outlets per category, so no single outlet's style dominates.
# The feed's section is stored only to balance sampling; annotators never see it.
FEEDS = {
    "World": ["https://feeds.bbci.co.uk/news/world/rss.xml",
              "https://www.theguardian.com/world/rss",
              "https://feeds.npr.org/1004/rss.xml"],
    "Sports": ["https://feeds.bbci.co.uk/sport/rss.xml",
               "https://www.theguardian.com/uk/sport/rss",
               "https://feeds.npr.org/1055/rss.xml"],
    "Business": ["https://feeds.bbci.co.uk/news/business/rss.xml",
                 "https://www.theguardian.com/uk/business/rss",
                 "https://feeds.npr.org/1006/rss.xml"],
    "Sci/Tech": ["https://feeds.bbci.co.uk/news/technology/rss.xml",
                 "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
                 "https://www.theguardian.com/uk/technology/rss",
                 "https://www.theguardian.com/science/rss",
                 "https://feeds.npr.org/1019/rss.xml"],
}

FIELDS = ["item_id", "title", "description", "text", "url", "published_utc", "outlet", "feed_section", "collected_utc"]

_TAG = re.compile(r"<[^>]+>")


def clean(s: str | None) -> str:
    s = html.unescape(_TAG.sub(" ", s or ""))
    return " ".join(s.split())


def _date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        d = parsedate_to_datetime(s)  # RSS
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))  # Atom
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def parse_feed(xml_text: str) -> list[dict]:
    """RSS 2.0 or Atom -> [{title, description, url, published}]"""
    root = ET.fromstring(xml_text)
    out = []
    for it in root.iter("item"):
        out.append({"title": clean(it.findtext("title")), "description": clean(it.findtext("description")),
                    "url": (it.findtext("link") or "").strip(), "published": _date(it.findtext("pubDate"))})
    atom = "{http://www.w3.org/2005/Atom}"
    for e in root.iter(f"{atom}entry"):
        link = e.find(f"{atom}link")
        out.append({"title": clean(e.findtext(f"{atom}title")),
                    "description": clean(e.findtext(f"{atom}summary") or e.findtext(f"{atom}content")),
                    "url": link.get("href", "") if link is not None else "",
                    "published": _date(e.findtext(f"{atom}published") or e.findtext(f"{atom}updated"))})
    return out


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
