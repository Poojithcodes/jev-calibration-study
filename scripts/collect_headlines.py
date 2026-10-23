"""Collect fresh news items for AG-Fresh-2026. Run once a day until you have ~700 candidates.

  python scripts/collect_headlines.py --since 2026-10-22

Only items published on/after --since are kept. Duplicates (same URL or same title) are skipped,
so running it repeatedly is safe.
"""
import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import requests  # noqa: E402

from jevcal import fresh  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--since", default="2026-10-22", help="YYYY-MM-DD (UTC); keep items published on/after this")
args = ap.parse_args()
since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

rows = fresh.read_csv(fresh.CANDIDATES)
seen_urls = {r["url"] for r in rows}
seen_titles = {fresh.norm_title(r["title"]) for r in rows}
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
added = Counter()
headers = {"User-Agent": "Mozilla/5.0 (research data collection; RVCE student project)"}

for section, urls in fresh.FEEDS.items():
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            items = fresh.parse_feed(r.text)
        except Exception as e:  # one dead feed must not stop the rest
            print(f"  skip {url}: {e}")
            continue
        for it in items:
            t = fresh.norm_title(it["title"])
            if (not it["title"] or not it["url"] or it["published"] is None or it["published"] < since
                    or it["url"] in seen_urls or t in seen_titles):
                continue
            seen_urls.add(it["url"])
            seen_titles.add(t)
            rows.append({
                "item_id": f"c{len(rows):05d}", "title": it["title"], "description": it["description"],
                "text": f"{it['title']} {it['description']}".strip(), "url": it["url"],
                "published_utc": it["published"].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "outlet": urlparse(url).netloc, "feed_section": section, "collected_utc": now,
            })
            added[section] += 1

fresh.write_csv(fresh.CANDIDATES, rows, fresh.FIELDS)
total = Counter(r["feed_section"] for r in rows)
print(f"added {sum(added.values())} new items {dict(added)}")
print(f"total candidates: {len(rows)} {dict(total)} -> {fresh.CANDIDATES}")
