"""One Jev call to check the key, credits and response format.
Run from the repo root:  python scripts/smoke_test.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv  # noqa: E402

from jevcal.jev_client import JevClient, JevError  # noqa: E402

load_dotenv()
cfg = json.loads(Path("configs/labels/ag_news.json").read_text())
criteria = {l["name"]: l["description"] for l in cfg["labels"]}

client = JevClient(cache_path="cache/smoke_cache.sqlite")
try:
    a = client.classify("Stocks rallied after the central bank held interest rates steady.",
                        criteria, cfg["instructions"])
except JevError as e:
    print("FAILED:", e)
    print("Check: OPENROUTER_API_KEY in .env, credits on openrouter.ai/settings/credits.")
    sys.exit(1)

print("predicted     :", a.predicted_label)
print("probabilities :", dict(zip(a.class_order, a.probabilities)))
print("confidence    :", a.confidence)
print("model snapshot:", a.model_snapshot)
print("cost (USD)    :", a.usage.get("cost"))
print("latency (ms)  :", round(a.latency_ms or 0), "(cached)" if a.from_cache else "")
print("\nOK - Jev works. Record the model snapshot in the run ledger.")
