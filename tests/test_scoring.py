import numpy as np

from jevcal import scoring as S
from jevcal.contract import write_jsonl

ORDER = ["A", "B", "C"]


def make_rows(model, n, acc, conf, seed, budget=None, rseed=None, logits=False, prefix="d:test:"):
    rng = np.random.default_rng(rseed or 0)
    rows = []
    for i in range(n):
        y = i % 3
        pred = y if rng.random() < acc else (y + 1) % 3
        p = np.full(3, (1 - conf) / 2)
        p[pred] = conf
        r = {"example_id": f"{prefix}{i}", "dataset": "d", "split": prefix.split(":")[1], "model": model,
             "label_config": "cfg" if model == "jev" else None, "budget": budget, "seed": seed,
             "true_label": ORDER[y], "predicted_label": ORDER[pred], "class_order": ORDER,
             "class_probabilities": p.tolist()}
        if logits:
            r["logits"] = np.log(p).tolist()
        rows.append(r)
    return rows


def test_temperature_scaling_fixes_overconfidence():
    ev = make_rows("enc", 600, acc=0.7, conf=0.98, seed=1, budget=8, rseed=1, logits=True)
    cal = make_rows("enc", 600, acc=0.7, conf=0.98, seed=1, budget=8, rseed=2, logits=True, prefix="d:train:")
    res = S.score_run(ev, cal)
    assert res["temperature"] > 1
    assert res["ts_ece_15_width"] < res["raw_ece_15_width"]


def test_routing_prefers_confident_jev():
    # Jev: right when confident (0.95), wrong when not (0.5). Encoder: 70% everywhere.
    def jev(prefix, n):
        rows = []
        for i in range(n):
            y = i % 3
            good = i % 2 == 0
            pred = y if good else (y + 1) % 3
            p = np.full(3, 0.025 if good else 0.25)
            p[pred] = 0.95 if good else 0.5
            rows.append({"example_id": f"{prefix}{i}", "model": "jev", "true_label": ORDER[y],
                         "class_order": ORDER, "class_probabilities": p.tolist()})
        return rows
    enc_ev = make_rows("enc", 400, 0.7, 0.8, 1, 8, rseed=3)
    enc_cal = make_rows("enc", 400, 0.7, 0.8, 1, 8, rseed=4, prefix="d:train:")
    rt = S.routing(jev("d:test:", 400), enc_ev, jev("d:train:", 400), enc_cal)
    assert 0.5 < rt["tau"] <= 0.95
    assert rt["jev_share"] == 0.5
    assert rt["routed_accuracy"] > max(rt["jev_accuracy"], rt["encoder_accuracy"])


def test_load_runs_groups_by_run(tmp_path):
    write_jsonl(make_rows("enc", 10, 0.8, 0.9, 13, 8), tmp_path / "a.jsonl")
    write_jsonl(make_rows("enc", 10, 0.8, 0.9, 42, 8), tmp_path / "b.jsonl")
    runs = S.load_runs([str(tmp_path / "*.jsonl")])
    assert len(runs) == 2
