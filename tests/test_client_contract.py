import pytest

from jevcal import jev_client as jc
from jevcal.contract import normalise, validate

CRITERIA = {"World": "news", "Sports": None, "Business": "money", "Sci/Tech": "tech"}

FAKE = {
    "id": "gen-dec-test",
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": {"label": {"type": "choice", "choice": "Business", "confidence": 0.6,
                          "probabilities": {"World": 0.1, "Business": 0.7, "Sci/Tech": 0.2}}},
    "usage": {"input_tokens": 100, "output_tokens": 10, "cost": 0.00001},
}


def test_parse_fills_missing_classes_with_zero():
    a = jc.parse_choice(FAKE, "label", list(CRITERIA), "h")
    assert a.probabilities == [0.1, 0.0, 0.7, 0.2]
    assert a.model_snapshot == "typesafe/jev-1.13-20260917"


def test_parse_rejects_unknown_labels():
    bad = {**FAKE, "answers": {"label": {"choice": "X", "probabilities": {"X": 1.0}}}}
    with pytest.raises(jc.JevError):
        jc.parse_choice(bad, "label", list(CRITERIA), "h")


def test_cache_prevents_second_call(tmp_path, monkeypatch):
    calls = []
    client = jc.JevClient(api_key="test", cache_path=tmp_path / "c.sqlite")
    monkeypatch.setattr(client, "_post", lambda payload: (calls.append(1) or FAKE, 42.0))
    a1 = client.classify("some text", CRITERIA, "topic?")
    a2 = client.classify("some text", CRITERIA, "topic?")
    assert len(calls) == 1 and not a1.from_cache and a2.from_cache
    assert a2.probabilities == a1.probabilities


def test_hash_changes_with_label_wording():
    p1 = jc.build_choice_payload("t", {"A": "x"}, "q")
    p2 = jc.build_choice_payload("t", {"A": "y"}, "q")
    assert jc.request_hash(p1) != jc.request_hash(p2)


def test_validate_catches_problems():
    order = ["A", "B"]
    probs, _ = normalise([0.3, 0.6])
    good = {"example_id": "d:test:1", "dataset": "d", "split": "test", "model": "jev",
            "true_label": "A", "predicted_label": "B", "class_order": order, "class_probabilities": probs}
    assert validate([good], order, {"d:test:1"}) == []
    bad = {**good, "class_probabilities": [0.3, 0.6]}
    assert validate([bad, good], order)  # unnormalised + duplicate id
    assert validate([good], order, {"d:test:1", "d:test:2"})  # missing id


def test_stop_error_skips_rest_but_keeps_cached(tmp_path, monkeypatch):
    client = jc.JevClient(api_key="test", cache_path=tmp_path / "c.sqlite")
    monkeypatch.setattr(client, "_post", lambda payload: (FAKE, 1.0))
    client.classify("cached one", CRITERIA, "topic?")
    calls = []

    def refuse(payload):
        calls.append(1)
        raise jc.JevStopError('HTTP 402: {"error":"Insufficient credits"}')
    monkeypatch.setattr(client, "_post", refuse)
    out = client.classify_many(["cached one"] + [f"new {i}" for i in range(20)], CRITERIA, "topic?", concurrency=1)
    assert isinstance(out[0], jc.JevAnswer) and out[0].from_cache
    assert all(isinstance(o, jc.JevStopError) for o in out[1:])
    assert len(calls) == 1  # stopped after the first refusal


def test_repeat_tag_uses_separate_cache_entry(tmp_path, monkeypatch):
    calls = []
    client = jc.JevClient(api_key="test", cache_path=tmp_path / "c.sqlite")
    monkeypatch.setattr(client, "_post", lambda payload: (calls.append(payload) or FAKE, 1.0))
    client.classify("t", CRITERIA, "q")
    client.classify("t", CRITERIA, "q", repeat_tag="r1")
    client.classify("t", CRITERIA, "q", repeat_tag="r1")
    assert len(calls) == 2 and calls[0] == calls[1]  # identical payload, one extra real call
