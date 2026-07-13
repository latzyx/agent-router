from collections import Counter

from lazy_agent_router.training.generate_corpus import build_rows


def test_generated_corpus_is_balanced_and_valid():
    rows = build_rows()
    counts = Counter(row["intent"] for row in rows)
    assert len(rows) == 192
    assert set(counts) == {"workflow.query", "workflow.start", "workflow.approve", "knowledge.search"}
    assert set(counts.values()) == {48}
    assert all(row["text"] for row in rows)
