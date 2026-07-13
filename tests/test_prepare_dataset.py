import json

from lazy_agent_router.training.prepare_dataset import load_uploaded_rows
from lazy_agent_router.training.evaluate_uploaded import load_uploaded_cases


def test_uploaded_dataset_is_deduplicated_and_has_thirteen_intents():
    rows = load_uploaded_rows("datasets/uploads")

    assert len(rows) == 390
    assert len({row["text"] for row in rows}) == 390
    assert len({row["intent"] for row in rows}) == 13


def test_uploaded_regression_loader_keeps_raw_file_counts():
    rows, files = load_uploaded_cases("datasets/uploads")

    assert len(rows) == 29250
    assert len(files) == 3
    assert {item["rows"] for item in files} == {9750}
    assert len({item["sha256"] for item in files}) == 1
