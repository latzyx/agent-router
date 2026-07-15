import torch
import numpy as np
import pytest
from types import SimpleNamespace

from lazy_agent_router.training.trainer import (
    classification_metrics,
    deduplicate_rows,
    reward_penalty_loss,
    stratified_split,
    TrainingProgressCallback,
)


def test_confident_error_is_penalized_more_than_correct_prediction():
    labels = torch.tensor([0])
    correct_logits = torch.tensor([[5.0, 0.0]])
    wrong_logits = torch.tensor([[0.0, 5.0]])

    correct_loss = reward_penalty_loss(correct_logits, labels)
    wrong_loss = reward_penalty_loss(wrong_logits, labels)

    assert wrong_loss > correct_loss


def test_reward_penalty_loss_validates_strengths():
    logits = torch.tensor([[1.0, 0.0]])
    labels = torch.tensor([0])

    for kwargs in (
        {"reward_strength": 1.0},
        {"penalty_strength": -0.1},
        {"cross_group_penalty_strength": -0.1},
    ):
        try:
            reward_penalty_loss(logits, labels, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reward/penalty strength must fail")


def test_cross_group_error_is_penalized_more_than_same_group_error():
    labels = torch.tensor([0])
    group_ids = torch.tensor([0, 0, 1])
    same_group_logits = torch.tensor([[0.0, 5.0, 0.0]])
    cross_group_logits = torch.tensor([[0.0, 0.0, 5.0]])

    same_group_loss = reward_penalty_loss(
        same_group_logits,
        labels,
        cross_group_penalty_strength=1.5,
        label_group_ids=group_ids,
    )
    cross_group_loss = reward_penalty_loss(
        cross_group_logits,
        labels,
        cross_group_penalty_strength=1.5,
        label_group_ids=group_ids,
    )

    assert cross_group_loss > same_group_loss


def test_stratified_split_keeps_each_intent_in_train_and_validation():
    rows = [{"text": f"q-{label}-{index}", "intent": label} for label in ("query", "start") for index in range(5)]
    train_rows, validation_rows = stratified_split(rows, validation_split=0.2, seed=42)

    assert {row["intent"] for row in train_rows} == {"query", "start"}
    assert {row["intent"] for row in validation_rows} == {"query", "start"}
    assert {row["text"] for row in train_rows}.isdisjoint({row["text"] for row in validation_rows})


def test_training_rows_are_normalized_and_deduplicated():
    rows = [
        {"text": " 查询流程 ", "intent": " query "},
        {"text": "查询流程", "intent": "query"},
    ]
    assert deduplicate_rows(rows) == [{"text": "查询流程", "intent": "query"}]


def test_conflicting_labels_are_rejected():
    rows = [
        {"text": "查询流程", "intent": "query"},
        {"text": "查询流程", "intent": "approve"},
    ]
    with pytest.raises(ValueError, match="冲突标签"):
        deduplicate_rows(rows)


def test_classification_metrics_include_macro_f1():
    logits = np.array([[4.0, 0.0], [0.0, 4.0], [3.0, 1.0]])
    labels = np.array([0, 1, 1])
    metrics = classification_metrics((logits, labels))
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert 0 < metrics["macro_f1"] < 1


def test_training_progress_callback_reports_real_steps_and_caps_at_99():
    updates = []
    callback = TrainingProgressCallback(updates.append)
    state = SimpleNamespace(global_step=8, max_steps=10, epoch=1.25)

    callback.on_step_end(None, state, None)
    assert updates[-1] == {
        "percent": 80,
        "step": 8,
        "total_steps": 10,
        "epoch": 1.25,
    }

    state.global_step = 10
    callback.on_step_end(None, state, None)
    assert updates[-1]["percent"] == 99
