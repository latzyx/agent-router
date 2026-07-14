import json
from collections import Counter
from pathlib import Path


DATASET = Path("datasets/evaluation/realistic_queries.jsonl")
CHALLENGE_DATASET = Path("datasets/evaluation/upload_challenge/challenge.jsonl")
AUGMENTATION_DATASET = Path("datasets/augmentation/v12_business_paraphrases.jsonl")
HARD_EXAMPLES_DATASET = Path("datasets/augmentation/v13_hard_examples.jsonl")
CHALLENGE_V2_DATASET = Path("datasets/evaluation/upload_challenge_v2/challenge.jsonl")
V14_AUGMENTATION_DATASET = Path("datasets/augmentation/v14_boundary_examples.jsonl")
CHALLENGE_V3_DATASET = Path("datasets/evaluation/upload_challenge_v3/challenge.jsonl")
V15_AUGMENTATION_DATASET = Path("datasets/augmentation/v15_confusion_boundaries.jsonl")
CHALLENGE_V4_DATASET = Path("datasets/evaluation/upload_challenge_v4/challenge.jsonl")
V20_AUGMENTATION_DATASET = Path("datasets/augmentation/v20_confusion_boundaries.jsonl")
CHALLENGE_V5_DATASET = Path("datasets/evaluation/upload_challenge_v5/challenge.jsonl")
V21_AUGMENTATION_DATASET = Path("datasets/augmentation/v21_agent_boundaries.jsonl")


def test_realistic_evaluation_dataset_is_balanced_and_complete():
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]

    assert len(rows) == 24
    assert Counter(row["intent"] for row in rows) == {
        "workflow.query": 6,
        "workflow.start": 6,
        "workflow.approve": 6,
        "knowledge.search": 6,
    }
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["text"] and row["agent"] and row["tool"] for row in rows)
    assert all(isinstance(row["entities"], dict) for row in rows)


def test_upload_challenge_dataset_is_balanced_and_not_in_training_source():
    challenge = [json.loads(line) for line in CHALLENGE_DATASET.read_text(encoding="utf-8").splitlines() if line]
    training = {
        json.loads(line)["text"]
        for line in Path("datasets/processed/uploaded_intents.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    }

    assert len(challenge) == 39
    assert set(Counter(row["intent"] for row in challenge).values()) == {3}
    assert {row["text"] for row in challenge}.isdisjoint(training)


def test_v12_augmentation_is_balanced_and_does_not_copy_challenge_answers():
    challenge_texts = {
        json.loads(line)["text"]
        for line in CHALLENGE_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    }
    augmentation = [
        json.loads(line)
        for line in AUGMENTATION_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert len(augmentation) == 39
    assert set(Counter(row["intent"] for row in augmentation).values()) == {3}
    assert {row["text"] for row in augmentation}.isdisjoint(challenge_texts)


def test_v13_hard_examples_do_not_copy_frozen_challenge():
    challenge_texts = {
        json.loads(line)["text"]
        for line in CHALLENGE_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    }
    hard_examples = [
        json.loads(line)
        for line in HARD_EXAMPLES_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert len(hard_examples) == 20
    assert {row["text"] for row in hard_examples}.isdisjoint(challenge_texts)


def test_v14_blind_challenge_is_frozen_and_disjoint_from_training_data():
    challenge_v2 = [
        json.loads(line)
        for line in CHALLENGE_V2_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]
    v14_augmentation = [
        json.loads(line)
        for line in V14_AUGMENTATION_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]
    existing_training = {
        json.loads(line)["text"]
        for line in Path("datasets/processed/uploaded_intents_v13.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    }

    assert len(challenge_v2) == 39
    assert set(Counter(row["intent"] for row in challenge_v2).values()) == {3}
    assert len(v14_augmentation) == 16
    assert {row["text"] for row in challenge_v2}.isdisjoint(existing_training)
    assert {row["text"] for row in challenge_v2}.isdisjoint({row["text"] for row in v14_augmentation})


def test_v15_blind_challenge_is_balanced_and_kept_out_of_training():
    challenge_v3 = [
        json.loads(line)
        for line in CHALLENGE_V3_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]
    v15_augmentation = [
        json.loads(line)
        for line in V15_AUGMENTATION_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]
    existing_training = {
        json.loads(line)["text"]
        for line in Path("datasets/processed/uploaded_intents_v14.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    }

    challenge_texts = {row["text"] for row in challenge_v3}
    assert len(challenge_v3) == 39
    assert set(Counter(row["intent"] for row in challenge_v3).values()) == {3}
    assert len(v15_augmentation) == 24
    assert challenge_texts.isdisjoint(existing_training)
    assert challenge_texts.isdisjoint({row["text"] for row in v15_augmentation})


def test_v20_challenge_is_balanced_and_separate_from_augmentation():
    challenge = [
        json.loads(line)
        for line in CHALLENGE_V4_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]
    augmentation = [
        json.loads(line)
        for line in V20_AUGMENTATION_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]
    previous_challenges = {
        json.loads(line)["text"]
        for path in (CHALLENGE_DATASET, CHALLENGE_V2_DATASET, CHALLENGE_V3_DATASET)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }
    training = {
        json.loads(line)["text"]
        for line in Path("datasets/processed/uploaded_intents_v15.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    }

    challenge_texts = {row["text"] for row in challenge}
    assert len(challenge) == 39
    assert set(Counter(row["intent"] for row in challenge).values()) == {3}
    assert len(augmentation) == 42
    assert challenge_texts.isdisjoint(training | previous_challenges)
    assert challenge_texts.isdisjoint({row["text"] for row in augmentation})


def test_v21_challenge_is_balanced_and_separate_from_all_training_data():
    challenge = [
        json.loads(line)
        for line in CHALLENGE_V5_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]
    augmentation = [
        json.loads(line)
        for line in V21_AUGMENTATION_DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]
    existing_texts = {
        json.loads(line)["text"]
        for path in (
            Path("datasets/processed/uploaded_intents_v20.jsonl"),
            CHALLENGE_DATASET,
            CHALLENGE_V2_DATASET,
            CHALLENGE_V3_DATASET,
            CHALLENGE_V4_DATASET,
        )
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }

    challenge_texts = {row["text"] for row in challenge}
    assert len(challenge) == 39
    assert set(Counter(row["intent"] for row in challenge).values()) == {3}
    assert len(augmentation) == 36
    assert challenge_texts.isdisjoint(existing_texts)
    assert challenge_texts.isdisjoint({row["text"] for row in augmentation})
