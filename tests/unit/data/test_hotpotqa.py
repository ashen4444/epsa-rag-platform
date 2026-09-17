from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request

import pytest

from epsa_rag.core.exceptions import SourceValidationError
from epsa_rag.core.ids import stable_digest
from epsa_rag.data.config import HardTestPreparationConfig, PreparationConfig
from epsa_rag.data.hotpotqa import download_source, load_selected_source, load_source


def raw_example(question_id: str = "q1") -> dict[str, object]:
    return {
        "_id": question_id,
        "question": "  Which answer?  ",
        "answer": "Answer",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": [["Title", 1]],
        "context": [["Title", ["First sentence.", " Second sentence."]]],
    }


def source_bytes(*records: dict[str, object]) -> bytes:
    return json.dumps(list(records), ensure_ascii=False).encode("utf-8")


def test_download_source_is_atomic_verified_and_idempotent(tmp_path: Path) -> None:
    content = source_bytes(raw_example())
    expected = hashlib.sha256(content).hexdigest()
    destination = tmp_path / "raw" / "source.json"
    requests: list[Request] = []

    def open_url(request: Request) -> io.BytesIO:
        requests.append(request)
        return io.BytesIO(content)

    assert download_source(
        uri="https://example.test/source.json",
        destination=destination,
        expected_sha256=expected,
        open_url=open_url,
    ) == destination
    assert destination.read_bytes() == content

    download_source(
        uri="https://example.test/source.json",
        destination=destination,
        expected_sha256=expected,
        open_url=lambda request: pytest.fail(f"unexpected request: {request.full_url}"),
    )
    assert len(requests) == 1


def test_download_source_rejects_digest_mismatch_and_removes_temporary_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "source.json"

    with pytest.raises(SourceValidationError, match="SHA-256 mismatch"):
        download_source(
            uri="https://example.test/source.json",
            destination=destination,
            expected_sha256="0" * 64,
            open_url=lambda request: io.BytesIO(b"not expected"),
        )

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_load_source_preserves_native_whitespace_and_indices(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    content = source_bytes(raw_example())
    path.write_bytes(content)

    examples = load_source(path, expected_sha256=hashlib.sha256(content).hexdigest())

    assert examples[0].question == "  Which answer?  "
    assert examples[0].context[0][1][1] == " Second sentence."
    assert examples[0].supporting_facts == (("Title", 1),)


def test_load_source_preserves_empty_native_sentence_entries(tmp_path: Path) -> None:
    record = raw_example()
    record["context"] = [["Title", ["", "Supporting sentence."]]]
    path = tmp_path / "source.json"
    path.write_bytes(source_bytes(record))

    examples = load_source(path)

    assert examples[0].context[0][1] == ("", "Supporting sentence.")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (json.dumps({"not": "a list"}), "JSON array"),
        (json.dumps([]), "at least one"),
        ("not-json", "unable to load"),
    ],
)
def test_load_source_rejects_invalid_roots(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "source.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(SourceValidationError, match=message):
        load_source(path)


def test_load_source_rejects_duplicate_question_ids(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_bytes(source_bytes(raw_example(), raw_example()))

    with pytest.raises(SourceValidationError, match="duplicate question id"):
        load_source(path)


def test_selected_source_records_invalid_unselected_examples(tmp_path: Path) -> None:
    seed = 42
    ids = ["q1", "q2", "q3"]
    ranked_ids = sorted(ids, key=lambda value: (stable_digest(str(seed), value), value))
    invalid_id = ranked_ids[-1]
    records = [raw_example(question_id) for question_id in ids]
    for record in records:
        if record["_id"] == invalid_id:
            record["supporting_facts"] = [["Title", 99]]
    content = source_bytes(*records)
    path = tmp_path / "source.json"
    path.write_bytes(content)
    config = PreparationConfig(
        question_count=2,
        selection_seed=seed,
        expected_source_sha256=hashlib.sha256(content).hexdigest(),
    )

    selected = load_selected_source(path, config=config)

    assert tuple(example.question_id for example in selected.examples) == tuple(ranked_ids[:2])
    assert selected.source_record_count == 3
    assert selected.eligible_record_count == 3
    assert selected.invalid_unselected_question_ids == (invalid_id,)


def test_selected_source_filters_difficulty_before_deterministic_ranking(
    tmp_path: Path,
) -> None:
    records = [
        raw_example("hard-1"),
        {**raw_example("medium-1"), "level": "medium"},
        raw_example("hard-2"),
        {**raw_example("easy-1"), "level": "easy"},
    ]
    content = source_bytes(*records)
    path = tmp_path / "source.json"
    path.write_bytes(content)
    config = HardTestPreparationConfig(
        question_count=2,
        expected_source_sha256=hashlib.sha256(content).hexdigest(),
    )

    selected = load_selected_source(path, config=config)

    expected_ids = sorted(
        ("hard-1", "hard-2"),
        key=lambda value: (stable_digest(str(config.selection_seed), value), value),
    )
    assert [example.question_id for example in selected.examples] == expected_ids
    assert all(example.level == "hard" for example in selected.examples)
    assert selected.source_record_count == 4
    assert selected.eligible_record_count == 2


def test_hard_test_excludes_invalid_records_before_deterministic_ranking(
    tmp_path: Path,
) -> None:
    seed = 42
    ids = ["hard-1", "hard-2", "hard-3", "hard-4"]
    ranked_ids = sorted(ids, key=lambda value: (stable_digest(str(seed), value), value))
    invalid_id = ranked_ids[0]
    records = [raw_example(question_id) for question_id in ids]
    for record in records:
        if record["_id"] == invalid_id:
            record["supporting_facts"] = [["Title", 99]]
    content = source_bytes(*records)
    path = tmp_path / "source.json"
    path.write_bytes(content)
    config = HardTestPreparationConfig(
        question_count=2,
        selection_seed=seed,
        expected_source_sha256=hashlib.sha256(content).hexdigest(),
    )

    selected = load_selected_source(path, config=config)

    valid_ranked_ids = [question_id for question_id in ranked_ids if question_id != invalid_id]
    assert [example.question_id for example in selected.examples] == valid_ranked_ids[:2]
    assert selected.source_record_count == 4
    assert selected.eligible_record_count == 3
    assert selected.invalid_unselected_question_ids == (invalid_id,)


def test_selected_source_rejects_more_questions_than_eligible_records(tmp_path: Path) -> None:
    records = [raw_example("hard-1"), {**raw_example("medium-1"), "level": "medium"}]
    content = source_bytes(*records)
    path = tmp_path / "source.json"
    path.write_bytes(content)
    config = HardTestPreparationConfig(
        question_count=2,
        expected_source_sha256=hashlib.sha256(content).hexdigest(),
    )

    with pytest.raises(SourceValidationError, match="1 eligible records"):
        load_selected_source(path, config=config)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (json.dumps({"not": "a list"}), "JSON array"),
        (json.dumps([]), "at least one"),
        ('[{"_id":"q1"}', "unable to load"),
        ('[{"_id":}]', "unable to load"),
        ('[{"_id":"q1"} {"_id":"q2"}]', "expected a comma"),
        ("[] trailing", "content follows"),
    ],
)
def test_selected_source_stream_rejects_invalid_roots(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "source.json"
    path.write_text(content, encoding="utf-8")
    config = PreparationConfig(
        question_count=1,
        expected_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    with pytest.raises(SourceValidationError, match=message):
        load_selected_source(path, config=config)


def test_selected_source_rejects_an_invalid_selected_example(tmp_path: Path) -> None:
    record = raw_example("q1")
    record["supporting_facts"] = [["Title", 99]]
    content = source_bytes(record)
    path = tmp_path / "source.json"
    path.write_bytes(content)
    config = PreparationConfig(
        question_count=1,
        expected_source_sha256=hashlib.sha256(content).hexdigest(),
    )

    with pytest.raises(SourceValidationError, match="selected HotPotQA example"):
        load_selected_source(path, config=config)


def test_load_source_rejects_missing_question_identifier(tmp_path: Path) -> None:
    record = raw_example()
    del record["_id"]
    path = tmp_path / "source.json"
    path.write_bytes(source_bytes(record))

    with pytest.raises(SourceValidationError, match="no string _id"):
        load_source(path)


@pytest.mark.parametrize(
    "mutation",
    [
        {"supporting_facts": [["Missing", 0]]},
        {"supporting_facts": [["Title", 4]]},
        {"supporting_facts": []},
        {"context": [["Title", []]]},
        {"context": [["Title", ["One."]], ["Title", ["Two."]]]},
    ],
)
def test_load_source_rejects_invalid_evidence_references(
    tmp_path: Path,
    mutation: dict[str, object],
) -> None:
    record = raw_example()
    record.update(mutation)
    path = tmp_path / "source.json"
    path.write_bytes(source_bytes(record))

    with pytest.raises(SourceValidationError, match="invalid HotPotQA example"):
        load_source(path)
