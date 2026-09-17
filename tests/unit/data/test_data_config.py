from epsa_rag.data.config import HardTestPreparationConfig, PreparationConfig


def test_legacy_development_configuration_fingerprint_is_stable() -> None:
    assert (
        PreparationConfig().fingerprint()
        == "8e3004e82bfc154263f19a7b2fbcb75b58589e70d81fb1e240de7bdbba4413ec"
    )


def test_hard_test_configuration_records_research_critical_choices() -> None:
    config = HardTestPreparationConfig()

    assert config.benchmark_role == "test"
    assert config.source_split == "train"
    assert config.difficulty_filter == "hard"
    assert config.question_count == 10_000
    assert config.selection_seed == 42
    assert config.dataset_version == "hotpotqa_hard_10000_test_v1"
    assert config.corpus_version == "hotpotqa_hard_10000_test_corpus_v1"
