"""Explicit, persisted configuration for Phase 2 preparation."""

from typing import Literal

from pydantic import Field

from epsa_rag.core.config import ConfigModel
from epsa_rag.core.ids import Identifier

DEFAULT_SOURCE_URI = (
    "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)
DEFAULT_DOWNLOAD_URI = (
    "https://huggingface.co/datasets/RAGLAB/data/resolve/main/"
    "eval_datasets/HotPotQA/hotpot_dev_distractor_v1.json?download=true"
)
DEFAULT_SOURCE_SHA256 = "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea"


class PreparationConfig(ConfigModel):
    """Research-critical choices for the first frozen HotPotQA benchmark."""

    dataset_version: Identifier = "hotpotqa_1000_v1"
    corpus_version: Identifier = "hotpotqa_10000_v1"
    question_count: int = Field(default=1_000, ge=1)
    selection_seed: int = 42
    selection_method: Literal["sha256-rank-v1"] = "sha256-rank-v1"
    chunk_id_method: Literal["sha256-title-sentences-v1"] = "sha256-title-sentences-v1"
    deduplication_method: Literal["exact-chunk-id-v1"] = "exact-chunk-id-v1"
    source_dataset: Literal["HotPotQA"] = "HotPotQA"
    source_configuration: Literal["distractor"] = "distractor"
    source_split: Literal["dev"] = "dev"
    source_uri: str = DEFAULT_SOURCE_URI
    download_uri: str = DEFAULT_DOWNLOAD_URI
    expected_source_sha256: str = DEFAULT_SOURCE_SHA256
