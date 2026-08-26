"""Shared mutation-frequency group definitions for TCGA-KIRC."""

from __future__ import annotations

TOTAL_KIRC_TUMOR_SAMPLES = 368
VERY_LOW_MUTATION_MAX_COUNT = 1
LOW_FREQUENCY_MUTATION_MIN_COUNT = 2
LOW_FREQUENCY_MUTATION_MAX_COUNT = 18
HIGH_FREQUENCY_MUTATION_MIN_COUNT = 19

MUTATION_GROUPS = ("very_low", "low_frequency", "high_frequency")


def validate_mutation_patient_count(patient_count: int) -> int:
    count = int(patient_count)
    if count < 0:
        raise ValueError(f"MutationPatientCount must be non-negative, got {patient_count!r}")
    return count


def classify_mutation_frequency(patient_count: int) -> str:
    count = validate_mutation_patient_count(patient_count)
    if count < LOW_FREQUENCY_MUTATION_MIN_COUNT:
        return "very_low"
    if count <= LOW_FREQUENCY_MUTATION_MAX_COUNT:
        return "low_frequency"
    return "high_frequency"


def mutation_frequency(patient_count: int, total_samples: int = TOTAL_KIRC_TUMOR_SAMPLES) -> float:
    count = validate_mutation_patient_count(patient_count)
    total = int(total_samples)
    if total <= 0:
        raise ValueError(f"total_samples must be positive, got {total_samples!r}")
    return count / total


def mutation_frequency_pct(patient_count: int, total_samples: int = TOTAL_KIRC_TUMOR_SAMPLES) -> float:
    return mutation_frequency(patient_count, total_samples) * 100.0
