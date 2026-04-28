"""Unit tests for strict Shor post-processing primitives."""

from shor_benchmark.strict_postprocess import (
    continued_fraction_denominator,
    strict_predict_order,
)


def test_continued_fraction_denominator_returns_expected_value() -> None:
    """CF denominator should match known exact rational cases."""

    denominator: int = continued_fraction_denominator(y=4, q=16, n=15)
    assert denominator == 4


def test_strict_predict_order_returns_order_when_validated() -> None:
    """Strict predictor should return the minimum validated candidate order."""

    predicted: int | None = strict_predict_order(a=2, n=15, denominators=[2, 4])
    assert predicted == 4


def test_strict_predict_order_returns_none_when_no_candidate_validates() -> None:
    """Strict predictor should return None when validation set is empty."""

    predicted: int | None = strict_predict_order(a=2, n=15, denominators=[3])
    assert predicted is None
