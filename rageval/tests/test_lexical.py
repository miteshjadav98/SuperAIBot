"""Lexical answer metrics — hand-checked fixtures, no keys, fully deterministic.

Each expected value below is computed by hand from the formula in ``lexical.py`` so the
test pins the metric's definition, not just its current output.
"""

from __future__ import annotations

import math

import pytest

from rageval.metrics.lexical import cosine_similarity, rouge_l, token_f1


class TestTokenF1:
    def test_identical_is_one(self) -> None:
        assert token_f1("the cat sat", "the cat sat") == 1.0

    def test_subset_answer(self) -> None:
        # answer "paris" (1 tok) vs reference "it is paris" (3 tok): overlap 1.
        # precision = 1/1, recall = 1/3, F1 = 2*(1)*(1/3)/(1 + 1/3) = 0.5
        assert token_f1("Paris", "It is Paris.") == pytest.approx(0.5)

    def test_no_overlap_is_zero(self) -> None:
        assert token_f1("dog", "cat") == 0.0

    def test_bag_intersection_counts_multiplicity(self) -> None:
        # answer has "the" twice, reference once: overlap counts one, not two.
        # a=[the,the], r=[the]: overlap=1, p=1/2, r=1/1, F1=2*(0.5)*(1)/(1.5)=2/3
        assert token_f1("the the", "the") == pytest.approx(2 / 3)

    def test_both_empty_is_one(self) -> None:
        assert token_f1("", "") == 1.0

    def test_one_empty_is_zero(self) -> None:
        assert token_f1("something", "") == 0.0

    def test_punctuation_insensitive(self) -> None:
        assert token_f1("Paris!", "paris") == 1.0


class TestRougeL:
    def test_identical_is_one(self) -> None:
        assert rouge_l("the cat sat", "the cat sat") == 1.0

    def test_subsequence_gap(self) -> None:
        # a=[a,b,c], r=[a,c]: LCS "a c" = 2. p=2/3, r=2/2=1, F=2*(2/3)*1/(5/3)=0.8
        assert rouge_l("a b c", "a c") == pytest.approx(0.8)

    def test_order_sensitive_unlike_token_f1(self) -> None:
        # token_f1 would score 1.0 here (same bag); LCS respects order → only 1 match.
        # a=[dog,bites,man], r=[man,bites,dog]: max ordered LCS = 1 ("bites").
        # p=1/3, r=1/3, F = 2*(1/9)/(2/3) = 1/3
        assert token_f1("dog bites man", "man bites dog") == 1.0
        assert rouge_l("dog bites man", "man bites dog") == pytest.approx(1 / 3)

    def test_no_overlap_is_zero(self) -> None:
        assert rouge_l("dog", "cat") == 0.0

    def test_both_empty_is_one(self) -> None:
        assert rouge_l("", "") == 1.0


class TestCosineSimilarity:
    def test_identical_direction_is_one(self) -> None:
        assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_is_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_is_negative_one(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_forty_five_degrees(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1 / math.sqrt(2))

    def test_zero_vector_is_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity([], [])
