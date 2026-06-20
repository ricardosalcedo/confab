"""Tests for confab.engine."""

from confab.engine import Claim, _content_words, _normalize, annotate_response, extract_claims, score_claims


class TestExtractClaims:
    def test_prose(self):
        text = "Python was created in 1991. It supports multiple paradigms. It is dynamically typed."
        claims = extract_claims(text)
        assert len(claims) == 3
        assert "Python was created in 1991" in claims[0]

    def test_bullets(self):
        text = "- First point here.\n- Second point here.\n- Third point here."
        claims = extract_claims(text)
        assert len(claims) == 3

    def test_numbered_list(self):
        text = "1. First item here.\n2. Second item here."
        claims = extract_claims(text)
        assert len(claims) == 2

    def test_short_text_ignored(self):
        text = "Short.\nAnother short one that is long enough to pass."
        claims = extract_claims(text)
        assert all(len(c) > 10 for c in claims)

    def test_empty_input(self):
        assert extract_claims("") == []
        assert extract_claims("   \n  \n  ") == []

    def test_colon_line(self):
        text = "Language: Python"
        claims = extract_claims(text)
        assert len(claims) == 1
        assert "Python" in claims[0]


class TestScoreClaims:
    def test_high_confidence_when_all_agree(self):
        primary = "Python was created by Guido van Rossum in 1991."
        responses = [primary] * 5
        claims = score_claims(primary, responses)
        assert len(claims) == 1
        assert claims[0].confidence == 1.0
        assert claims[0].level == "high"

    def test_low_confidence_when_none_agree(self):
        primary = "Xylophone quantum blockchain synergy."
        responses = [primary, "Totally different response.", "Nothing in common here.", "Another unrelated text."]
        claims = score_claims(primary, responses)
        assert len(claims) >= 1
        assert claims[0].confidence < 0.5

    def test_mixed_confidence(self):
        primary = "Python was created in 1991. It was invented by aliens."
        responses = [
            primary,
            "Python was created in 1991 by Guido van Rossum.",
            "Python first appeared in 1991.",
            "Python released in 1991.",
        ]
        claims = score_claims(primary, responses)
        assert len(claims) == 2
        # First claim (1991) should be high, second (aliens) should be low
        assert claims[0].confidence > claims[1].confidence

    def test_support_count_tracking(self):
        primary = "Python was created in 1991."
        responses = [primary, "Python appeared in 1991.", "Something else entirely."]
        claims = score_claims(primary, responses)
        assert claims[0].support_count >= 1
        assert claims[0].total_samples == 3


class TestClaim:
    def test_level_high(self):
        c = Claim(text="test", confidence=0.9, support_count=4, total_samples=5)
        assert c.level == "high"

    def test_level_medium(self):
        c = Claim(text="test", confidence=0.6, support_count=3, total_samples=5)
        assert c.level == "medium"

    def test_level_low(self):
        c = Claim(text="test", confidence=0.3, support_count=1, total_samples=5)
        assert c.level == "low"

    def test_boundary_high(self):
        c = Claim(text="test", confidence=0.8, support_count=4, total_samples=5)
        assert c.level == "high"

    def test_boundary_medium(self):
        c = Claim(text="test", confidence=0.5, support_count=2, total_samples=5)
        assert c.level == "medium"


class TestAnnotateResponse:
    def test_formatting(self):
        claims = [
            Claim(text="High claim", confidence=0.9, support_count=4, total_samples=5),
            Claim(text="Low claim", confidence=0.2, support_count=1, total_samples=5),
        ]
        result = annotate_response(claims)
        assert "high 90%" in result
        assert "low 20%" in result
        assert "High claim" in result


class TestHelpers:
    def test_normalize(self):
        assert _normalize("Hello, World!") == "hello world"
        assert _normalize("Test 123") == "test 123"

    def test_content_words_filters_stopwords(self):
        words = _content_words("the cat is on the mat")
        assert "the" not in words
        assert "cat" in words
        assert "mat" in words
