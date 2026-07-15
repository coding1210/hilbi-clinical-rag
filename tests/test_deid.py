"""Pseudonymisation tests.

Forced onto the regex backend so they are hermetic (no Presidio/spaCy model
download). The regex backend covers MRN, dates, phones, emails, SSNs, and titled
names — enough to prove the pseudonymise → reversible-map → re-identify contract
and the leak check.
"""
from clinical_rag.config import PrivacyCfg
from clinical_rag.deid import Pseudonymizer

CFG = PrivacyCfg(
    entities=["PERSON", "DATE_TIME", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN"],
    score_threshold=0.4,
    enable_mrn_recognizer=True,
)


def _regex_pseudonymizer() -> Pseudonymizer:
    p = Pseudonymizer(CFG)
    p._backend = "regex"  # force hermetic backend
    return p


def test_mrn_and_phone_are_replaced():
    p = _regex_pseudonymizer()
    text = "Mr. John Smith MRN 4839201, phone 555-203-8890."
    res = p.pseudonymize(text)
    assert "4839201" not in res.text
    assert "555-203-8890" not in res.text
    assert "[MRN_1]" in res.text
    assert "[PHONE_NUMBER_1]" in res.text


def test_mapping_is_reversible():
    p = _regex_pseudonymizer()
    text = "Mr. John Smith MRN 4839201 seen on 02/04/2025."
    res = p.pseudonymize(text)
    assert res.re_identify(res.text) == text  # round-trips exactly


def test_consistent_tokens_for_repeated_value():
    p = _regex_pseudonymizer()
    text = "MRN 12345 today. Repeat: MRN 12345 again."
    res = p.pseudonymize(text)
    # Same MRN value must map to the same surrogate token.
    assert res.text.count("[MRN_1]") == 2
    assert "[MRN_2]" not in res.text


def test_no_leak_after_pseudonymisation():
    p = _regex_pseudonymizer()
    text = "Dr. Alan Kim, patient MRN 77120, email a@b.com, SSN 123-45-6789."
    res = p.pseudonymize(text)
    assert p.find_leaks(res.text, res) == []
