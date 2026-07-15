"""PHI pseudonymisation.

Privacy model
-------------
In clinical RAG the *query / patient context* carries PHI, not the public
knowledge base. So we pseudonymise the incoming query **before** it is embedded,
searched, put into a prompt, or sent to any external LLM. Nothing with raw PHI
crosses the process/network boundary.

Pseudonymisation vs redaction: we do not blank PHI out. Each entity is mapped to
a **consistent, reversible surrogate** token (``[PERSON_1]``, ``[DATE_1]`` ...).
Consistent → the same value always gets the same token, so the model can still
reason about "the same patient". Reversible → we keep a local-only mapping and
re-identify the final answer for the clinician. The mapping never leaves the box.

HIPAA nuance: under Safe Harbor only ages **over 89** are PHI, so ordinary ages
(62, 74) are intentionally left intact — that is correct behaviour, not a miss.

Backend: Microsoft Presidio (spaCy NER + recognizers) is primary, with a custom
MRN recognizer. A dependency-free regex backend is used only if Presidio/spaCy
are unavailable, so the pipeline still runs offline; the active backend is logged.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .config import PrivacyCfg

log = logging.getLogger(__name__)

# MRN like "MRN 4839201" / "MRN: 55129" / "medical record number 77120".
_MRN_RE = re.compile(r"\b(?:MRN|medical record (?:no|number))\s*[:#]?\s*(\d{4,10})\b", re.I)

# Regex-backend patterns (fallback only).
_REGEX_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("EMAIL_ADDRESS", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE_NUMBER", re.compile(r"\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    ("DATE_TIME", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    # Titled names: Mr./Mrs./Ms./Dr. Firstname [Lastname]
    ("PERSON", re.compile(r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")),
]


@dataclass
class Entity:
    entity_type: str
    start: int
    end: int
    text: str


@dataclass
class DeidResult:
    original: str
    text: str  # pseudonymised text
    mapping: Dict[str, str] = field(default_factory=dict)  # token -> original value
    entities: List[Entity] = field(default_factory=list)

    def re_identify(self, text: str) -> str:
        """Replace surrogate tokens back with the original PHI (for the clinician)."""
        out = text
        # Longest tokens first to avoid partial collisions ([PERSON_1] vs [PERSON_10]).
        for token in sorted(self.mapping, key=len, reverse=True):
            out = out.replace(token, self.mapping[token])
        return out


class Pseudonymizer:
    def __init__(self, cfg: PrivacyCfg):
        self.cfg = cfg
        self._analyzer = None
        self._backend = None  # "presidio" | "regex"

    # ---- backend setup -----------------------------------------------------
    def _ensure_backend(self) -> str:
        if self._backend is not None:
            return self._backend
        try:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

            analyzer = AnalyzerEngine()
            if self.cfg.enable_mrn_recognizer:
                mrn = PatternRecognizer(
                    supported_entity="MRN",
                    patterns=[Pattern(name="mrn", regex=_MRN_RE.pattern, score=0.85)],
                )
                analyzer.registry.add_recognizer(mrn)
            self._analyzer = analyzer
            self._backend = "presidio"
            log.info("Pseudonymizer backend: Presidio")
        except Exception as exc:  # pragma: no cover - exercised only w/o presidio
            log.warning("Presidio unavailable (%s); using regex de-id fallback.", exc)
            self._backend = "regex"
        return self._backend

    @property
    def backend(self) -> str:
        return self._ensure_backend()

    # ---- detection ---------------------------------------------------------
    def _detect(self, text: str) -> List[Entity]:
        if self._ensure_backend() == "presidio":
            entities = list(self.cfg.entities)
            if self.cfg.enable_mrn_recognizer:
                entities = entities + ["MRN"]
            results = self._analyzer.analyze(
                text=text,
                entities=entities,
                language="en",
                score_threshold=self.cfg.score_threshold,
            )
            found = [
                Entity(r.entity_type, r.start, r.end, text[r.start : r.end])
                for r in results
            ]
        else:
            found = self._detect_regex(text)
        return _resolve_overlaps(found)

    def _detect_regex(self, text: str) -> List[Entity]:
        found: List[Entity] = []
        for m in _MRN_RE.finditer(text):
            found.append(Entity("MRN", m.start(1), m.end(1), m.group(1)))
        for etype, pat in _REGEX_PATTERNS:
            if etype not in self.cfg.entities and etype != "MRN":
                continue
            for m in pat.finditer(text):
                found.append(Entity(etype, m.start(), m.end(), m.group()))
        return found

    # ---- pseudonymisation --------------------------------------------------
    def pseudonymize(self, text: str) -> DeidResult:
        entities = self._detect(text)
        value_to_token: Dict[str, str] = {}
        token_to_value: Dict[str, str] = {}
        counters: Dict[str, int] = {}

        # Assign consistent tokens (stable order: first appearance in text).
        for ent in sorted(entities, key=lambda e: e.start):
            key = f"{ent.entity_type}:{ent.text}"
            if key not in value_to_token:
                counters[ent.entity_type] = counters.get(ent.entity_type, 0) + 1
                token = f"[{ent.entity_type}_{counters[ent.entity_type]}]"
                value_to_token[key] = token
                token_to_value[token] = ent.text

        # Rebuild text replacing spans right-to-left (keeps offsets valid).
        out = text
        for ent in sorted(entities, key=lambda e: e.start, reverse=True):
            token = value_to_token[f"{ent.entity_type}:{ent.text}"]
            out = out[: ent.start] + token + out[ent.end :]

        return DeidResult(
            original=text, text=out, mapping=token_to_value, entities=entities
        )

    # ---- leakage check (used by the privacy metric) ------------------------
    def find_leaks(self, text: str, result: DeidResult) -> List[str]:
        """Return any original PHI values that still appear verbatim in ``text``."""
        return [v for v in result.mapping.values() if v and v in text]


def _resolve_overlaps(entities: List[Entity]) -> List[Entity]:
    """Drop overlapping spans, preferring longer matches (more specific)."""
    ordered = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
    kept: List[Entity] = []
    last_end = -1
    for ent in ordered:
        if ent.start >= last_end:
            kept.append(ent)
            last_end = ent.end
    return kept
