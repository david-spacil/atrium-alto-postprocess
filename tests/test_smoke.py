"""
tests/test_smoke.py
===================
End-to-end smoke tests for the pipeline categorization logic with mocked model inferences.

The mocking stops at the MODELS. Everything downstream of them goes through the
production scoring step (``classify_TEXT.score_line``, reached via
``recategorize_from_csv._rescore_row``) rather than a local re-implementation of
it -- see ``_process_mocked_line`` for why that distinction is load-bearing.
"""

import sys
import types
from pathlib import Path

import pytest

# Stub the GPU/ML stack before importing the tool (it imports classify_TEXT),
# mirroring tests/test_calibration.py and tests/test_rotation_regression.py.
for _n in ("torch", "tqdm", "fasttext", "transformers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["tqdm"].tqdm = lambda x, **k: x  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from recategorize_from_csv import _load_lang_config, _rescore_row  # noqa: E402

import classify_TEXT as LC  # noqa: E402
from text_util import pre_filter_line  # noqa: E402

_EXPECTED, _KNOWN = _load_lang_config(str(_ROOT / "setup" / "config.txt"))


class TestFullPipelineSmoke:
    def _process_mocked_line(self, line_text, mock_ppl, mock_lang_score):
        """Categorise one line through the REAL production path, models aside.

        This used to hand-roll its own copy of the orchestrator, and the copy
        was wrong in three ways that all pointed the same direction -- towards
        making garbage look cleaner than production sees it:

          * it passed ``min(mock_lang_score, 0.75)`` -- the ``LANG_SCORE_REMAP``
            cap, i.e. the score RECORDED in the CSV -- where production passes
            the two-tier ``trust_lang_score``;
          * it never passed ``orig_lang_score``, leaving it at the ``1.0``
            default, which silently disables ``rule_hard_sweep`` (needs
            ``< 0.45``), ``rule_extreme_ppl`` (needs ``< 0.85``) and one clause
            of ``_has_strong_garbage_evidence``;
          * it never passed ``garbage_density``, leaving it at ``0.0``, which
            disables that predicate's density clause too.

        So three Trash routes could not fire in this module at all, and every
        assertion here was being made about a signal vector the pipeline never
        produces. This was the third hand-rolled harness in the suite with that
        class of bug; the other two were fixed in
        ``tests/test_rotation_regression.py`` and ``tests/test_calibration.py``,
        and the repo's rule is one scoring engine, not two (CONTRIBUTING.md,
        ``tests/test_scoring_single_source.py``).

        ``original_lang`` is ces (expected, trust tier 1.0), matching
        ``tests/test_calibration.py::_categ``. That is the strict choice for a
        smoke test: tier 1.0 is the LARGEST score the tiers can produce, so the
        "must be Trash" assertions are harder to satisfy, not easier.
        """
        action, clean_text = pre_filter_line(line_text)
        if action != "Process":
            return action

        row = {
            "text": clean_text,
            "original_text": line_text,
            "original_lang": "ces_Latn",
            "orig_lang_score": f"{mock_lang_score}",
            "perplex": f"{mock_ppl}",
            "categ": "Noisy",
            "word_count": str(len(clean_text.split())),
        }
        return _rescore_row(row, _EXPECTED, _KNOWN)["categ"]

    def test_clean_czech_prose_is_clear_or_noisy(self):
        prose_lines = [
            "Poučení o povinnosti ku taxe vojenské.",
            "Tento nález byl učiněn v hloubce 30 cm pod povrchem.",
            "Keramické zlomky s vlnovkou.",
        ]
        for line in prose_lines:
            cat = self._process_mocked_line(line, mock_ppl=150.0, mock_lang_score=0.95)
            assert cat in ("Clear", "Noisy"), f"Clean text '{line}' misclassified as {cat}"

    def test_garbage_and_mirror_is_trash_or_nontext(self):
        garbage_lines = [
            "TYRSOVA5===aras",
            "WVL e##xon w!wx",  # Added symbols so weirdness tanks the QS
            "pbqdnuwmoxszeyv!!",  # Added punctuation so weirdness > 0, triggering rot_penalty
            "AAMMNAbSSOAO###",  # Spurious caps + symbols
            "123 456 789",  # Pure digits -> Non-text
        ]
        for line in garbage_lines:
            cat = self._process_mocked_line(line, mock_ppl=3000.0, mock_lang_score=0.15)
            assert cat in ("Trash", "Non-text"), f"Garbage text '{line}' misclassified as {cat}"

    # ────────────────────────────────────────────────────────────────────────
    # (#3) Real-data calibration fixtures.
    #
    # IMPORTANT boundary: only garbage that the PER-LINE path can route on its
    # own belongs here. Multi-token / interspersed inverted garbage (e.g.
    # "NU -", "e.ao u", "wL-U kyuto Cona JaaVHUoaAL") scores as Noisy in
    # isolation and is only reclassified by the page-level inverted-scan sweep —
    # those cases live in tests/test_page_postprocess.py, which exercises
    # apply_document_postprocessing. Asserting Trash for them here would be
    # dishonest about where the fix actually lives.
    # ────────────────────────────────────────────────────────────────────────
    def test_real_short_garbage_is_trash_per_line(self):
        # 'olie' -> short-garbage route; '° 47' -> plain quality-score Trash.
        for line in ["olie", "° 47"]:
            cat = self._process_mocked_line(line, mock_ppl=300.0, mock_lang_score=0.40)
            assert cat == "Trash", f"Short garbage '{line}' misclassified as {cat}"

    def test_real_clean_prose_promotes_to_clear(self):
        # Diacritic-rich Czech prose dense in short function words must reach Clear
        # now that compute_valid_ratio counts those short words (#3 C).
        clear_lines = [
            "svým jménem, nýbrž i lidovým podáním,které tvrdí,že v místech těchto stávala",
            "Pátral Jsem v první řadě po stříbrných penězích .které prý",
        ]
        for line in clear_lines:
            cat = self._process_mocked_line(line, mock_ppl=40.0, mock_lang_score=0.97)
            assert cat == "Clear", f"Clean prose '{line[:30]}…' misclassified as {cat}"

    def test_clean_czech_never_demoted_to_trash(self):
        # Regression guard for the short-garbage route: genuine short Czech must
        # never be Trashed.
        for line in ["Náčrt sondy.", "Praha", "kostra hrob náramek"]:
            cat = self._process_mocked_line(line, mock_ppl=200.0, mock_lang_score=0.97)
            assert cat != "Trash", f"Clean Czech '{line}' wrongly Trashed ({cat})"

    def test_harness_feeds_the_trust_tier_not_the_remap_cap(self):
        """Regression lock on the harness bug this module used to carry.

        Mirrors tests/test_calibration.py::
        test_fixture_languages_reach_the_guards_through_the_trust_tier, and is
        asserted on the MECHANISM for the same reason: a category-level check
        goes vacuous the moment the two numbers happen to agree, which is
        precisely when the lock matters least and the bug hides best.

        The old harness passed min(lang_score, LANG_SCORE_REMAP) and omitted
        orig_lang_score entirely. Production passes trust_lang_score to the
        structural guards and the RAW FastText score to the perplexity routes;
        those are three different numbers on the same line.
        """
        sig = LC.score_line(
            text_content="malakofauna",
            original_text="malakofauna",
            original_lang="isl_Latn",
            original_lang_score=0.56,
            perplexity=1210.0,
            known_lang_bases=_KNOWN,
            expected_langs=_EXPECTED,
        )
        assert sig["trust_lang_score"] == pytest.approx(0.56 * LC.TRUST_TIER_UNKNOWN), (
            "the structural guards must see the tier-scaled score"
        )
        assert sig["trust_lang_score"] < 0.56, "an unknown language base must actually be scaled down"
        assert sig["valid_word_ratio"] == 1.0, "the signal vector must be the full one, not the defaults"
        assert sig["garbage_density"] == 0.0
