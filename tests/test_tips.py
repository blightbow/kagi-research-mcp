"""Tests for the `tip` frontmatter field: registry, fire-once ledger, scoping."""

import pytest

from parkour_mcp import markdown
from parkour_mcp.markdown import FMEntries, _build_frontmatter


@pytest.fixture(autouse=True)
def _demo_tips(monkeypatch):
    """Register two throwaway tips for every test in this module."""
    monkeypatch.setitem(markdown._TIPS, "demo_tip", "Demo tip text.")
    monkeypatch.setitem(markdown._TIPS, "other_tip", "Other tip text.")


def test_registered_tip_renders():
    fm = FMEntries({"source": "https://example.com"})
    fm.set_tip("demo_tip")
    out = _build_frontmatter(fm)
    assert "tip: Demo tip text." in out
    assert "source: https://example.com" in out


def test_tip_fires_once_per_process():
    fm1 = FMEntries({"source": "https://a.example"})
    fm1.set_tip("demo_tip")
    assert "tip: Demo tip text." in _build_frontmatter(fm1)

    # Second build, different FMEntries — the ledger suppresses the re-fire.
    fm2 = FMEntries({"source": "https://b.example"})
    fm2.set_tip("demo_tip")
    assert "tip:" not in _build_frontmatter(fm2)


def test_url_scoped_tip_fires_once_per_url():
    url_a = "https://a.example/page"
    url_b = "https://b.example/page"

    fm_a1 = FMEntries({})
    fm_a1.set_tip("demo_tip", url=url_a)
    assert "tip: Demo tip text." in _build_frontmatter(fm_a1)

    # Same base tip, different URL — still fires.
    fm_b = FMEntries({})
    fm_b.set_tip("demo_tip", url=url_b)
    assert "tip: Demo tip text." in _build_frontmatter(fm_b)

    # Same URL again — suppressed.
    fm_a2 = FMEntries({})
    fm_a2.set_tip("demo_tip", url=url_a)
    assert "tip:" not in _build_frontmatter(fm_a2)


def test_url_scoped_and_session_scoped_are_independent():
    # A session-scoped fire does not consume the URL-scoped ledger entry.
    fm_session = FMEntries({})
    fm_session.set_tip("demo_tip")
    assert "tip: Demo tip text." in _build_frontmatter(fm_session)

    fm_url = FMEntries({})
    fm_url.set_tip("demo_tip", url="https://example.com")
    assert "tip: Demo tip text." in _build_frontmatter(fm_url)


def test_set_tip_unknown_id_raises():
    fm = FMEntries({})
    with pytest.raises(ValueError, match="not registered"):
        fm.set_tip("no_such_tip")


def test_set_tip_rejects_scope_suffix_in_id():
    fm = FMEntries({})
    with pytest.raises(ValueError, match="without a '::'"):
        fm.set_tip("demo_tip::abc123")


def test_set_tip_is_single_write():
    fm = FMEntries({})
    fm.set_tip("demo_tip")
    with pytest.raises(TypeError, match="single-write"):
        fm.set_tip("other_tip")


def test_direct_tip_assignment_rejected():
    fm = FMEntries({})
    with pytest.raises(TypeError, match="set_tip"):
        fm["tip"] = "some text"


def test_tip_via_update_rejected():
    fm = FMEntries({})
    with pytest.raises(TypeError, match="set_tip"):
        fm.update({"tip": "demo_tip"})


def test_unknown_tip_id_renders_nothing():
    # A plain-dict caller smuggling an unregistered id — silently dropped,
    # the rest of the frontmatter still builds.
    out = _build_frontmatter({"source": "https://example.com", "tip": "ghost_tip"})
    assert "tip:" not in out
    assert "source: https://example.com" in out


def test_ledger_reset_fixture_isolates_tests():
    # If the autouse _reset_fired_tips fixture works, demo_tip is unfired
    # here even though sibling tests in this module fired it.
    fm = FMEntries({})
    fm.set_tip("demo_tip")
    assert "tip: Demo tip text." in _build_frontmatter(fm)
