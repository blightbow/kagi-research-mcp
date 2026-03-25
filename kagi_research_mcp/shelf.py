"""Research shelf — in-memory session-scoped document tracker.

Passively populated by arXiv, Semantic Scholar, and DOI handlers when a
single paper is inspected.  Provides BibTeX/RIS export, JSON import/export
for agent memory persistence, and an MCP tool for interactive management.

The shelf lives in the MCP server process memory for the session lifetime.
Cross-session persistence is agent-managed via export json / import.

Concurrency: the shelf is shared across all agents in an MCP session
(subagents reuse the parent's MCP connections by default).  All public
methods that touch _records are serialized by an asyncio.Lock to prevent
race conditions from concurrent tool calls (e.g. two subagents tracking
the same paper via different DOIs simultaneously).
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Annotated, Optional

from pydantic import Field as PydanticField

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Citation record
# ---------------------------------------------------------------------------

@dataclass
class CitationRecord:
    """A single tracked paper on the research shelf."""

    doi: str                                    # primary key (prefer journal DOI over preprint)
    title: str
    authors: list[str] = field(default_factory=list)   # ["Last, First", ...]
    year: Optional[int] = None
    venue: Optional[str] = None
    alt_dois: list[str] = field(default_factory=list)  # alternate DOIs (preprint ↔ journal)
    arxiv_version: Optional[str] = None          # e.g. "v7" — the specific arXiv revision inspected
    source_tool: Optional[str] = None           # "arxiv", "semantic_scholar", "doi"
    bibtex: Optional[str] = None
    citation_apa: Optional[str] = None
    orcids: Optional[dict[str, str]] = None     # {"Author Name": "0000-..."}
    added: Optional[str] = None                 # ISO 8601 timestamp
    score: Optional[int] = None                 # LLM-assigned
    confirmed: bool = False                     # LLM-managed
    notes: Optional[str] = None                 # LLM-managed freetext


def _doi_priority(doi: str) -> int:
    """Return a priority score for DOI type (higher = more authoritative).

    Journal/publisher DOIs are preferred over preprint server DOIs,
    which are preferred over synthesized arXiv DOIs.
    """
    if doi.startswith("10.48550/arXiv."):
        return 0  # synthesized arXiv DOI — lowest
    if doi.startswith("10.1101/"):
        return 1  # bioRxiv/medRxiv — preprint server
    return 2      # journal/publisher — highest


def _is_preprint_doi(doi: str) -> bool:
    """Return True if the DOI is a preprint/repository identifier."""
    return _doi_priority(doi) < 2


# ---------------------------------------------------------------------------
# BibTeX / RIS formatting
# ---------------------------------------------------------------------------

def _sanitize_bibtex_key(record: CitationRecord) -> str:
    """Generate a BibTeX entry key from first author + year."""
    if record.authors:
        # Extract last name from "Last, First" or just use full name
        first_author = record.authors[0]
        last_name = first_author.split(",")[0].strip() if "," in first_author else first_author.split()[-1]
        last_name = re.sub(r'[^a-zA-Z]', '', last_name).lower()
    else:
        last_name = "unknown"
    year = str(record.year) if record.year else "nd"
    return f"{last_name}{year}"


def _escape_bibtex(text: str) -> str:
    """Escape special LaTeX characters in BibTeX field values."""
    for char, escaped in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        text = text.replace(char, escaped)
    return text


def record_to_bibtex(record: CitationRecord) -> str:
    """Format a CitationRecord as a BibTeX entry."""
    # Use pre-existing BibTeX from S2 if available
    if record.bibtex:
        return record.bibtex.strip()

    key = _sanitize_bibtex_key(record)
    fields = []
    if record.authors:
        authors_str = " and ".join(record.authors)
        fields.append(f"  author = {{{_escape_bibtex(authors_str)}}}")
    fields.append(f"  title = {{{_escape_bibtex(record.title)}}}")
    if record.year:
        fields.append(f"  year = {{{record.year}}}")
    if record.venue:
        fields.append(f"  journal = {{{_escape_bibtex(record.venue)}}}")
    if record.doi:
        fields.append(f"  doi = {{{record.doi}}}")

    # arXiv eprint fields per convention (archivePrefix + eprint ID)
    arxiv_id = _extract_arxiv_id(record)
    if arxiv_id:
        fields.append(f"  eprint = {{{arxiv_id}}}")
        fields.append("  archivePrefix = {arXiv}")

    fields_str = ",\n".join(fields)
    return f"@misc{{{key},\n{fields_str}\n}}"


def _extract_arxiv_id(record: CitationRecord) -> Optional[str]:
    """Extract arXiv paper ID from a record's DOI or alt_dois."""
    _PREFIX = "10.48550/arXiv."
    if record.doi.startswith(_PREFIX):
        return record.doi[len(_PREFIX):]
    for alt in record.alt_dois:
        if alt.startswith(_PREFIX):
            return alt[len(_PREFIX):]
    return None


def record_to_ris(record: CitationRecord) -> str:
    """Format a CitationRecord as an RIS entry."""
    lines = ["TY  - GEN"]
    for author in record.authors:
        lines.append(f"AU  - {author}")
    lines.append(f"TI  - {record.title}")
    if record.venue:
        lines.append(f"JO  - {record.venue}")
    if record.year:
        lines.append(f"PY  - {record.year}")
    if record.doi:
        lines.append(f"DO  - {record.doi}")
    lines.append("ER  - ")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shelf storage (in-memory, session-scoped)
# ---------------------------------------------------------------------------
# The shelf lives in MCP server process memory for the session lifetime.
# Cross-session persistence is agent-managed via export json / import:
# agents write exports to their memory files and import on future sessions.
# This avoids cross-project contamination from a shared file path.


class ResearchShelf:
    """In-memory research document tracker for the current session.

    All public methods are async and serialized by an internal lock to
    prevent race conditions when multiple agents share the same MCP server.
    """

    def __init__(self):
        self._records: dict[str, CitationRecord] = {}
        self._lock = asyncio.Lock()

    def _find_by_alt_doi(self, record: CitationRecord) -> Optional[str]:
        """Find an existing record that shares a DOI with the new record.

        Checks:
        1. New record's alt_dois against existing primary keys
        2. New record's primary DOI against existing alt_dois
        Returns the existing primary key, or None.
        """
        for alt in record.alt_dois:
            if alt in self._records:
                return alt
        for key, existing in self._records.items():
            if record.doi in existing.alt_dois:
                return key
        return None

    def _track_unlocked(self, record: CitationRecord) -> None:
        """Core upsert logic — caller must hold self._lock."""
        existing_key = record.doi if record.doi in self._records else None

        if not existing_key:
            existing_key = self._find_by_alt_doi(record)

        if existing_key:
            existing = self._records[existing_key]
            record.score = existing.score
            record.confirmed = existing.confirmed
            record.notes = existing.notes
            record.added = existing.added
            all_dois = set(existing.alt_dois) | set(record.alt_dois)
            all_dois.add(existing.doi)
            all_dois.add(record.doi)
            record.doi = max(all_dois, key=_doi_priority)
            record.alt_dois = sorted(d for d in all_dois if d != record.doi)
            if existing_key != record.doi and existing_key in self._records:
                del self._records[existing_key]
        else:
            record.added = record.added or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._records[record.doi] = record

    async def track(self, record: CitationRecord) -> None:
        """Upsert a record — updates metadata, preserves score/confirmed/notes.

        Deduplicates across preprint/journal DOIs: when the same paper is
        tracked via different DOIs (e.g. arXiv + bioRxiv, or preprint +
        journal), merges into a single entry keyed on the journal DOI.
        """
        async with self._lock:
            self._track_unlocked(record)

    def _resolve_doi(self, doi: str) -> Optional[str]:
        """Resolve a DOI to its primary key, checking alt_dois as fallback."""
        if doi in self._records:
            return doi
        for key, rec in self._records.items():
            if doi in rec.alt_dois:
                return key
        return None

    async def remove(self, dois: list[str]) -> list[str]:
        """Batch remove by DOI (resolves alt_dois). Returns list of actually removed DOIs."""
        async with self._lock:
            removed = []
            for doi in dois:
                key = self._resolve_doi(doi)
                if key:
                    del self._records[key]
                    removed.append(doi)
            return removed

    async def set_score(self, doi: str, value: int) -> bool:
        """Set score for a paper. Returns False if DOI not found."""
        async with self._lock:
            key = self._resolve_doi(doi)
            if not key:
                return False
            self._records[key].score = value
            return True

    async def confirm(self, doi: str) -> bool:
        """Mark a paper as confirmed. Returns False if DOI not found."""
        async with self._lock:
            key = self._resolve_doi(doi)
            if not key:
                return False
            self._records[key].confirmed = True
            return True

    async def set_note(self, doi: str, text: str) -> bool:
        """Set freetext note. Returns False if DOI not found."""
        async with self._lock:
            key = self._resolve_doi(doi)
            if not key:
                return False
            self._records[key].notes = text
            return True

    async def list_all(self) -> list[CitationRecord]:
        """Return all records sorted by added timestamp."""
        async with self._lock:
            return sorted(self._records.values(), key=lambda r: r.added or "")

    async def export_bibtex(self) -> str:
        """Export all records as a BibTeX file."""
        async with self._lock:
            entries = [record_to_bibtex(r) for r in sorted(
                self._records.values(), key=lambda r: r.added or "",
            )]
            return "\n\n".join(entries)

    async def export_ris(self) -> str:
        """Export all records as an RIS file."""
        async with self._lock:
            entries = [record_to_ris(r) for r in sorted(
                self._records.values(), key=lambda r: r.added or "",
            )]
            return "\n\n".join(entries)

    async def export_json(self) -> str:
        """Export full shelf as JSON string for agent memory persistence."""
        async with self._lock:
            return json.dumps(
                {doi: asdict(rec) for doi, rec in self._records.items()},
                indent=2, ensure_ascii=False,
            )

    async def import_json(self, data: str) -> tuple[int, int]:
        """Import shelf from JSON string. Returns (new_count, updated_count).

        Acquires the lock once for the entire import batch to ensure
        atomicity and avoid deadlock (track() also acquires the lock).
        """
        async with self._lock:
            parsed = json.loads(data)
            new_count = 0
            updated_count = 0
            for doi, rec_dict in parsed.items():
                is_new = doi not in self._records
                self._track_unlocked(CitationRecord(**rec_dict))
                if is_new:
                    new_count += 1
                else:
                    updated_count += 1
            return new_count, updated_count

    async def clear(self) -> int:
        """Remove all entries. Returns count removed."""
        async with self._lock:
            count = len(self._records)
            self._records.clear()
            return count

    async def status_line(self) -> Optional[str]:
        """Compact status for frontmatter. Returns None if shelf is empty."""
        async with self._lock:
            total = len(self._records)
            if total == 0:
                return None
            confirmed = sum(1 for r in self._records.values() if r.confirmed)
            return f"{total} tracked ({confirmed} confirmed) — use ResearchShelf to review"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_shelf: Optional[ResearchShelf] = None


def _get_shelf() -> ResearchShelf:
    """Return the global in-memory shelf instance."""
    global _shelf
    if _shelf is None:
        _shelf = ResearchShelf()
    return _shelf


def _reset_shelf() -> None:
    """Reset the global shelf instance (for testing)."""
    global _shelf
    _shelf = None


async def _track_on_shelf(record: CitationRecord) -> Optional[str]:
    """Track a record on the shelf and return the status line.

    Fire-and-forget helper for handlers — catches all exceptions
    and returns None on failure so tracking never blocks tool output.
    """
    try:
        shelf = _get_shelf()
        await shelf.track(record)
        return await shelf.status_line()
    except Exception:
        logger.debug("Shelf tracking failed for %s", record.doi, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_shelf_list(records: list[CitationRecord]) -> str:
    """Format shelf contents as a compact markdown table."""
    if not records:
        return "Research shelf is empty."

    lines = [
        "| # | Score | Status | Title | DOI | Source |",
        "|---|-------|--------|-------|-----|--------|",
    ]
    for i, r in enumerate(records, 1):
        score = str(r.score) if r.score is not None else "—"
        status = "confirmed" if r.confirmed else ""
        title = r.title[:50] + "..." if len(r.title) > 50 else r.title
        source = r.source_tool or "—"
        lines.append(f"| {i} | {score} | {status} | {title} | {r.doi} | {source} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------

async def research_shelf(
    action: Annotated[str, PydanticField(
        description=(
            "The operation to perform. "
            "list: show all tracked papers. "
            "confirm: mark a paper as confirmed/useful. "
            "remove: batch remove papers by DOI (comma-separated). "
            "score: set an integer score for a paper. "
            "note: set a freetext note on a paper. "
            "export: export shelf in bibtex, ris, or json format. "
            "import: import shelf from a JSON export string. "
            "clear: remove all entries from the shelf."
        ),
    )],
    query: Annotated[str, PydanticField(
        description=(
            "For confirm/score/note: the DOI of the paper. "
            "For remove: comma-separated DOIs to remove. "
            "For score: DOI followed by space and integer value (e.g. '10.1234/foo 8'). "
            "For note: DOI followed by space and note text. "
            "For export: format name (bibtex, ris, json). "
            "For import: the JSON string to import. "
            "For list/clear: ignored (pass any value)."
        ),
    )] = "",
) -> str:
    """Manage the research shelf — a persistent tracker for inspected papers."""
    shelf = _get_shelf()

    if action == "list":
        records = await shelf.list_all()
        return _format_shelf_list(records)

    elif action == "confirm":
        doi = query.strip()
        if not doi:
            return "Error: DOI is required for confirm action."
        if await shelf.confirm(doi):
            return f"Confirmed: {doi}"
        return f"Error: DOI not found on shelf: {doi}"

    elif action == "remove":
        dois = [d.strip() for d in query.split(",") if d.strip()]
        if not dois:
            return "Error: At least one DOI is required for remove action."
        removed = await shelf.remove(dois)
        if removed:
            return f"Removed {len(removed)} paper(s): {', '.join(removed)}"
        return "No matching DOIs found on shelf."

    elif action == "score":
        parts = query.strip().split(None, 1)
        if len(parts) != 2:
            return "Error: score action requires 'DOI VALUE' (e.g. '10.1234/foo 8')."
        doi, value_str = parts
        try:
            value = int(value_str)
        except ValueError:
            return f"Error: Score must be an integer, got '{value_str}'."
        if await shelf.set_score(doi, value):
            return f"Score set to {value} for {doi}"
        return f"Error: DOI not found on shelf: {doi}"

    elif action == "note":
        parts = query.strip().split(None, 1)
        if len(parts) < 2:
            return "Error: note action requires 'DOI TEXT'."
        doi, text = parts
        if await shelf.set_note(doi, text):
            return f"Note set for {doi}"
        return f"Error: DOI not found on shelf: {doi}"

    elif action == "export":
        fmt = query.strip().lower()
        if fmt == "bibtex":
            result = await shelf.export_bibtex()
            return result if result else "Shelf is empty."
        elif fmt == "ris":
            result = await shelf.export_ris()
            return result if result else "Shelf is empty."
        elif fmt == "json":
            return await shelf.export_json()
        else:
            return f"Error: Unknown export format '{fmt}'. Use bibtex, ris, or json."

    elif action == "import":
        if not query.strip():
            return "Error: JSON data is required for import action."
        try:
            new, updated = await shelf.import_json(query)
            parts = []
            if new:
                parts.append(f"{new} new")
            if updated:
                parts.append(f"{updated} updated")
            return f"Imported: {', '.join(parts)}." if parts else "No records in import data."
        except (json.JSONDecodeError, TypeError) as e:
            return f"Error: Invalid JSON — {e}"

    elif action == "clear":
        count = await shelf.clear()
        return f"Cleared {count} record(s) from shelf."

    else:
        return (
            f"Error: Unknown action '{action}'. "
            "Valid actions: list, confirm, remove, score, note, export, import, clear"
        )
