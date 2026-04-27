"""YouTube integration via yt-dlp (metadata) and youtube-transcript-api (captions).

Currently implements the ``video`` and ``transcript`` actions. Channel,
playlist, and search actions land in later commits per the implementation
sequencing in the design discussion.

URL detection covers ``youtube.com/watch``, ``youtu.be``, ``shorts``, ``clip``,
``@handle``, ``/channel/UC...``, ``/c/`` , ``/user/``, and ``/playlist``.
``music.youtube.com`` is intentionally excluded — it's deferred as a sibling
tool because the music-track shape (album/artist/track) differs meaningfully
from the video shape.

Transcript rendering uses a quality-aware coalescer that snaps window
boundaries to natural pauses (or sentence-end punctuation when caption
quality permits), then renders one of four output shapes — ``compact``
(default; sparse anchors plus outlier pause markers), ``absolute`` (per-line
timestamps), ``none`` (flat text, no timing), and ``structured`` (YAML).
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional

from pydantic import Field

from .markdown import (
    FMEntries,
    _build_frontmatter,
    _fence_content,
    _TRUST_ADVISORY,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------
# Patterns target only youtube.com / youtu.be / m.youtube.com.
# music.youtube.com is intentionally NOT matched (deferred to a sibling tool).

# Video IDs are always exactly 11 chars in YouTube's base64-ish alphabet.
_VIDEO_ID = r"[A-Za-z0-9_-]{11}"

_YT_VIDEO_RE = re.compile(
    r"https?://"
    r"(?:"
        r"(?:www\.|m\.)?youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|embed/|v/)"
        r"|youtu\.be/"
    r")"
    rf"({_VIDEO_ID})",
    re.IGNORECASE,
)

# Clip URLs use a different identifier shape (variable length, e.g.
# ``UgkxAbCdEf12...``) and need their own pattern. yt-dlp resolves them
# to the underlying video on extraction; we just need to recognize the
# kind here so the dispatcher routes them to ``_video``.
_YT_CLIP_RE = re.compile(
    r"https?://(?:www\.|m\.)?youtube\.com/clip/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)

# Handle channels: /@handle (case-sensitive in canonical form).
_YT_CHANNEL_HANDLE_RE = re.compile(
    r"https?://(?:www\.|m\.)?youtube\.com/@([A-Za-z0-9._-]+)",
    re.IGNORECASE,
)

# Channel ID / vanity / legacy user URLs.
_YT_CHANNEL_ID_RE = re.compile(
    r"https?://(?:www\.|m\.)?youtube\.com/"
    r"(?:channel/(UC[A-Za-z0-9_-]{22})|c/([A-Za-z0-9._-]+)|user/([A-Za-z0-9._-]+))",
    re.IGNORECASE,
)

# Playlist IDs are variable-length, prefixed PL/UU/LL/FL/RD/WL/OL.
_YT_PLAYLIST_RE = re.compile(
    r"https?://(?:www\.|m\.)?youtube\.com/playlist\?(?:[^#]*&)?list=([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)

# music.youtube.com — explicitly *excluded* from the youtube tool's scope.
# Detection here only exists so we can emit a clear "use a different tool"
# error instead of misidentifying as a regular video.
_YT_MUSIC_RE = re.compile(
    r"https?://music\.youtube\.com/",
    re.IGNORECASE,
)


def _detect_youtube_url(url: str) -> Optional[tuple[str, str]]:
    """Classify a YouTube URL.

    Returns ``(kind, identifier)`` on match, or ``None`` for non-YouTube
    URLs. Kinds: ``"video"``, ``"channel"``, ``"playlist"``, ``"music"``.
    The ``music`` kind is recognized only to produce an informative
    error; callers should treat it as out-of-scope.
    """
    if _YT_MUSIC_RE.search(url):
        return ("music", url)
    m = _YT_VIDEO_RE.search(url)
    if m:
        return ("video", m.group(1))
    m = _YT_CLIP_RE.search(url)
    if m:
        return ("video", m.group(1))
    m = _YT_CHANNEL_HANDLE_RE.search(url)
    if m:
        return ("channel", "@" + m.group(1))
    m = _YT_CHANNEL_ID_RE.search(url)
    if m:
        ident = m.group(1) or m.group(2) or m.group(3) or ""
        return ("channel", ident)
    m = _YT_PLAYLIST_RE.search(url)
    if m:
        return ("playlist", m.group(1))
    return None


# ---------------------------------------------------------------------------
# yt-dlp instance (lazy singleton, video mode)
# ---------------------------------------------------------------------------
# A single YoutubeDL instance per process is the recommended embedding
# pattern: PoToken caches and JS player solves are instance-scoped, so reuse
# avoids redundant work on subsequent calls. Channel/playlist/search actions
# (added in later commits) need different opts (extract_flat) and will get
# their own singleton.

_YDL_OPTS_VIDEO: dict[str, Any] = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "noplaylist": True,
    "extract_flat": False,
    "logger": logging.getLogger("yt_dlp"),
}

_ydl_video: Any = None


def _get_ydl_video() -> Any:
    """Return the lazily-constructed video-mode YoutubeDL singleton."""
    global _ydl_video
    if _ydl_video is None:
        from yt_dlp import YoutubeDL  # type: ignore[import-not-found]
        _ydl_video = YoutubeDL(_YDL_OPTS_VIDEO)
    return _ydl_video


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

def _map_yt_dlp_error(exc: Exception) -> str:
    """Translate a yt-dlp exception to a user-facing error string.

    yt-dlp's exception hierarchy distinguishes only a few classes
    cleanly (geo, unavailable); bot detection, private, age-restricted,
    and members-only all surface as ``ExtractorError`` / ``DownloadError``
    with the relevant text in the message. Match on substrings.
    """
    try:
        from yt_dlp.utils import (  # type: ignore[import-not-found]
            DownloadError,
            ExtractorError,
            GeoRestrictedError,
            UnavailableVideoError,
        )
    except ImportError:
        # yt-dlp not importable — surface the raw type/message
        return f"Error: yt-dlp extraction failed ({type(exc).__name__})."

    if isinstance(exc, GeoRestrictedError):
        return "Error: Video is geo-restricted in this region."
    if isinstance(exc, UnavailableVideoError):
        return "Error: Video is unavailable."
    if isinstance(exc, (ExtractorError, DownloadError)):
        msg = str(exc).lower()
        if "sign in to confirm you" in msg or "confirm you're not a bot" in msg:
            return (
                "Error: YouTube blocked the request as suspected bot traffic. "
                "If on a residential connection, retry shortly. "
                "On cloud IPs, route through a residential proxy via HTTPS_PROXY."
            )
        if "private video" in msg:
            return "Error: Video is private."
        if "members-only" in msg or "members only" in msg:
            return "Error: Video is members-only and requires authentication."
        if "age" in msg and ("restrict" in msg or "confirm your age" in msg):
            return "Error: Video is age-restricted; cannot access without auth."
        if "video unavailable" in msg or "this video is not available" in msg:
            return "Error: Video unavailable."
        short = str(exc).splitlines()[0][:200]
        return f"Error: yt-dlp extraction failed ({type(exc).__name__}): {short}"
    short = str(exc).splitlines()[0][:200]
    return f"Error: yt-dlp extraction failed ({type(exc).__name__}): {short}"


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: Optional[float]) -> Optional[str]:
    """Render a seconds count as ``M:SS`` or ``H:MM:SS``."""
    if seconds is None:
        return None
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _format_upload_date(yyyymmdd: Optional[str]) -> Optional[str]:
    """Convert yt-dlp's ``YYYYMMDD`` date format to ISO ``YYYY-MM-DD``."""
    if not yyyymmdd or len(yyyymmdd) != 8 or not yyyymmdd.isdigit():
        return yyyymmdd
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def _captions_summary(info: dict) -> tuple[list[str], bool]:
    """Return ``(available_languages, has_auto_only)``.

    Manual and automatic captions are merged into a single sorted
    language list; the second element flags videos where only
    auto-generated captions exist (a reliable quality signal — see the
    transcript renderer plan for how this routes branching later).
    """
    manual = list((info.get("subtitles") or {}).keys())
    auto = list((info.get("automatic_captions") or {}).keys())
    langs = sorted(set(manual + auto))
    has_auto_only = bool(auto and not manual)
    return langs, has_auto_only


# ---------------------------------------------------------------------------
# Action: video
# ---------------------------------------------------------------------------

async def _video(url: str) -> str:
    """Fetch metadata + description for a single YouTube video URL."""
    ydl = _get_ydl_video()
    try:
        info = await asyncio.to_thread(ydl.extract_info, url, download=False)
    except Exception as exc:
        return _map_yt_dlp_error(exc)

    if info is None:
        return f"Error: yt-dlp returned no metadata for {url}"

    info = ydl.sanitize_info(info)
    if not isinstance(info, dict):
        return f"Error: Unexpected yt-dlp response shape for {url}"

    video_id = info.get("id") or ""
    title = info.get("title") or "Untitled"
    description = info.get("description") or ""

    captions_langs, captions_auto_only = _captions_summary(info)

    fm_entries = FMEntries({
        "title": title,
        "source": (
            f"https://www.youtube.com/watch?v={video_id}" if video_id else url
        ),
        "api": "yt-dlp",
        "video_id": video_id,
        "channel": info.get("channel") or info.get("uploader"),
        "channel_id": info.get("channel_id"),
        "channel_url": info.get("channel_url"),
        "duration": _format_duration(info.get("duration")),
        "upload_date": _format_upload_date(info.get("upload_date")),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "language": info.get("language"),
        "live_status": info.get("live_status"),
        "availability": info.get("availability"),
        "captions_available": captions_langs or None,
        "captions_auto_only": True if captions_auto_only else None,
        "trust": _TRUST_ADVISORY,
    })

    fm = _build_frontmatter(fm_entries)
    body = description.strip() if description else "(no description)"
    return fm + "\n\n" + _fence_content(body, title=title)


# ---------------------------------------------------------------------------
# Transcript: data types and constants
# ---------------------------------------------------------------------------

TimestampMode = Literal["compact", "absolute", "none", "structured"]


@dataclass(frozen=True)
class Segment:
    """A single caption cue: start time, duration, and text.

    Mirrors the shape returned by ``youtube-transcript-api``'s
    ``FetchedTranscriptSnippet`` but as a frozen value object that is
    safe to share across the coalescer and renderers.
    """
    start: float
    duration: float
    text: str

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class Window:
    """A coalesced ~30s span of consecutive segments.

    Windows are the unit of presentation (one anchor per window) and the
    unit a future Tantivy index will treat as a document. A window's
    ``start`` and ``end`` come from its first and last segments.
    """
    start: float
    end: float
    segments: tuple[Segment, ...]


# Window coalescing band: target ~30s, with a [25, 35] tolerance band where
# we look for natural pause boundaries. Matches WhisperX Cut & Merge: cap
# at the upper bound, but prefer cuts at the largest pause within the band
# rather than at the time threshold itself.
_WINDOW_TARGET_DURATION = 30.0
_WINDOW_MIN_DURATION = 25.0
_WINDOW_MAX_DURATION = 35.0

# Inter-segment gap that earns a soft window boundary. Gaps shorter than
# this are treated as continuous speech.
_PAUSE_BOUNDARY = 1.0

# Punctuation density threshold for the quality gate: above this, treat
# the transcript as punctuated and route to the sentence-aware coalescer.
# 0.05 sentence-enders per word ≈ one sentence per 20 words, which is the
# floor for natural prose (typical English averages ~14 wpw per sentence).
_PUNCTUATION_DENSITY_THRESHOLD = 0.05

# Outlier-gap detection: pure rolling-median rule. For windows ≥ this many
# inter-segment gaps, compute the rolling median and flag gaps exceeding
# max(2 × median, 1.5s). Below the threshold, fall back to a fixed cutoff
# because the rolling median is unstable on small samples.
_OUTLIER_WINDOW = 10
_OUTLIER_MULTIPLE = 2.0
_OUTLIER_FLOOR = 1.5
_OUTLIER_FALLBACK = 3.0

# Sentence-final punctuation set used by the sentence-aware coalescer.
_SENTENCE_END = (".", "!", "?")


# ---------------------------------------------------------------------------
# Transcript: helpers
# ---------------------------------------------------------------------------

def _mmss(seconds: float) -> str:
    """Format ``seconds`` as zero-padded ``MM:SS`` or ``HH:MM:SS``."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _median(xs: list[float]) -> float:
    """Statistical median over a non-empty list of floats."""
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _punctuation_density(segments: list[Segment]) -> float:
    """Estimate sentence-ender density (per word) across all segments."""
    if not segments:
        return 0.0
    text = " ".join(s.text for s in segments)
    words = text.split()
    if not words:
        return 0.0
    enders = sum(1 for c in text if c in _SENTENCE_END)
    return enders / len(words)


def _segment_ends_sentence(seg: Segment) -> bool:
    """Whether this segment's text ends with sentence-final punctuation."""
    t = seg.text.rstrip()
    return bool(t) and t[-1] in _SENTENCE_END


def _detect_outlier_gaps(segments: list[Segment]) -> list[bool]:
    """Flag each inter-segment gap as an outlier or not.

    Returns a list aligned with ``segments`` where ``out[i]`` is ``True``
    iff the gap *after* segment ``i`` is unusually large. ``out[-1]`` is
    always ``False`` (the last segment has no following gap).

    For transcripts shorter than ``_OUTLIER_WINDOW`` gaps, applies a fixed
    threshold (``_OUTLIER_FALLBACK``) since the rolling median is unstable
    on small samples. For longer transcripts, computes a rolling median
    over a window of ``_OUTLIER_WINDOW`` gaps centered on each position
    and flags gaps exceeding ``max(_OUTLIER_MULTIPLE × median, _OUTLIER_FLOOR)``.
    """
    n = len(segments)
    if n < 2:
        return [False] * n

    gaps = [
        segments[i + 1].start - segments[i].end
        for i in range(n - 1)
    ]

    if len(gaps) < _OUTLIER_WINDOW:
        result = [g >= _OUTLIER_FALLBACK for g in gaps]
        result.append(False)
        return result

    half = _OUTLIER_WINDOW // 2
    out: list[bool] = []
    for i, gap in enumerate(gaps):
        lo = max(0, i - half)
        hi = min(len(gaps), lo + _OUTLIER_WINDOW)
        med = _median(gaps[lo:hi])
        threshold = max(_OUTLIER_MULTIPLE * med, _OUTLIER_FLOOR)
        out.append(gap >= threshold)
    out.append(False)
    return out


# ---------------------------------------------------------------------------
# Transcript: window coalescer (quality-aware, branched)
# ---------------------------------------------------------------------------

def coalesce_windows(
    segments: list[Segment],
    *,
    sentence_aware: bool,
    minimum: float = _WINDOW_MIN_DURATION,
    maximum: float = _WINDOW_MAX_DURATION,
    pause_boundary: float = _PAUSE_BOUNDARY,
) -> list[Window]:
    """Coalesce timed segments into ~30s windows.

    Walks segments in order, accumulating until the running duration
    enters the [minimum, maximum] tolerance band. Once in the band, cuts
    at the next natural boundary (sentence-end punctuation when
    ``sentence_aware`` is True, otherwise the next pause >= ``pause_boundary``).
    Forces a cut when adding the next segment would exceed ``maximum``.
    WhisperX Cut & Merge with a text-quality switch.

    The ``sentence_aware`` flag is the load-bearing branch: punctuated
    captions get cuts that respect prose structure; unpunctuated captions
    fall back to pure pause-based segmentation, which is the safest
    strategy when no linguistic signal is reliable.
    """
    if not segments:
        return []

    windows: list[Window] = []
    current: list[Segment] = []

    def _close(seg_list: list[Segment]) -> None:
        windows.append(Window(
            start=seg_list[0].start,
            end=seg_list[-1].end,
            segments=tuple(seg_list),
        ))

    for seg in segments:
        if not current:
            current.append(seg)
            continue

        # Would adding this segment exceed the upper bound? Cut first.
        prospective_dur = seg.end - current[0].start
        if prospective_dur > maximum:
            _close(current)
            current = [seg]
            continue

        # In the tolerance band: look for a natural boundary at the join.
        candidate_dur = current[-1].end - current[0].start
        if candidate_dur >= minimum:
            cut = False
            if sentence_aware and _segment_ends_sentence(current[-1]):
                cut = True
            else:
                gap = seg.start - current[-1].end
                if gap >= pause_boundary:
                    cut = True
            if cut:
                _close(current)
                current = [seg]
                continue

        current.append(seg)

    if current:
        _close(current)
    return windows


# ---------------------------------------------------------------------------
# Transcript: rendering modes
# ---------------------------------------------------------------------------

def _render_flat(windows: list[Window]) -> str:
    """Concatenate all segment text with single spaces; no timing."""
    parts = [s.text.strip() for w in windows for s in w.segments]
    return " ".join(p for p in parts if p)


def _render_absolute(windows: list[Window]) -> str:
    """Per-line absolute ``[MM:SS]`` timestamps."""
    lines = []
    for w in windows:
        for s in w.segments:
            text = s.text.strip()
            if text:
                lines.append(f"[{_mmss(s.start)}] {text}")
    return "\n".join(lines)


def _render_compact(windows: list[Window]) -> str:
    """Default rendering: anchor per window, segments on own lines, outlier
    pause markers between segments, blank lines between windows.

    Outlier detection runs over the FULL transcript so the rolling median
    is stable; per-window detection would oscillate on short windows.
    Inter-window gaps are not annotated because the next window's anchor
    already implies the transition.
    """
    if not windows:
        return ""

    all_segments = [s for w in windows for s in w.segments]
    outliers = _detect_outlier_gaps(all_segments)

    # Map (window_idx, in_window_idx) -> bool by walking the flat sequence
    flat_idx = 0
    outlier_at: dict[tuple[int, int], bool] = {}
    for wi, w in enumerate(windows):
        for si in range(len(w.segments)):
            outlier_at[(wi, si)] = outliers[flat_idx]
            flat_idx += 1

    lines: list[str] = []
    for wi, w in enumerate(windows):
        if wi > 0:
            lines.append("")  # blank line between windows
        lines.append(f"[{_mmss(w.start)}]")
        n = len(w.segments)
        for si, seg in enumerate(w.segments):
            text = seg.text.strip()
            if text:
                lines.append(text)
            # Inline pause marker only between segments WITHIN this window
            if si < n - 1 and outlier_at.get((wi, si), False):
                gap = w.segments[si + 1].start - seg.end
                lines.append(f"[+{int(round(gap))}s]")
    return "\n".join(lines)


def _render_structured(windows: list[Window]) -> str:
    """YAML list of segments with start/duration/text, for machine consumers."""
    import yaml
    data = []
    for w in windows:
        for s in w.segments:
            data.append({
                "t": round(s.start, 2),
                "d": round(s.duration, 2),
                "text": s.text,
            })
    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


def render_transcript(
    windows: list[Window],
    *,
    mode: TimestampMode = "compact",
) -> str:
    """Render coalesced windows in the requested timestamp mode."""
    if mode == "none":
        return _render_flat(windows)
    if mode == "absolute":
        return _render_absolute(windows)
    if mode == "structured":
        return _render_structured(windows)
    return _render_compact(windows)


# ---------------------------------------------------------------------------
# Transcript: error mapping
# ---------------------------------------------------------------------------

def _map_transcript_error(exc: Exception) -> str:
    """Translate a youtube-transcript-api exception to a user-facing string.

    Order matters: more-specific subclasses are checked before their
    superclasses (``IpBlocked`` before ``RequestBlocked``).
    """
    try:
        from youtube_transcript_api import (
            AgeRestricted,
            CouldNotRetrieveTranscript,
            InvalidVideoId,
            IpBlocked,
            NoTranscriptFound,
            PoTokenRequired,
            RequestBlocked,
            TranscriptsDisabled,
            VideoUnavailable,
            YouTubeRequestFailed,
        )
    except ImportError:
        return f"Error: youtube-transcript-api error ({type(exc).__name__})."

    if isinstance(exc, IpBlocked):
        return (
            "Error: YouTube blocked the transcript request based on IP "
            "reputation. If running from a cloud IP (AWS/GCP/Azure/etc.), "
            "configure HTTPS_PROXY to a residential proxy."
        )
    if isinstance(exc, RequestBlocked):
        return (
            "Error: YouTube blocked the transcript request as suspected "
            "bot traffic. Retry shortly, or configure HTTPS_PROXY to a "
            "residential proxy if blocks persist."
        )
    if isinstance(exc, PoTokenRequired):
        return (
            "Error: This video's captions require a Botguard PoToken; "
            "youtube-transcript-api has no current workaround. A yt-dlp "
            "fallback path is on the roadmap."
        )
    if isinstance(exc, TranscriptsDisabled):
        return "Error: The uploader has disabled transcripts for this video."
    if isinstance(exc, NoTranscriptFound):
        return (
            "Error: No transcript available in the requested language(s). "
            "Try omitting the languages= argument to fall back to the "
            "video's default caption track."
        )
    if isinstance(exc, AgeRestricted):
        return (
            "Error: Video is age-restricted; transcript unavailable "
            "without authentication."
        )
    if isinstance(exc, VideoUnavailable):
        return "Error: Video unavailable."
    if isinstance(exc, InvalidVideoId):
        return "Error: Invalid YouTube video ID."
    if isinstance(exc, YouTubeRequestFailed):
        short = str(exc).splitlines()[0][:200]
        return f"Error: YouTube request failed: {short}"
    if isinstance(exc, CouldNotRetrieveTranscript):
        short = str(exc).splitlines()[0][:200]
        return f"Error: Could not retrieve transcript ({type(exc).__name__}): {short}"
    short = str(exc).splitlines()[0][:200]
    return f"Error: Transcript fetch failed ({type(exc).__name__}): {short}"


# ---------------------------------------------------------------------------
# Action: transcript
# ---------------------------------------------------------------------------

def _fetch_transcript_sync(video_id: str, languages: list[str]):
    """Sync wrapper around YouTubeTranscriptApi().fetch().

    Lives at module scope so ``asyncio.to_thread`` can pickle it cleanly
    on platforms that need it. The library itself is sync-only.
    """
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    return api.fetch(video_id, languages=languages)


async def _transcript(
    url: str,
    languages: list[str],
    timestamps: TimestampMode,
) -> str:
    """Fetch and render the transcript for a YouTube video URL."""
    detected = _detect_youtube_url(url)
    if detected is None:
        return f"Error: Not a recognized YouTube URL: {url}"
    if detected[0] == "music":
        return (
            "Error: music.youtube.com URLs are out of scope for this tool."
        )
    if detected[0] != "video":
        return (
            f"Error: URL is a {detected[0]}, not a video. "
            "The transcript action only accepts video URLs."
        )
    video_id = detected[1]

    try:
        fetched = await asyncio.to_thread(
            _fetch_transcript_sync, video_id, languages,
        )
    except Exception as exc:
        return _map_transcript_error(exc)

    snippets = list(fetched.snippets)
    if not snippets:
        return "Error: Transcript fetched but contains no segments."

    # Caption cues often contain embedded newlines for display wrapping
    # (a single utterance rendered across two visual lines on the player).
    # Those newlines aren't semantic and break readability when rendered;
    # collapse internal whitespace to single spaces here so each segment
    # presents as one coherent line in compact/absolute output.
    segments = [
        Segment(
            start=float(s.start),
            duration=float(s.duration),
            text=" ".join(s.text.split()),
        )
        for s in snippets
    ]

    is_auto = bool(fetched.is_generated)
    density = _punctuation_density(segments)
    sentence_aware = (not is_auto) and density >= _PUNCTUATION_DENSITY_THRESHOLD

    windows = coalesce_windows(segments, sentence_aware=sentence_aware)
    body = render_transcript(windows, mode=timestamps)

    fm_entries = FMEntries({
        "source": f"https://www.youtube.com/watch?v={video_id}",
        "api": "youtube-transcript-api",
        "video_id": video_id,
        "transcript_language": fetched.language_code,
        "transcript_kind": "auto" if is_auto else "manual",
        "total_segments": len(segments),
        "total_windows": len(windows),
        "chunking_strategy": "sentence" if sentence_aware else "time_window",
        "duration": _format_duration(segments[-1].end),
        "trust": _TRUST_ADVISORY,
    })

    fm = _build_frontmatter(fm_entries)
    title = f"Transcript ({fetched.language_code})"
    return fm + "\n\n" + _fence_content(body, title=title)


# ---------------------------------------------------------------------------
# MCP-facing dispatcher
# ---------------------------------------------------------------------------

async def youtube(
    action: Annotated[str, Field(
        description=(
            "The operation to perform. "
            "video: fetch video metadata + description from a YouTube URL. "
            "transcript: fetch the caption transcript for a video URL."
        ),
    )],
    url: Annotated[Optional[str], Field(
        description=(
            "YouTube URL for the 'video' or 'transcript' action. "
            "Accepts watch, youtu.be, shorts, clip, embed, and v/ forms."
        ),
    )] = None,
    languages: Annotated[Optional[list[str]], Field(
        description=(
            "For 'transcript': caption language preference list, tried in "
            "order (e.g. ['en', 'en-US']). Defaults to ['en']."
        ),
    )] = None,
    timestamps: Annotated[TimestampMode, Field(
        description=(
            "For 'transcript': output shape. "
            "'compact' (default) emits sparse anchors plus inline markers "
            "for unusually long pauses, with each source caption cue on "
            "its own line. 'absolute' emits a per-line [MM:SS] prefix on "
            "every cue. 'none' returns flat text with no timing. "
            "'structured' returns a YAML list of {t, d, text} triples for "
            "machine consumers."
        ),
    )] = "compact",
) -> str:
    """YouTube integration via yt-dlp and youtube-transcript-api."""
    if action == "video":
        if not url:
            return "Error: 'url' is required for action='video'."
        kind = _detect_youtube_url(url)
        if kind is None:
            return f"Error: Not a recognized YouTube URL: {url}"
        if kind[0] == "music":
            return (
                "Error: music.youtube.com URLs are out of scope for this tool. "
                "Music tracks have a different shape (album/artist/track) and "
                "will be handled by a sibling tool."
            )
        if kind[0] != "video":
            return (
                f"Error: URL is a {kind[0]}, not a video. "
                f"The {kind[0]} action is not yet implemented."
            )
        return await _video(url)
    if action == "transcript":
        if not url:
            return "Error: 'url' is required for action='transcript'."
        return await _transcript(
            url,
            languages=languages or ["en"],
            timestamps=timestamps,
        )
    return f"Error: Unknown action '{action}'. Valid actions: video, transcript"
