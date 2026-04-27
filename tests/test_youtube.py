"""Tests for parkour_mcp.youtube module."""

import sys

import pytest

import parkour_mcp.youtube  # noqa: F401
_yt_module = sys.modules["parkour_mcp.youtube"]

from parkour_mcp.youtube import (  # noqa: E402
    Segment,
    Window,
    _captions_summary,
    _detect_outlier_gaps,
    _detect_youtube_url,
    _format_duration,
    _format_upload_date,
    _map_transcript_error,
    _map_yt_dlp_error,
    _mmss,
    _punctuation_density,
    _segment_ends_sentence,
    coalesce_windows,
    render_transcript,
    youtube,
)


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------

class TestDetectYouTubeUrl:
    def test_watch_url(self):
        assert _detect_youtube_url(
            "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        ) == ("video", "jNQXAC9IVRw")

    def test_watch_url_with_extra_query(self):
        assert _detect_youtube_url(
            "https://www.youtube.com/watch?feature=share&v=jNQXAC9IVRw&t=42s"
        ) == ("video", "jNQXAC9IVRw")

    def test_short_url(self):
        assert _detect_youtube_url("https://youtu.be/jNQXAC9IVRw") == (
            "video", "jNQXAC9IVRw",
        )

    def test_short_url_with_timestamp(self):
        assert _detect_youtube_url("https://youtu.be/jNQXAC9IVRw?t=10") == (
            "video", "jNQXAC9IVRw",
        )

    def test_shorts_url(self):
        assert _detect_youtube_url(
            "https://www.youtube.com/shorts/abc123def45"
        ) == ("video", "abc123def45")

    def test_clip_url(self):
        assert _detect_youtube_url(
            "https://www.youtube.com/clip/UgkxAbCdEf12"
        ) == ("video", "UgkxAbCdEf12")

    def test_embed_url(self):
        assert _detect_youtube_url(
            "https://www.youtube.com/embed/jNQXAC9IVRw"
        ) == ("video", "jNQXAC9IVRw")

    def test_mobile_url(self):
        assert _detect_youtube_url(
            "https://m.youtube.com/watch?v=jNQXAC9IVRw"
        ) == ("video", "jNQXAC9IVRw")

    def test_handle_channel(self):
        assert _detect_youtube_url(
            "https://www.youtube.com/@MarquesBrownlee"
        ) == ("channel", "@MarquesBrownlee")

    def test_channel_id(self):
        result = _detect_youtube_url(
            "https://www.youtube.com/channel/UCBJycsmduvYEL83R_U4JriQ"
        )
        assert result == ("channel", "UCBJycsmduvYEL83R_U4JriQ")

    def test_legacy_user(self):
        assert _detect_youtube_url(
            "https://www.youtube.com/user/Computerphile"
        ) == ("channel", "Computerphile")

    def test_vanity_channel(self):
        assert _detect_youtube_url(
            "https://www.youtube.com/c/Veritasium"
        ) == ("channel", "Veritasium")

    def test_playlist(self):
        result = _detect_youtube_url(
            "https://www.youtube.com/playlist?list=PLrAXtmRdnEQy6nuLMt9H1Pj7RgqZTjB"
        )
        assert result is not None
        assert result[0] == "playlist"

    def test_music_excluded(self):
        # music.youtube.com is deferred to a sibling tool; detection must
        # surface it as 'music' kind so the dispatcher can emit a clear error.
        result = _detect_youtube_url("https://music.youtube.com/watch?v=jNQXAC9IVRw")
        assert result is not None
        assert result[0] == "music"

    def test_non_youtube_url(self):
        assert _detect_youtube_url("https://example.com/watch?v=jNQXAC9IVRw") is None
        assert _detect_youtube_url("https://vimeo.com/123456") is None
        assert _detect_youtube_url("https://twitch.tv/somestreamer") is None


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

class TestFormatHelpers:
    def test_duration_under_minute(self):
        assert _format_duration(19) == "0:19"

    def test_duration_under_hour(self):
        assert _format_duration(125) == "2:05"

    def test_duration_over_hour(self):
        assert _format_duration(3725) == "1:02:05"

    def test_duration_zero(self):
        assert _format_duration(0) == "0:00"

    def test_duration_none(self):
        assert _format_duration(None) is None

    def test_duration_float(self):
        # yt-dlp returns floats; we truncate to whole seconds
        assert _format_duration(125.7) == "2:05"

    def test_upload_date(self):
        assert _format_upload_date("20050423") == "2005-04-23"

    def test_upload_date_invalid_passthrough(self):
        # Non-8-digit strings pass through unchanged so callers can decide
        assert _format_upload_date("notadate") == "notadate"

    def test_upload_date_none(self):
        assert _format_upload_date(None) is None

    def test_captions_summary_manual_only(self):
        info = {"subtitles": {"en": [], "fr": []}, "automatic_captions": {}}
        langs, auto_only = _captions_summary(info)
        assert langs == ["en", "fr"]
        assert auto_only is False

    def test_captions_summary_auto_only(self):
        info = {"subtitles": {}, "automatic_captions": {"en": []}}
        langs, auto_only = _captions_summary(info)
        assert langs == ["en"]
        assert auto_only is True

    def test_captions_summary_mixed(self):
        info = {
            "subtitles": {"en": []},
            "automatic_captions": {"en": [], "fr": []},
        }
        langs, auto_only = _captions_summary(info)
        assert langs == ["en", "fr"]
        # Mixed = manual exists; not auto-only
        assert auto_only is False

    def test_captions_summary_none(self):
        langs, auto_only = _captions_summary({})
        assert langs == []
        assert auto_only is False


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

class TestErrorMapping:
    def test_bot_detection(self):
        from yt_dlp.utils import ExtractorError  # type: ignore[import-not-found]
        err = ExtractorError("Sign in to confirm you're not a bot.")
        result = _map_yt_dlp_error(err)
        assert "bot" in result.lower()
        assert "residential" in result.lower()

    def test_private_video(self):
        from yt_dlp.utils import ExtractorError  # type: ignore[import-not-found]
        err = ExtractorError("Private video")
        result = _map_yt_dlp_error(err)
        assert "private" in result.lower()

    def test_video_unavailable(self):
        from yt_dlp.utils import ExtractorError  # type: ignore[import-not-found]
        err = ExtractorError("Video unavailable")
        result = _map_yt_dlp_error(err)
        assert "unavailable" in result.lower()

    def test_geo_restricted(self):
        from yt_dlp.utils import GeoRestrictedError  # type: ignore[import-not-found]
        err = GeoRestrictedError("Not available in your country")
        result = _map_yt_dlp_error(err)
        assert "geo" in result.lower()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TestDispatcher:
    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result = await youtube(action="unknown")
        assert result.startswith("Error:")
        assert "video" in result  # lists valid actions

    @pytest.mark.asyncio
    async def test_video_missing_url(self):
        result = await youtube(action="video")
        assert "Error:" in result
        assert "url" in result.lower()

    @pytest.mark.asyncio
    async def test_video_non_youtube_url(self):
        result = await youtube(
            action="video", url="https://example.com/foo",
        )
        assert "Error:" in result
        assert "recognized" in result.lower() or "youtube" in result.lower()

    @pytest.mark.asyncio
    async def test_video_with_channel_url(self):
        result = await youtube(
            action="video", url="https://www.youtube.com/@MarquesBrownlee",
        )
        assert "Error:" in result
        assert "channel" in result.lower()

    @pytest.mark.asyncio
    async def test_video_with_music_url(self):
        result = await youtube(
            action="video",
            url="https://music.youtube.com/watch?v=jNQXAC9IVRw",
        )
        assert "Error:" in result
        assert "music" in result.lower()


# ---------------------------------------------------------------------------
# _video action with mocked yt-dlp
# ---------------------------------------------------------------------------

# Modeled after yt-dlp's actual output for "Me at the zoo" (jNQXAC9IVRw),
# trimmed to the fields _video reads.
_SAMPLE_INFO = {
    "id": "jNQXAC9IVRw",
    "title": "Me at the zoo",
    "description": "The first video on YouTube. Maybe ever.",
    "channel": "jawed",
    "uploader": "jawed",
    "channel_id": "UC4QobU6STFB0P71PMvOGN5A",
    "channel_url": "https://www.youtube.com/channel/UC4QobU6STFB0P71PMvOGN5A",
    "duration": 19.0,
    "upload_date": "20050423",
    "view_count": 365129877,
    "like_count": 10728475,
    "language": "en",
    "live_status": "not_live",
    "availability": "public",
    "subtitles": {"en": [{"ext": "vtt"}]},
    "automatic_captions": {"en": [{"ext": "vtt"}], "fr": [{"ext": "vtt"}]},
}


class _FakeYoutubeDL:
    """Minimal stand-in for yt_dlp.YoutubeDL covering what _video calls."""

    def __init__(self, payload):
        # payload may be a dict (for extract_info to return) or an exception
        # to raise on extract_info.
        self._payload = payload

    def extract_info(self, *args, **kwargs):
        del args, kwargs
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload

    @staticmethod
    def sanitize_info(info):
        # The real implementation strips non-JSON-safe values; our fixture
        # is already JSON-safe so passthrough is correct.
        return info


class TestVideoAction:
    @pytest.mark.asyncio
    async def test_metadata_and_description(self, monkeypatch):
        monkeypatch.setattr(
            _yt_module, "_get_ydl_video",
            lambda: _FakeYoutubeDL(_SAMPLE_INFO),
        )
        result = await youtube(
            action="video",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )

        # Frontmatter fields
        assert "title: Me at the zoo" in result
        assert "video_id: jNQXAC9IVRw" in result
        assert "channel: jawed" in result
        assert "duration: 0:19" in result
        assert "upload_date: 2005-04-23" in result
        assert "view_count: 365129877" in result
        assert "language: en" in result
        # Description in body
        assert "The first video on YouTube" in result
        # Trust advisory (content fence) is present
        assert "untrusted content" in result

    @pytest.mark.asyncio
    async def test_no_description_fallback(self, monkeypatch):
        info = dict(_SAMPLE_INFO)
        info["description"] = ""
        monkeypatch.setattr(
            _yt_module, "_get_ydl_video",
            lambda: _FakeYoutubeDL(info),
        )
        result = await youtube(
            action="video",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        assert "no description" in result.lower()

    @pytest.mark.asyncio
    async def test_yt_dlp_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            _yt_module, "_get_ydl_video",
            lambda: _FakeYoutubeDL(None),
        )
        result = await youtube(
            action="video",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        assert "Error:" in result
        assert "no metadata" in result.lower()

    @pytest.mark.asyncio
    async def test_bot_detection_propagates(self, monkeypatch):
        from yt_dlp.utils import ExtractorError  # type: ignore[import-not-found]
        exc = ExtractorError("Sign in to confirm you're not a bot.")
        monkeypatch.setattr(
            _yt_module, "_get_ydl_video",
            lambda: _FakeYoutubeDL(exc),
        )
        result = await youtube(
            action="video",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        assert "Error:" in result
        assert "bot" in result.lower()

    @pytest.mark.asyncio
    async def test_captions_auto_only_flag(self, monkeypatch):
        info = dict(_SAMPLE_INFO)
        info["subtitles"] = {}
        info["automatic_captions"] = {"en": [{"ext": "vtt"}]}
        monkeypatch.setattr(
            _yt_module, "_get_ydl_video",
            lambda: _FakeYoutubeDL(info),
        )
        result = await youtube(
            action="video",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        assert "captions_auto_only: True" in result


# ---------------------------------------------------------------------------
# Transcript: helpers
# ---------------------------------------------------------------------------

class TestTranscriptHelpers:
    def test_mmss_under_minute(self):
        assert _mmss(19) == "00:19"

    def test_mmss_under_hour(self):
        assert _mmss(125) == "02:05"

    def test_mmss_over_hour(self):
        assert _mmss(3725) == "01:02:05"

    def test_punctuation_density_punctuated(self):
        segs = [
            Segment(0, 2, "Hello there. How are you?"),
            Segment(2, 2, "I am well. Thanks!"),
        ]
        d = _punctuation_density(segs)
        # 4 enders, 9 words → ~0.44
        assert d > 0.3

    def test_punctuation_density_unpunctuated(self):
        segs = [
            Segment(0, 2, "all right so here we are in front of"),
            Segment(2, 2, "the elephants the cool thing about these"),
        ]
        d = _punctuation_density(segs)
        assert d == 0.0

    def test_punctuation_density_empty(self):
        assert _punctuation_density([]) == 0.0

    def test_segment_ends_sentence_period(self):
        assert _segment_ends_sentence(Segment(0, 1, "Hello world.")) is True

    def test_segment_ends_sentence_no(self):
        assert _segment_ends_sentence(Segment(0, 1, "Hello world")) is False

    def test_segment_ends_sentence_trailing_whitespace(self):
        # Trailing whitespace shouldn't fool the detector
        assert _segment_ends_sentence(Segment(0, 1, "Hello world. ")) is True

    def test_segment_ends_sentence_empty(self):
        assert _segment_ends_sentence(Segment(0, 1, "")) is False


# ---------------------------------------------------------------------------
# Outlier gap detection
# ---------------------------------------------------------------------------

class TestOutlierGaps:
    def test_empty(self):
        assert _detect_outlier_gaps([]) == []

    def test_single(self):
        assert _detect_outlier_gaps([Segment(0, 1, "a")]) == [False]

    def test_short_with_outlier_uses_fallback(self):
        # 3 segments → 2 gaps. Below the rolling-window threshold, so the
        # 3.0s fixed fallback applies.
        segs = [
            Segment(0, 1, "a"),     # ends at 1
            Segment(2, 1, "b"),     # gap=1.0 — under fallback
            Segment(8, 1, "c"),     # gap=5.0 — over fallback
        ]
        out = _detect_outlier_gaps(segs)
        assert out == [False, True, False]

    def test_long_with_outlier_uses_rolling(self):
        # 12 segments at steady 1s gaps + one ~5s outlier gap. The rolling
        # median is ~1.0, threshold = max(2*1, 1.5) = 2.0; the 5s gap
        # crosses, the 1s gaps don't.
        segs: list[Segment] = []
        t = 0.0
        for i in range(11):
            segs.append(Segment(t, 1.0, f"seg{i}"))
            t = t + 1.0 + 1.0  # 1s segment + 1s gap
        # Inject outlier: bump next segment's start so gap is ~5s
        segs.append(Segment(t + 4.0, 1.0, "outlier"))
        out = _detect_outlier_gaps(segs)
        # Last in-gap position before outlier should flag
        assert out[-2] is True
        # Steady-cadence positions should not flag
        assert all(o is False for o in out[:-2])
        assert out[-1] is False

    def test_floor_blocks_tiny_outliers(self):
        # All gaps <0.5s; nothing should flag even though some are 4× the median
        segs = [
            Segment(0, 0.1, "a"),   # ends 0.1
            Segment(0.2, 0.1, "b"), # gap 0.1
            Segment(0.3, 0.1, "c"), # gap 0.0
            Segment(0.5, 0.1, "d"), # gap 0.1 — but tiny, under 1.5 floor
        ]
        out = _detect_outlier_gaps(segs)
        assert all(o is False for o in out)


# ---------------------------------------------------------------------------
# Window coalescer
# ---------------------------------------------------------------------------

def _make_segments(spec: list[tuple[float, float, str]]) -> list[Segment]:
    """Helper: build segments from (start, duration, text) tuples."""
    return [Segment(s, d, t) for s, d, t in spec]


class TestCoalesceWindows:
    def test_empty(self):
        assert coalesce_windows([], sentence_aware=False) == []

    def test_short_input_one_window(self):
        # Total duration < min — everything goes in one window
        segs = _make_segments([
            (0, 2, "a"),
            (2, 2, "b"),
            (4, 2, "c"),
        ])
        windows = coalesce_windows(segs, sentence_aware=False)
        assert len(windows) == 1
        assert windows[0].start == 0
        assert windows[0].end == 6
        assert len(windows[0].segments) == 3

    def test_max_duration_forces_cut(self):
        # 8 segments × 5s each = 40s total, max is 35s. Should cut.
        segs = _make_segments([(i * 5.0, 5.0, f"s{i}") for i in range(8)])
        windows = coalesce_windows(segs, sentence_aware=False)
        assert len(windows) >= 2
        # All windows must respect max duration
        for w in windows:
            assert (w.end - w.start) <= 35.0 + 0.01  # tiny float slack

    def test_pause_boundary_triggers_cut_in_band(self):
        # 6 segments × 5s reaching 30s, then a 3s pause, then more segments.
        # The pause should cut once we're in the [25, 35] band.
        segs = _make_segments([
            (0, 5, "a"),
            (5, 5, "b"),
            (10, 5, "c"),
            (15, 5, "d"),
            (20, 5, "e"),
            (25, 5, "f"),  # ends at 30
            # 3s gap
            (33, 5, "g"),
            (38, 5, "h"),
        ])
        windows = coalesce_windows(segs, sentence_aware=False)
        # Expect the cut at the pause
        assert len(windows) == 2
        assert windows[0].segments[-1].text == "f"
        assert windows[1].segments[0].text == "g"

    def test_no_pause_runs_to_max(self):
        # No pauses at all → cut at max
        segs = _make_segments([(i * 4.0, 4.0, f"s{i}") for i in range(15)])
        windows = coalesce_windows(segs, sentence_aware=False)
        for w in windows:
            assert (w.end - w.start) <= 35.0 + 0.01

    def test_sentence_aware_cuts_at_period(self):
        # 5 segments × 6s. After the third, the text ends with a period.
        # In sentence-aware mode, that should cut even without a pause.
        segs = _make_segments([
            (0, 6, "first"),     # ends 6
            (6, 6, "second"),    # ends 12
            (12, 6, "third."),   # ends 18, but in band? 18 < 25 — no cut yet
            (18, 6, "fourth"),   # ends 24, still < 25 — no cut yet
            (24, 6, "fifth."),   # ends 30 — IN band, sentence end → cut
            (30, 6, "sixth"),    # next window
        ])
        windows = coalesce_windows(segs, sentence_aware=True)
        assert len(windows) == 2
        assert windows[0].segments[-1].text == "fifth."
        assert windows[1].segments[0].text == "sixth"

    def test_sentence_aware_off_uses_pause_only(self):
        # Same input but sentence_aware=False — the sentence break shouldn't
        # cut; we'd need a pause boundary to cut. With contiguous timing,
        # that means everything stays in one window (or hits max).
        segs = _make_segments([
            (0, 6, "first"),
            (6, 6, "second."),
            (12, 6, "third"),
            (18, 6, "fourth."),
        ])
        windows = coalesce_windows(segs, sentence_aware=False)
        # No pauses, no max breach (24s) → one window
        assert len(windows) == 1


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

class TestRenderTranscript:
    @staticmethod
    def _windows() -> list[Window]:
        # Modeled after "Me at the zoo"
        segs = [
            Segment(1.20, 2.16, "All right, so here we are, in front of the elephants"),
            Segment(5.32, 2.66, "the cool thing about these guys is that they have really..."),
            Segment(7.97, 4.64, "really really long trunks"),
            Segment(12.62, 1.75, "and that's cool"),
            Segment(14.42, 1.31, "(baaaaaaaaaaahhh!!)"),
            Segment(16.88, 2.00, "and that's pretty much all there is to say"),
        ]
        return [Window(start=segs[0].start, end=segs[-1].end, segments=tuple(segs))]

    def test_none_mode_flat_text(self):
        out = render_transcript(self._windows(), mode="none")
        assert "[" not in out
        assert "All right, so here we are" in out
        assert "all there is to say" in out

    def test_absolute_mode_per_line_timestamps(self):
        out = render_transcript(self._windows(), mode="absolute")
        assert "[00:01]" in out
        assert "[00:05]" in out
        assert "[00:16]" in out
        assert "All right, so here we are" in out

    def test_compact_mode_single_window_anchor(self):
        out = render_transcript(self._windows(), mode="compact")
        # Window anchor present
        assert "[00:01]" in out
        # Each segment on its own line
        assert "All right, so here we are, in front of the elephants" in out
        assert "(baaaaaaaaaaahhh!!)" in out
        # Compact mode emits one anchor (window start), not per-line
        assert out.count("[00:") == 1 or out.count("[00:") <= 2

    def test_compact_mode_multi_window_anchors(self):
        # Two windows with a clear gap between
        segs1 = [Segment(i * 5.0, 5.0, f"win1 seg{i}") for i in range(6)]
        segs2 = [Segment(35.0 + i * 5.0, 5.0, f"win2 seg{i}") for i in range(4)]
        windows = [
            Window(0, 30, tuple(segs1)),
            Window(35, 55, tuple(segs2)),
        ]
        out = render_transcript(windows, mode="compact")
        assert "[00:00]" in out
        assert "[00:35]" in out

    def test_structured_mode_yaml(self):
        out = render_transcript(self._windows(), mode="structured")
        # Should be parseable YAML
        import yaml
        data = yaml.safe_load(out)
        assert isinstance(data, list)
        assert len(data) == 6
        assert data[0]["t"] == 1.2
        assert "elephants" in data[0]["text"]

    def test_empty_windows(self):
        assert render_transcript([], mode="compact") == ""
        assert render_transcript([], mode="none") == ""
        assert render_transcript([], mode="absolute") == ""


# ---------------------------------------------------------------------------
# Transcript error mapping
# ---------------------------------------------------------------------------

class TestTranscriptErrorMapping:
    def test_ip_blocked(self):
        from youtube_transcript_api import IpBlocked
        err = IpBlocked("vid")
        out = _map_transcript_error(err)
        assert "IP" in out and "residential proxy" in out.lower()

    def test_request_blocked(self):
        from youtube_transcript_api import RequestBlocked
        err = RequestBlocked("vid")
        out = _map_transcript_error(err)
        assert "bot" in out.lower()

    def test_po_token_required(self):
        from youtube_transcript_api import PoTokenRequired
        err = PoTokenRequired("vid")
        out = _map_transcript_error(err)
        assert "PoToken" in out

    def test_transcripts_disabled(self):
        from youtube_transcript_api import TranscriptsDisabled
        err = TranscriptsDisabled("vid")
        out = _map_transcript_error(err)
        assert "disabled" in out.lower()

    def test_no_transcript_found(self):
        from youtube_transcript_api import NoTranscriptFound
        # NoTranscriptFound has a specific signature; pass minimal args
        err = NoTranscriptFound("vid", ["en"], None)
        out = _map_transcript_error(err)
        assert "no transcript" in out.lower()

    def test_video_unavailable(self):
        from youtube_transcript_api import VideoUnavailable
        err = VideoUnavailable("vid")
        out = _map_transcript_error(err)
        assert "unavailable" in out.lower()


# ---------------------------------------------------------------------------
# _transcript action
# ---------------------------------------------------------------------------

class _FakeFetchedTranscript:
    """Stand-in for youtube-transcript-api's FetchedTranscript."""

    def __init__(self, snippets, language_code="en", is_generated=False):
        self.snippets = snippets
        self.language = "English"
        self.language_code = language_code
        self.is_generated = is_generated
        self.video_id = "fake"


class _FakeSnippet:
    """Stand-in for FetchedTranscriptSnippet (simple value object)."""
    def __init__(self, start, duration, text):
        self.start = start
        self.duration = duration
        self.text = text


# Modeled on "Me at the zoo" — manual captions, punctuated.
_ZOO_SNIPPETS = [
    _FakeSnippet(1.20, 2.16, "All right, so here we are, in front of the elephants"),
    _FakeSnippet(5.32, 2.66, "the cool thing about these guys is that they have really..."),
    _FakeSnippet(7.97, 4.64, "really really long trunks"),
    _FakeSnippet(12.62, 1.75, "and that's cool"),
    _FakeSnippet(14.42, 1.31, "(baaaaaaaaaaahhh!!)"),
    _FakeSnippet(16.88, 2.00, "and that's pretty much all there is to say"),
]

# Auto-caption shape: lowercase, no punctuation.
_AUTO_SNIPPETS = [
    _FakeSnippet(0.0, 3.0, "all right so here we are in front of the"),
    _FakeSnippet(3.0, 3.0, "elephants the cool thing about these guys is"),
    _FakeSnippet(6.0, 3.0, "that they have really really really long trunks"),
    _FakeSnippet(9.0, 3.0, "and that's cool"),
]


class TestTranscriptAction:
    @pytest.mark.asyncio
    async def test_punctuated_returns_compact_default(self, monkeypatch):
        def fake_fetch(video_id, languages):
            del video_id, languages
            return _FakeFetchedTranscript(_ZOO_SNIPPETS, "en", is_generated=False)
        monkeypatch.setattr(_yt_module, "_fetch_transcript_sync", fake_fetch)

        result = await youtube(
            action="transcript",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        assert "transcript_kind: manual" in result
        assert "transcript_language: en" in result
        assert "All right, so here we are, in front of the elephants" in result
        assert "untrusted content" in result

    @pytest.mark.asyncio
    async def test_auto_caption_uses_time_window(self, monkeypatch):
        def fake_fetch(video_id, languages):
            del video_id, languages
            return _FakeFetchedTranscript(_AUTO_SNIPPETS, "en", is_generated=True)
        monkeypatch.setattr(_yt_module, "_fetch_transcript_sync", fake_fetch)

        result = await youtube(
            action="transcript",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        assert "transcript_kind: auto" in result
        assert "chunking_strategy: time_window" in result

    @pytest.mark.asyncio
    async def test_punctuated_uses_sentence_strategy(self, monkeypatch):
        def fake_fetch(video_id, languages):
            del video_id, languages
            return _FakeFetchedTranscript(_ZOO_SNIPPETS, "en", is_generated=False)
        monkeypatch.setattr(_yt_module, "_fetch_transcript_sync", fake_fetch)

        result = await youtube(
            action="transcript",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        # Density of _ZOO_SNIPPETS is high (commas/periods/exclam) so sentence-aware
        assert "chunking_strategy: sentence" in result

    @pytest.mark.asyncio
    async def test_timestamps_absolute_mode(self, monkeypatch):
        def fake_fetch(video_id, languages):
            del video_id, languages
            return _FakeFetchedTranscript(_ZOO_SNIPPETS, "en", is_generated=False)
        monkeypatch.setattr(_yt_module, "_fetch_transcript_sync", fake_fetch)

        result = await youtube(
            action="transcript",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
            timestamps="absolute",
        )
        # Absolute mode emits a per-line [MM:SS]
        assert "[00:01]" in result
        assert "[00:05]" in result

    @pytest.mark.asyncio
    async def test_timestamps_none_mode(self, monkeypatch):
        def fake_fetch(video_id, languages):
            del video_id, languages
            return _FakeFetchedTranscript(_ZOO_SNIPPETS, "en", is_generated=False)
        monkeypatch.setattr(_yt_module, "_fetch_transcript_sync", fake_fetch)

        result = await youtube(
            action="transcript",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
            timestamps="none",
        )
        # No bracketed timestamps in the body
        body_start = result.index("\n\n") + 2
        body = result[body_start:]
        # `[` only appears in fence markers and (baaaaa...!!) lines
        # Specifically, no [00:NN] timestamps
        import re as _re
        assert _re.search(r"\[\d+:\d+\]", body) is None

    @pytest.mark.asyncio
    async def test_no_url(self):
        result = await youtube(action="transcript")
        assert "Error" in result and "url" in result.lower()

    @pytest.mark.asyncio
    async def test_non_youtube_url(self):
        result = await youtube(
            action="transcript",
            url="https://example.com/video",
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_channel_url_rejected(self):
        result = await youtube(
            action="transcript",
            url="https://www.youtube.com/@somechan",
        )
        assert "Error" in result and "channel" in result.lower()

    @pytest.mark.asyncio
    async def test_transcripts_disabled_propagates(self, monkeypatch):
        from youtube_transcript_api import TranscriptsDisabled

        def fake_fetch(video_id, languages):
            del languages
            raise TranscriptsDisabled(video_id)
        monkeypatch.setattr(_yt_module, "_fetch_transcript_sync", fake_fetch)

        result = await youtube(
            action="transcript",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        assert "Error" in result
        assert "disabled" in result.lower()

    @pytest.mark.asyncio
    async def test_request_blocked_propagates(self, monkeypatch):
        from youtube_transcript_api import RequestBlocked

        def fake_fetch(video_id, languages):
            del languages
            raise RequestBlocked(video_id)
        monkeypatch.setattr(_yt_module, "_fetch_transcript_sync", fake_fetch)

        result = await youtube(
            action="transcript",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        assert "Error" in result
        assert "bot" in result.lower()

    @pytest.mark.asyncio
    async def test_normalizes_embedded_newlines(self, monkeypatch):
        # YouTube caption cues frequently contain embedded \n for display
        # wrapping. Each segment must render as one coherent line.
        snippets = [
            _FakeSnippet(0.0, 2.0, "First line of\ncaption"),
            _FakeSnippet(2.0, 2.0, "Second  \n  line\nhere"),
        ]

        def fake_fetch(video_id, languages):
            del video_id, languages
            return _FakeFetchedTranscript(snippets, "en", is_generated=False)
        monkeypatch.setattr(_yt_module, "_fetch_transcript_sync", fake_fetch)

        result = await youtube(
            action="transcript",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        assert "First line of caption" in result
        assert "Second line here" in result

    @pytest.mark.asyncio
    async def test_empty_snippets_handled(self, monkeypatch):
        def fake_fetch(video_id, languages):
            del video_id, languages
            return _FakeFetchedTranscript([], "en", is_generated=False)
        monkeypatch.setattr(_yt_module, "_fetch_transcript_sync", fake_fetch)

        result = await youtube(
            action="transcript",
            url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        assert "Error" in result
        assert "no segments" in result.lower()
