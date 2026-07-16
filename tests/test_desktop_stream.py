"""Phase 6C — client-side SSE parsing into sanitized ActivityVMs."""

from desktop import stream
from desktop.models import ActivityVM


def _lines(text):
    # emulate httpx.iter_lines(): newline-stripped lines
    return text.split("\n")


def test_parse_single_frame():
    frames = list(stream.parse_sse_lines(_lines(
        'event: activity\ndata: {"domain":"email","status":"ok"}\n\n')))
    assert frames == [("activity", {"domain": "email", "status": "ok"})]


def test_parse_multiple_frames_and_skips_heartbeats():
    text = (': keep-alive\n\n'
            'event: hello\ndata: {"x":1}\n\n'
            'event: activity\ndata: {"domain":"knowledge"}\n\n')
    frames = list(stream.parse_sse_lines(_lines(text)))
    assert ("hello", {"x": 1}) in frames
    assert ("activity", {"domain": "knowledge"}) in frames


def test_parse_multiline_data_concatenates():
    frames = list(stream.parse_sse_lines(_lines('data: {"a":\ndata: 1}\n\n')))
    assert frames == [("message", {"a": 1})]


def test_malformed_data_is_skipped_not_raised():
    frames = list(stream.parse_sse_lines(_lines('event: activity\ndata: not json\n\n')))
    assert frames == []


def test_events_from_lines_only_activity_and_sanitized():
    text = ('event: hello\ndata: {"title":"connected"}\n\n'
            'event: activity\ndata: {"domain":"web","event_type":"read","status":"ok",'
            '"severity":"info","title":"<script>x</script>"}\n\n')
    events = list(stream.events_from_lines(_lines(text)))
    assert len(events) == 1                       # the hello frame is not an activity
    ev = events[0]
    assert isinstance(ev, ActivityVM)
    assert "<script>" not in ev.title             # re-sanitized client-side


def test_trailing_frame_without_blank_line_flushes():
    frames = list(stream.parse_sse_lines(_lines('event: activity\ndata: {"domain":"x"}')))
    assert frames == [("activity", {"domain": "x"})]


def test_stream_client_refuses_remote_base():
    import pytest
    with pytest.raises(ValueError):
        stream.StreamClient("http://evil.example.com", "tok")


def test_stream_client_runs_injected_source_until_stop():
    text = ('event: activity\ndata: {"domain":"a","event_type":"e","status":"ok"}\n\n'
            'event: activity\ndata: {"domain":"b","event_type":"e","status":"ok"}\n\n')
    sc = stream.StreamClient("http://127.0.0.1:8000", "tok", line_source=_lines(text))
    seen = []
    sc.run(seen.append)
    assert [e.domain for e in seen] == ["a", "b"]
