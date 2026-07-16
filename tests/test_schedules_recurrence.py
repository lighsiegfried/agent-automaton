"""Deterministic date/time normalization, timezone + DST, recurrence (Phase 5C)."""

from datetime import datetime, timezone

from app.core import errors
from app.schedules import recurrence

TZ = "America/Guatemala"                       # UTC-6, no DST
NY = "America/New_York"                         # DST-observing
NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)   # 06:00 local Guatemala (Wed)


def _p(text, tz=TZ, now=NOW):
    return recurrence.parse_when(text, tz_name=tz, now_utc=now)


# --- relative + absolute one-time --------------------------------------------------


def test_relative_tomorrow_at_time():
    p = _p("llamar a Ana mañana a las nueve")
    assert p.ok and p.trigger_type == "one_time"
    assert p.run_at_utc == "2026-07-16T15:00:00+00:00"      # 09:00 Guatemala
    assert p.remainder == "llamar a Ana".lower() or "ana" in p.remainder.lower()


def test_relative_minutes():
    p = _p("recordar algo en 2 minutos")
    assert p.ok and p.run_at_utc == "2026-07-15T12:02:00+00:00"


def test_time_only_rolls_to_next_occurrence():
    # 03:00 local is already past 06:00 now → tomorrow.
    p = _p("a las 3 de la tarde")
    assert p.ok and p.run_at_utc == "2026-07-15T21:00:00+00:00"   # 15:00 today (ahead of 06:00)


def test_absolute_date():
    p = _p("el 20 de julio a las 8")
    assert p.ok and p.run_at_utc == "2026-07-20T14:00:00+00:00"   # 08:00 Guatemala


def test_english_relative():
    p = _p("call the team tomorrow at 9")
    assert p.ok and p.run_at_utc == "2026-07-16T15:00:00+00:00"


# --- ambiguity + errors ------------------------------------------------------------


def test_bare_weekday_is_ambiguous():
    assert _p("el viernes").code == errors.AMBIGUOUS_DATE


def test_date_without_time_is_ambiguous():
    assert _p("el 20 de julio").code == errors.AMBIGUOUS_DATE


def test_no_date_is_invalid():
    assert _p("algo sin fecha").code == errors.INVALID_DATE


def test_invalid_timezone():
    assert _p("mañana a las 9", tz="Mars/Phobos").code == errors.INVALID_TIMEZONE


# --- recurrence --------------------------------------------------------------------


def test_recurring_weekly():
    p = _p("cada lunes busca novedades del proyecto")
    assert p.ok and p.trigger_type == "recurring"
    assert p.recurrence == {"freq": "weekly", "weekday": 0, "at": "09:00"}
    assert p.run_at_utc == "2026-07-20T15:00:00+00:00"           # next Monday 09:00
    assert "novedades" in p.remainder


def test_recurring_daily_with_time():
    p = _p("todos los días a las 7 revisa el correo")
    assert p.recurrence == {"freq": "daily", "at": "07:00"}


def test_recurring_interval():
    p = _p("cada 2 minutos")
    assert p.recurrence == {"freq": "interval", "seconds": 120}


def test_compute_next_run_daily_and_weekly():
    tz = recurrence.resolve_timezone(TZ)
    nxt = recurrence.compute_next_run({"freq": "daily", "at": "09:00"}, NOW, tz)
    assert nxt.astimezone(tz).hour == 9
    wk = recurrence.compute_next_run({"freq": "weekly", "weekday": 0, "at": "09:00"}, NOW, tz)
    assert wk.astimezone(tz).weekday() == 0


# --- DST ---------------------------------------------------------------------------


def test_dst_spring_forward_gap_pushed_past():
    tz = recurrence.resolve_timezone(NY)
    # 2026-03-08: 02:00 -> 03:00 EDT; 02:30 does not exist.
    start = datetime(2026, 3, 8, 5, 0, tzinfo=timezone.utc)      # ~00:00 EST
    nxt = recurrence.compute_next_run({"freq": "daily", "at": "02:30"}, start, tz)
    local = nxt.astimezone(tz)
    assert (local.hour, local.minute) == (3, 30)                # advanced past the gap
    assert local.tzname() == "EDT"


def test_dst_offset_changes_across_year():
    tz = recurrence.resolve_timezone(NY)
    jan = datetime(2026, 1, 15, 12, tzinfo=tz).utcoffset()
    jul = datetime(2026, 7, 15, 12, tzinfo=tz).utcoffset()
    assert jan != jul                                           # EST vs EDT


def test_format_local():
    assert "09:00" in recurrence.format_local("2026-07-16T15:00:00+00:00", TZ)
    assert TZ in recurrence.format_local("2026-07-16T15:00:00+00:00", TZ)
