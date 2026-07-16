"""Deterministic date/time normalization + recurrence (Phase 5C, item 3).

Turns a Spanish/English relative or absolute expression ("mañana a las nueve",
"cada lunes", "en 2 minutos", "tomorrow at 9") into an absolute UTC instant plus an
optional recurrence rule, resolving through a configured IANA timezone with EXPLICIT
daylight-saving handling. Ambiguous expressions (a bare weekday with no time) are
refused rather than guessed. Pure and stdlib-only (zoneinfo), so the whole thing is
unit-testable with an injected "now".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core import errors

# 0 = Monday … 6 = Sunday (matches datetime.weekday()).
_WEEKDAYS = {
    "lunes": 0, "monday": 0, "martes": 1, "tuesday": 1,
    "miercoles": 2, "miércoles": 2, "wednesday": 2, "jueves": 3, "thursday": 3,
    "viernes": 4, "friday": 4, "sabado": 5, "sábado": 5, "saturday": 5,
    "domingo": 6, "sunday": 6,
}
_HOUR_WORDS = {
    "una": 1, "uno": 1, "one": 1, "dos": 2, "two": 2, "tres": 3, "three": 3,
    "cuatro": 4, "four": 4, "cinco": 5, "five": 5, "seis": 6, "six": 6,
    "siete": 7, "seven": 7, "ocho": 8, "eight": 8, "nueve": 9, "nine": 9,
    "diez": 10, "ten": 10, "once": 11, "eleven": 11, "doce": 12, "twelve": 12,
    "mediodia": 12, "mediodía": 12, "noon": 12, "medianoche": 0, "midnight": 0,
}
_MONTHS = {
    "enero": 1, "january": 1, "febrero": 2, "february": 2, "marzo": 3, "march": 3,
    "abril": 4, "april": 4, "mayo": 5, "may": 5, "junio": 6, "june": 6,
    "julio": 7, "july": 7, "agosto": 8, "august": 8, "septiembre": 9, "setiembre": 9,
    "september": 9, "octubre": 10, "october": 10, "noviembre": 11, "november": 11,
    "diciembre": 12, "december": 12,
}
_DEFAULT_TIME = time(9, 0)     # for recurring rules stated without a time


@dataclass
class ParsedWhen:
    ok: bool
    trigger_type: str = ""            # "one_time" | "recurring"
    run_at_utc: str = ""
    recurrence: dict | None = None
    local_display: str = ""
    remainder: str = ""               # the non-temporal text (title/action)
    code: str | None = None
    reason: str = ""


def resolve_timezone(name: str):
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None


# --- DST-aware localization --------------------------------------------------------

def localize(naive_local: datetime, tz: ZoneInfo) -> datetime:
    """Attach ``tz`` to a naive local datetime, handling DST gaps and folds
    explicitly. A nonexistent local time (spring-forward gap) is pushed forward past
    the gap; an ambiguous time (fall-back fold) resolves to the FIRST occurrence."""
    aware = naive_local.replace(tzinfo=tz, fold=0)
    # Gap detection: inside a spring-forward gap, utcoffset(fold=0) != utcoffset(fold=1)
    # AND converting round-trips to a different wall time.
    alt = naive_local.replace(tzinfo=tz, fold=1)
    if aware.utcoffset() != alt.utcoffset():
        # Round-trip through UTC; if the wall clock shifted, we were in the gap.
        rt = aware.astimezone(timezone.utc).astimezone(tz)
        if rt.replace(tzinfo=None) != naive_local:
            # Nonexistent time — advance by the offset difference (past the gap).
            shift = alt.utcoffset() - aware.utcoffset()
            return (naive_local + shift).replace(tzinfo=tz, fold=0)
    return aware


def to_utc_iso(aware: datetime) -> str:
    return aware.astimezone(timezone.utc).isoformat(timespec="seconds")


def format_local(utc_iso: str, tz_name: str) -> str:
    tz = resolve_timezone(tz_name)
    if tz is None:
        return utc_iso
    dt = datetime.fromisoformat(utc_iso).astimezone(tz)
    days = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    return f"{days[dt.weekday()]} {dt.day:02d}/{dt.month:02d}/{dt.year} {dt.hour:02d}:{dt.minute:02d} ({tz_name})"


# --- token parsers -----------------------------------------------------------------

_TIME_NUM = re.compile(r"\b(?:a\s+las?|at)\s+(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", re.I)
_TIME_AMPM = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", re.I)
_TIME_WORD = re.compile(r"\b(?:a\s+las?|at)\s+(" + "|".join(map(re.escape, _HOUR_WORDS)) + r")\b", re.I)
_MERIDIEM_ES = re.compile(r"\bde\s+la\s+(mañana|tarde|noche|madrugada)\b", re.I)


def _apply_meridiem(hour: int, token: str) -> int:
    t = (token or "").lower().replace(".", "")
    if t in ("pm", "tarde", "noche") and hour < 12:
        return hour + 12
    if t in ("am", "mañana", "madrugada") and hour == 12:
        return 0
    return hour


def _parse_time(text: str):
    """Return (time, span) or None. Recognizes 'a las 9', '9:30', 'nueve', '3pm',
    '3 de la tarde'."""
    meri = _MERIDIEM_ES.search(text)
    for pattern in (_TIME_NUM, _TIME_AMPM):
        m = pattern.search(text)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2) or 0)
            token = m.group(3) or (meri.group(1) if meri else "")
            hour = _apply_meridiem(hour, token)
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return time(hour, minute), (m.start(), m.end())
    m = _TIME_WORD.search(text)
    if m:
        hour = _HOUR_WORDS[m.group(1).lower()]
        token = meri.group(1) if meri else ""
        hour = _apply_meridiem(hour, token)
        return time(hour % 24, 0), (m.start(), m.end())
    return None


_REL_OFFSET = re.compile(
    r"\b(?:en|dentro\s+de|in)\s+(\d+)\s+(minutos?|mins?|horas?|hours?|d[ií]as?|days?)\b", re.I)
_REL_DAY = re.compile(r"\b(pasado\s+mañana|day\s+after\s+tomorrow|mañana|manana|tomorrow|hoy|today)\b", re.I)
_ABS_DMY = re.compile(r"\b(\d{1,2})\s+de\s+(" + "|".join(_MONTHS) + r")(?:\s+de\s+(\d{4}))?\b", re.I)
_ABS_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_WEEKDAY = re.compile(r"\b(" + "|".join(map(re.escape, _WEEKDAYS)) + r")\b", re.I)
_RECUR = re.compile(r"\b(cada|todos\s+los|todas\s+las|every|each)\b", re.I)
_RECUR_INTERVAL = re.compile(
    r"\b(?:cada|every)\s+(\d+)\s+(minutos?|mins?|horas?|hours?|d[ií]as?|days?)\b", re.I)


def _unit_seconds(unit: str) -> int:
    u = unit.lower()
    if u.startswith(("min", "mins")):
        return 60
    if u.startswith(("hora", "hour")):
        return 3600
    return 86400


def _strip(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            if 0 <= i < len(chars):
                chars[i] = " "
    out = "".join(chars)
    # Remove temporal filler left behind by the parsed tokens.
    out = re.sub(r"\bde\s+la\s+(mañana|tarde|noche|madrugada)\b", " ", out, flags=re.I)
    out = re.sub(r"\b(a\s+las?|d[ií]as?|days?)\b", " ", out, flags=re.I)
    out = " ".join(out.split())
    # Drop dangling connectives at the edges ("el", "de", "que", "to", "the").
    out = re.sub(r"\b(el|la|los|las|de|del|que|para|to|the|at)\b\s*$", "", out, flags=re.I).strip()
    out = re.sub(r"^\s*(que|to|the)\s+", "", out, flags=re.I).strip()
    return out


def parse_when(text: str, *, tz_name: str, now_utc: datetime) -> ParsedWhen:
    """Resolve a temporal expression. ``remainder`` is the text with the temporal
    tokens removed (the reminder title or task action)."""
    tz = resolve_timezone(tz_name)
    if tz is None:
        return ParsedWhen(False, code=errors.INVALID_TIMEZONE, reason=f"unknown timezone {tz_name!r}")
    now_local = now_utc.astimezone(tz)
    original = text or ""
    lowered = original.lower()
    spans: list[tuple[int, int]] = []

    tm = _parse_time(lowered)
    parsed_time = None
    if tm:
        parsed_time, span = tm
        spans.append(span)

    # --- recurrence -----------------------------------------------------------------
    if _RECUR.search(lowered):
        rule, r_spans, reason = _parse_recurrence(lowered, parsed_time)
        spans.extend(r_spans)
        if rule is None:
            return ParsedWhen(False, code=errors.INVALID_RECURRENCE, reason=reason,
                              remainder=_strip(original, spans))
        run_at = compute_next_run(rule, now_utc, tz)
        return ParsedWhen(True, trigger_type="recurring", run_at_utc=to_utc_iso(run_at),
                          recurrence=rule, local_display=format_local(to_utc_iso(run_at), tz_name),
                          remainder=_strip(original, spans))

    # --- one-time: relative offset --------------------------------------------------
    off = _REL_OFFSET.search(lowered)
    if off:
        spans.append((off.start(), off.end()))
        run_at = now_utc + timedelta(seconds=int(off.group(1)) * _unit_seconds(off.group(2)))
        return ParsedWhen(True, trigger_type="one_time", run_at_utc=to_utc_iso(run_at),
                          local_display=format_local(to_utc_iso(run_at), tz_name),
                          remainder=_strip(original, spans))

    # --- one-time: a specific day ---------------------------------------------------
    day, day_span, ambiguous = _parse_day(lowered, now_local)
    if day_span:
        spans.append(day_span)

    if day is not None:
        if parsed_time is None:
            return ParsedWhen(False, code=errors.AMBIGUOUS_DATE,
                              reason="that date needs a specific time (e.g. 'a las 9')",
                              remainder=_strip(original, spans))
        run_at = localize(datetime.combine(day, parsed_time), tz)
        return ParsedWhen(True, trigger_type="one_time", run_at_utc=to_utc_iso(run_at),
                          local_display=format_local(to_utc_iso(run_at), tz_name),
                          remainder=_strip(original, spans))

    if ambiguous:
        return ParsedWhen(False, code=errors.AMBIGUOUS_DATE,
                          reason="that weekday could mean more than one date — add a specific date/time",
                          remainder=_strip(original, spans))

    # --- one-time: a time only → today if still ahead, else tomorrow ---------------
    if parsed_time is not None:
        target = datetime.combine(now_local.date(), parsed_time)
        if target <= now_local.replace(tzinfo=None):
            target += timedelta(days=1)
        run_at = localize(target, tz)
        return ParsedWhen(True, trigger_type="one_time", run_at_utc=to_utc_iso(run_at),
                          local_display=format_local(to_utc_iso(run_at), tz_name),
                          remainder=_strip(original, spans))

    return ParsedWhen(False, code=errors.INVALID_DATE,
                      reason="I couldn't find a date or time in that",
                      remainder=_strip(original, spans))


def _parse_day(lowered: str, now_local: datetime):
    """Return (date | None, span | None, ambiguous: bool)."""
    m = _REL_DAY.search(lowered)
    if m:
        token = m.group(1).lower()
        base = now_local.date()
        if token in ("hoy", "today"):
            return base, (m.start(), m.end()), False
        if token in ("mañana", "manana", "tomorrow"):
            return base + timedelta(days=1), (m.start(), m.end()), False
        return base + timedelta(days=2), (m.start(), m.end()), False   # pasado mañana
    m = _ABS_ISO.search(lowered)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), (m.start(), m.end()), False
        except ValueError:
            return None, (m.start(), m.end()), False
    m = _ABS_DMY.search(lowered)
    if m:
        year = int(m.group(3)) if m.group(3) else now_local.year
        try:
            d = date(year, _MONTHS[m.group(2).lower()], int(m.group(1)))
        except ValueError:
            return None, (m.start(), m.end()), False
        if not m.group(3) and d < now_local.date():
            d = date(year + 1, d.month, d.day)
        return d, (m.start(), m.end()), False
    m = _WEEKDAY.search(lowered)
    if m:
        # A bare weekday for a ONE-TIME schedule is ambiguous (which week? no time).
        return None, (m.start(), m.end()), True
    return None, None, False


def _parse_recurrence(lowered: str, parsed_time):
    """Return (rule | None, spans, reason)."""
    at = (parsed_time or _DEFAULT_TIME).strftime("%H:%M")
    interval = _RECUR_INTERVAL.search(lowered)
    if interval:
        seconds = int(interval.group(1)) * _unit_seconds(interval.group(2))
        return {"freq": "interval", "seconds": seconds}, [(interval.start(), interval.end())], ""

    m = _RECUR.search(lowered)
    spans = [(m.start(), m.end())]
    tail = lowered[m.end():]
    wd = _WEEKDAY.search(tail)
    if wd:
        spans.append((m.end() + wd.start(), m.end() + wd.end()))
        return {"freq": "weekly", "weekday": _WEEKDAYS[wd.group(1).lower()], "at": at}, spans, ""
    dm = re.search(r"\b(d[ií]as?|days?)\b", tail)
    if dm:
        spans.append((m.end() + dm.start(), m.end() + dm.end()))
        return {"freq": "daily", "at": at}, spans, ""
    if re.search(r"\b(semana|week)\b", tail):
        return {"freq": "weekly", "weekday": 0, "at": at}, spans, ""
    if re.search(r"\b(mes|month)\b", tail):
        return {"freq": "monthly", "day": 1, "at": at}, spans, ""
    # "cada lunes" already handled; a bare "cada" with nothing → default daily.
    return {"freq": "daily", "at": at}, spans, ""


# --- recurrence computation --------------------------------------------------------

def _at_time(rule) -> time:
    hh, mm = (rule.get("at") or "09:00").split(":")
    return time(int(hh), int(mm))


def compute_next_run(rule: dict, after_utc: datetime, tz: ZoneInfo) -> datetime:
    """The next UTC instant strictly after ``after_utc`` for a recurrence rule,
    computed in local time (so DST is respected)."""
    if rule.get("freq") == "interval":
        return after_utc + timedelta(seconds=int(rule["seconds"]))

    after_local = after_utc.astimezone(tz)
    at = _at_time(rule)
    freq = rule.get("freq")

    if freq == "daily":
        candidate = datetime.combine(after_local.date(), at)
        while localize(candidate, tz) <= after_utc:
            candidate += timedelta(days=1)
        return localize(candidate, tz)

    if freq == "weekly":
        weekday = int(rule.get("weekday", 0))
        days_ahead = (weekday - after_local.weekday()) % 7
        candidate = datetime.combine(after_local.date() + timedelta(days=days_ahead), at)
        while localize(candidate, tz) <= after_utc:
            candidate += timedelta(days=7)
        return localize(candidate, tz)

    if freq == "monthly":
        day = min(int(rule.get("day", 1)), 28)
        y, mth = after_local.year, after_local.month
        candidate = datetime.combine(date(y, mth, day), at)
        while localize(candidate, tz) <= after_utc:
            mth += 1
            if mth > 12:
                mth, y = 1, y + 1
            candidate = datetime.combine(date(y, mth, day), at)
        return localize(candidate, tz)

    # Fallback: one day later.
    return after_utc + timedelta(days=1)
