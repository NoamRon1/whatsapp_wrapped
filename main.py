from flask import Flask, request, render_template, Response, jsonify
import re
import json
import time
import unicodedata
import zipfile
import tempfile
from pathlib import Path
from typing import Optional
import statistics
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import io
import os

# try import wordcloud; if missing, endpoint will return a helpful JSON error
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

# try import bidi for correct RTL shaping/display (optional)
try:
    from bidi.algorithm import get_display
except Exception:
    get_display = None

app = Flask(__name__)

def _safe_filename(name: str) -> str:
    # normalize and replace unsafe chars (Windows-friendly)
    name = unicodedata.normalize("NFKD", name)
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)

def find_first_txt(extract_dir: Path) -> Optional[Path]:
    for p in extract_dir.rglob("*.txt"):
        return p
    return None

def parse_whatsapp_txt(path: Path):
    """
    Tolerant WhatsApp text parser:
    - Detects new messages when a line contains ' - ' and the left part starts with a digit (date/time).
    - Splits sender and message on the first ': ' when present.
    - Appends continuation lines to the previous message (multi-line support).
    - Keeps raw datetime string for later parsing.
    """
    messages = []
    last = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if not line:
                if last is not None:
                    last["message"] += "\n"
                continue

            if " - " in line:
                left, right = line.split(" - ", 1)
                if re.match(r"^\d", left.strip()):
                    if last is not None:
                        messages.append(last)
                    if ": " in right:
                        sender, msg = right.split(": ", 1)
                    else:
                        sender, msg = None, right
                    last = {"datetime": left.strip(), "sender": sender, "message": msg}
                    continue

            # continuation line
            if last is not None:
                last["message"] += "\n" + line
            else:
                # orphan/system line before any timestamped message
                last = {"datetime": None, "sender": None, "message": line}

    if last is not None:
        messages.append(last)
    return messages

# new helper: try parse common whatsapp datetime formats
def _try_parse_datetime(s: str):
    if not s:
        return None
    s = s.strip()
    # try to remove trailing timezone brackets or extra text
    s = s.split(" (")[0].strip()
    # common patterns WhatsApp exports use
    patterns = [
        "%d/%m/%Y, %H:%M:%S",
        "%d/%m/%Y, %H:%M",
        "%d/%m/%y, %H:%M:%S",
        "%d/%m/%y, %H:%M",
        "%m/%d/%Y, %I:%M %p",
        "%m/%d/%y, %I:%M %p",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d, %H:%M:%S",
        "%d.%m.%Y, %H:%M",
    ]
    for fmt in patterns:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # last-ditch: try to extract digits and parse with iso-like
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

# new helper: load all messages from chats/*.json and merge them
def _load_all_messages():
    chats_dir = Path("chats")
    msgs = []
    if not chats_dir.exists():
        return msgs
    for p in sorted(chats_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for m in data.get("messages", []):
                dt_raw = m.get("datetime")
                dt_parsed = _try_parse_datetime(dt_raw)
                msgs.append({
                    "datetime_raw": dt_raw,
                    "datetime": dt_parsed.isoformat() if dt_parsed else None,
                    "_dt_obj": dt_parsed,
                    "sender": m.get("sender"),
                    "message": m.get("message"),
                    "source_file": data.get("source_file"),
                })
        except Exception:
            # skip corrupt files
            continue
    # sort by parsed datetime when available, otherwise leave at end (None sorts last)
    msgs.sort(key=lambda x: (x["_dt_obj"] is None, x["_dt_obj"] or datetime.max))
    return msgs

# New modular stat functions ---------------------------------------------------

def compute_chats_activity(msgs, user):
    """
    Compute: total_messages_sent, total_messages_received, chats_participated,
    total_active_days, avg_messages_per_day, longest_message_sent,
    shortest_message_sent, most_active_hour, most_active_day_of_week.
    """
    chats = defaultdict(list)
    for m in msgs:
        chat = m.get("source_file") or "unknown"
        chats[chat].append(m)

    total_messages_sent = 0
    total_messages_received = 0
    chats_participated = 0
    dates_user_active = set()
    sent_texts = []
    per_hour = Counter()
    per_weekday = Counter()

    for chat, items in chats.items():
        user_sent = 0
        user_received = 0
        for m in items:
            s = m.get("sender")
            dt = m.get("_dt_obj")
            text = (m.get("message") or "") or ""
            if s == user:
                total_messages_sent += 1
                user_sent += 1
                sent_texts.append(text)
                if dt:
                    dates_user_active.add(dt.date())
                    per_hour[dt.hour] += 1
                    per_weekday[dt.weekday()] += 1
            elif s:
                total_messages_received += 1
                user_received += 1
        if user_sent > 0 or user_received > 0:
            chats_participated += 1

    total_active_days = len(dates_user_active)
    avg_messages_per_day = (total_messages_sent / total_active_days) if total_active_days else 0
    longest_message_sent = max(sent_texts, key=lambda s: len(s), default=None)
    shortest_message_sent = min([s for s in sent_texts if s and s.strip()], key=lambda s: len(s), default=None)
    most_active_hour = per_hour.most_common(1)[0][0] if per_hour else None
    most_active_day_of_week = per_weekday.most_common(1)[0][0] if per_weekday else None

    return {
        "total_messages_sent": total_messages_sent,
        "total_messages_received": total_messages_received,
        "total_chats_participated_in": chats_participated,
        "total_active_days": total_active_days,
        "avg_messages_per_day": avg_messages_per_day,
        "longest_message_sent": longest_message_sent,
        "shortest_message_sent": shortest_message_sent,
        "most_active_hour": most_active_hour,
        "most_active_day_of_week": most_active_day_of_week,
    }

def compute_top_senders(msgs, user, top_n=20):
    counter_from_others = Counter()
    for m in msgs:
        s = m.get("sender")
        if s and s != user:
            counter_from_others[s] += 1
    return [{"name": n, "count": c} for n, c in counter_from_others.most_common(top_n)]

def compute_reply_stats(msgs, user, max_reply_seconds=90*60):
    """
    For each message by someone != user, find the next message by user and:
    - if the delta is >0 and <= max_reply_seconds, count it as a reply.
    Returns overall average, per-partner reply lists, fastest partner, sample count.
    """
    reply_times_by_partner = defaultdict(list)
    overall_reply_deltas = []

    for i, m in enumerate(msgs):
        s = m.get("sender")
        dt = m.get("_dt_obj")
        if not s or s == user or not dt:
            continue
        for j in range(i+1, len(msgs)):
            m2 = msgs[j]
            dt2 = m2.get("_dt_obj")
            if not dt2:
                continue
            if m2.get("sender") == user:
                delta = (dt2 - dt).total_seconds()
                if delta > 0 and delta <= max_reply_seconds:
                    reply_times_by_partner[s].append(delta)
                    overall_reply_deltas.append(delta)
                break

    fastest_partner = None
    if reply_times_by_partner:
        stats = []
        for partner, deltas in reply_times_by_partner.items():
            avg = statistics.mean(deltas)
            stats.append((partner, avg, len(deltas)))
        stats.sort(key=lambda x: (x[1], -x[2]))
        partner, avg_sec, samples = stats[0]
        fastest_partner = {"name": partner, "avg_seconds": avg_sec, "samples": samples}

    overall_avg = statistics.mean(overall_reply_deltas) if overall_reply_deltas else None

    return {
        "overall_avg_reply_seconds": overall_avg,
        "fastest_reply_partner": fastest_partner,
        "reply_samples": len(overall_reply_deltas),
        "reply_threshold_seconds": max_reply_seconds,
        "reply_times_by_partner": {k: len(v) for k, v in reply_times_by_partner.items()},
    }

def compute_emoji_stats(msgs, user, top_n=5):
    """
    Count emojis in messages sent by `user`.
    Returns dict with most_used (emoji/count), top_n list and total count.
    """
    counter = Counter()
    total = 0
    for m in msgs:
        if m.get("sender") != user:
            continue
        text = m.get("message") or ""
        emojis = _extract_emojis(text)
        if not emojis:
            continue
        for e in emojis:
            counter[e] += 1
            total += 1

    top = [{"emoji": e, "count": c} for e, c in counter.most_common(top_n)]
    most_used = {"emoji": top[0]["emoji"], "count": top[0]["count"]} if top else None
    return {"most_used": most_used, "top_5": top, "total_emojis": total}

_duration_hms_re = re.compile(r'(?:(\d+):)?([0-5]?\d):([0-5]\d)')  # H:MM:SS or M:SS as group(1) optional

def _parse_duration_to_seconds(s: str) -> int:
    """
    Try to parse durations like '1:23' or '01:02:30' from a string.
    Returns seconds or 0 if not found.
    """
    if not s:
        return 0
    m = _duration_hms_re.search(s)
    if not m:
        return 0
    if m.group(1) is None:
        # mm:ss
        minutes = int(m.group(2))
        seconds = int(m.group(3))
        return minutes * 60 + seconds
    else:
        hours = int(m.group(1))
        minutes = int(m.group(2))
        seconds = int(m.group(3))
        return hours * 3600 + minutes * 60 + seconds

def compute_voice_stats(msgs, user):
    """
    Heuristics:
    - Count messages that likely represent voice notes (keywords).
    - Sum any durations that appear in message text (mm:ss or hh:mm:ss).
    Returns counts and total seconds.
    """
    voice_keywords = ["voice message", "voice note", "audio omitted", "<audio omitted>", "ptt", "ptt."]  # heuristic
    voice_count = 0
    total_seconds = 0
    for m in msgs:
        if m.get("sender") != user:
            continue
        text = (m.get("message") or "").lower()
        found_keyword = any(k in text for k in voice_keywords)
        # try to parse explicit durations seen in text
        secs = _parse_duration_to_seconds(text)
        if found_keyword:
            voice_count += 1
        if secs:
            total_seconds += secs
    return {"voice_messages_count": voice_count, "voice_total_seconds": total_seconds}

def _parse_years_param(param_values):
    """
    Accepts either:
      - None -> returns empty set (meaning no filtering)
      - a string like "2020,2021"
      - a list like ['2020','2021']
    Returns set[int] of years (may be empty).
    """
    if not param_values:
        return set()
    years = set()
    if isinstance(param_values, str):
        parts = [p.strip() for p in param_values.split(",") if p.strip()]
    else:
        # list from request.args.getlist
        parts = []
        for v in param_values:
            parts.extend([p.strip() for p in str(v).split(",") if p.strip()])
    for p in parts:
        try:
            y = int(p)
            if 1900 <= y <= 3000:
                years.add(y)
        except Exception:
            continue
    return years

@app.route("/api/years", methods=["GET"])
def api_years():
    msgs = _load_all_messages()
    years = set()
    for m in msgs:
        dt = m.get("_dt_obj")
        if dt:
            years.add(dt.year)
    years_list = sorted(years, reverse=True)
    return Response(json.dumps({"years": years_list}, ensure_ascii=False), mimetype="application/json; charset=utf-8")

# Revised api_stats: orchestrate small functions --------------------------------
@app.route("/api/stats", methods=["GET"])
def api_stats():
    user = request.args.get("user")
    if not user:
        return jsonify({"error": "missing user parameter"}), 400

    msgs = _load_all_messages()

    # parse years param: support ?years=2020,2021 or repeated ?years=2020&years=2021
    years_param = None
    if request.args.getlist("years"):
        years_param = request.args.getlist("years")
    elif request.args.get("years"):
        years_param = request.args.get("years")
    years_filter = _parse_years_param(years_param)

    if years_filter:
        msgs = [m for m in msgs if m.get("_dt_obj") and m["_dt_obj"].year in years_filter]

    # activity & message stats
    activity = compute_chats_activity(msgs, user)

    # top senders
    top_senders = compute_top_senders(msgs, user, top_n=20)

    # reply stats (90-minute threshold) - keep overall average and samples but remove fastest partner
    reply = compute_reply_stats(msgs, user, max_reply_seconds=90*60)

    # emoji & voice stats
    emoji_stats = compute_emoji_stats(msgs, user)
    voice_stats = compute_voice_stats(msgs, user)

    # time/streak stats
    time_stats = compute_time_stats(msgs, user)

    # new: word / deleted / group stats
    word_stats = compute_word_stats(msgs, user, top_n=50)
    del_edit_stats = compute_deleted_edited_stats(msgs, user)
    group_stats = compute_group_stats(msgs, user)

    result = {
        "user": user,
        "messages_total": len(msgs),
        "filtered_years": sorted(list(years_filter)) if years_filter else None,
        # activity fields
        **activity,
        # contacts & replies
        "top_senders": top_senders,
        # removed fastest_reply_partner by request
        "overall_avg_reply_seconds": reply["overall_avg_reply_seconds"],
        "reply_threshold_seconds": reply["reply_threshold_seconds"],
        "reply_samples": reply["reply_samples"],
        # emoji/voice
        "emoji_stats": emoji_stats,
        "voice_stats": voice_stats,
        # time/streaks
        **time_stats,
        # new extras
        "word_stats": word_stats,
        **del_edit_stats,
        **group_stats,
    }
    return Response(json.dumps(result, ensure_ascii=False, default=str, indent=2), mimetype="application/json; charset=utf-8")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["GET"])
def analyze_page():
    return render_template("analyze.html")

@app.route("/upload", methods=["POST"])
def upload():
    uploaded = request.files.get("file")
    if not uploaded:
        return jsonify({"error": "no file uploaded"}), 400
    filename = uploaded.filename or "upload.zip"
    if not filename.lower().endswith(".zip"):
        return jsonify({"error": "please upload a .zip file"}), 400

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        zip_path = tmpdir / "upload.zip"
        uploaded.save(str(zip_path))

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmpdir)
        except zipfile.BadZipFile:
            return jsonify({"error": "bad zip file"}), 400

        txt_path = find_first_txt(tmpdir)
        if not txt_path:
            return jsonify({"error": "no .txt chat file found inside zip"}), 400

        messages = parse_whatsapp_txt(txt_path)

        # prepare output directory `chats`
        chats_dir = Path("chats")
        chats_dir.mkdir(parents=True, exist_ok=True)

        # safe filename with timestamp prefix
        base_name = _safe_filename(txt_path.stem)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_filename = f"{timestamp}_{base_name}.json"
        out_path = chats_dir / out_filename

        # build payload and write using ensure_ascii=False to preserve Hebrew and emojis
        payload = {"source_file": str(txt_path.name), "messages_count": len(messages), "messages": messages, "out_filename": out_filename}
        json_text = json.dumps(payload, ensure_ascii=False, indent=2)

        out_path.write_text(json_text, encoding="utf-8")

        return Response(json_text, mimetype="application/json; charset=utf-8")

@app.route("/api/usernames", methods=["GET"])
def api_usernames():
    msgs = _load_all_messages()
    names = [m["sender"] for m in msgs if m.get("sender")]
    uniq = sorted(set(names), key=lambda s: s.lower())
    return Response(json.dumps({"usernames": uniq}, ensure_ascii=False), mimetype="application/json; charset=utf-8")

# small helper to extract emojis (approximate)
def _extract_emojis(text: str):
    if not text:
        return []
    # emoji ranges (approximate coverage)
    emoji_re = re.compile(
        "[" 
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F700-\U0001F77F"  # alchemical
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA00-\U0001FA6F"
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U00002700-\U000027BF"
        "]+", flags=re.UNICODE)
    return emoji_re.findall(text)

def compute_time_stats(msgs, user):
    """
    Compute time-based stats for messages sent by `user`.
    Returns a dict with:
      - late_night_messages_count (hours 0-3)
      - early_bird_messages_count (hours 4-6)
      - longest_daily_streak: {"streak": int, "start": iso, "end": iso} or None
      - busiest_single_day: {"date": iso, "count": int} or None
      - busiest_single_hour: {"hour": int, "count": int} or None
      - week_most_active: {"week": "YYYY-WW", "count": int} or None
      - month_with_most_messages: {"month": "YYYY-MM", "count": int} or None
    """
    from collections import Counter
    per_date = Counter()
    per_hour = Counter()
    per_week = Counter()
    per_month = Counter()
    user_dates = set()

    for m in msgs:
        if m.get("sender") != user:
            continue
        dt = m.get("_dt_obj")
        if not dt:
            continue
        date = dt.date()
        per_date[date] += 1
        per_hour[dt.hour] += 1
        iso_year, iso_week, _ = dt.isocalendar()
        per_week[(iso_year, iso_week)] += 1
        per_month[(dt.year, dt.month)] += 1
        user_dates.add(date)

    # late night: hours 0-3 (after midnight)
    late_night_count = sum(cnt for h, cnt in per_hour.items() if 0 <= h <= 3)
    # early bird: hours 4-6 (4:00 - 6:59)
    early_bird_count = sum(cnt for h, cnt in per_hour.items() if 4 <= h < 7)

    # helper for longest consecutive days streak
    def longest_consecutive(dates_set):
        if not dates_set:
            return 0, None, None
        dlist = sorted(dates_set)
        best = cur = 1
        best_start = cur_start = dlist[0]
        for a, b in zip(dlist, dlist[1:]):
            if (b - a).days == 1:
                cur += 1
            else:
                if cur > best:
                    best = cur
                    best_start = cur_start
                cur = 1
                cur_start = b
        if cur > best:
            best = cur
            best_start = cur_start
        best_end = best_start + timedelta(days=best-1) if best_start else None
        return best, best_start, best_end

    streak_len, streak_start, streak_end = longest_consecutive(user_dates)
    longest_streak = {"streak": streak_len, "start": streak_start.isoformat() if streak_start else None, "end": streak_end.isoformat() if streak_end else None} if streak_len else None

    busiest_single_day = None
    if per_date:
        d, c = per_date.most_common(1)[0]
        busiest_single_day = {"date": d.isoformat(), "count": c}

    busiest_single_hour = None
    if per_hour:
        h, c = per_hour.most_common(1)[0]
        busiest_single_hour = {"hour": h, "count": c}

    week_most_active = None
    if per_week:
        (y,w), c = max(per_week.items(), key=lambda kv: kv[1])
        week_most_active = {"week": f"{y}-W{w:02d}", "count": c}

    month_with_most_messages = None
    if per_month:
        (y,m), c = max(per_month.items(), key=lambda kv: kv[1])
        month_with_most_messages = {"month": f"{y}-{m:02d}", "count": c}

    return {
        "late_night_messages_count": late_night_count,
        "early_bird_messages_count": early_bird_count,
        "longest_daily_streak": longest_streak,
        "busiest_single_day": busiest_single_day,
        "busiest_single_hour": busiest_single_hour,
        "week_most_active": week_most_active,
        "month_with_most_messages": month_with_most_messages,
    }

# New: compute word-cloud / most used words
def compute_word_stats(msgs, user, top_n=50):
    """
    Returns top words used by `user`: list of {"word": w, "count": c} and total_words.
    Tokenization supports basic Hebrew/Latin; excludes placeholders like media/message/deleted/edited,
    and tokens inside angle brackets.
    """
    # allow letters (incl. Hebrew), numbers and apostrophes
    word_re = re.compile(r"[A-Za-z\u00C0-\u017F\u05D0-\u05EA0-9']+", flags=re.UNICODE)
    stopwords = {
        # small english stopword set -- extend as needed
        "the","and","to","of","in","it","you","i","that","for","on","with","was","is","are","a","an","this","be","have","has","at","by","from","or"
    }
    # stronger blacklist: any token that mentions media/message/omitted/deleted/edited or is like <...>
    placeholder_substrings = ("omitted", "deleted", "edited", "media", "message", "audio", "image", "sticker")
    counts = Counter()
    total = 0
    for m in msgs:
        if m.get("sender") != user:
            continue
        txt = (m.get("message") or "") or ""
        # skip obvious whole-placeholder messages quickly
        t_low = txt.strip().lower()
        if (t_low.startswith("<") and t_low.endswith(">")) or any(sub in t_low for sub in placeholder_substrings):
            continue
        for w in word_re.findall(txt):
            wlow = w.lower()
            # skip stopwords, pure numbers, very short tokens
            if wlow in stopwords or re.fullmatch(r"\d+", wlow) or len(wlow) <= 1:
                continue
            # skip placeholder-like tokens
            if any(sub in wlow for sub in placeholder_substrings):
                continue
            # skip tokens that look like "<...>"
            if wlow.startswith("<") and wlow.endswith(">"):
                continue
            counts[wlow] += 1
            total += 1
    top = [{"word": w, "count": c} for w, c in counts.most_common(top_n)]
    return {"top_words": top, "total_words": total}

# New: compute deleted / edited messages heuristics
def compute_deleted_edited_stats(msgs, user):
    """
    Heuristic counts of deleted and edited messages sent by user.
    """
    deleted_keywords = [
        "this message was deleted", "message deleted", "<deleted message>",
        "הודעה נמחקה", "הודעה הוסרה", "deleted"
    ]
    edited_keywords = ["(edited)", "edited", "נערך", "ערוך"]
    deleted = 0
    edited = 0
    for m in msgs:
        if m.get("sender") != user:
            continue
        txt = (m.get("message") or "").lower()
        if any(k in txt for k in deleted_keywords):
            deleted += 1
        if any(k in txt for k in edited_keywords):
            edited += 1
    return {"messages_deleted": deleted, "messages_edited": edited}

# New: group-related stats
def compute_group_stats(msgs, user):
    """
    Compute:
      - group_most_messages: chat with highest messages sent by user
      - group_most_emojis: chat with most emojis overall (or by user if preferred)
      - group_longest_silence_broken_by_user: chat where user's message followed the longest gap
    """
    per_chat_user_msgs = defaultdict(int)
    per_chat_emoji_count = defaultdict(int)
    per_chat_items = defaultdict(list)

    for m in msgs:
        chat = m.get("source_file") or "unknown"
        per_chat_items[chat].append(m)
        if m.get("sender") == user:
            per_chat_user_msgs[chat] += 1
        # count emojis in any message for chat
        emojis = _extract_emojis(m.get("message") or "")
        if emojis:
            per_chat_emoji_count[chat] += len(emojis)

    # group most messages (by user)
    group_most_messages = None
    if per_chat_user_msgs:
        chat, cnt = max(per_chat_user_msgs.items(), key=lambda kv: kv[1])
        group_most_messages = {"chat": chat, "count": cnt}

    # group most emojis
    group_most_emojis = None
    if per_chat_emoji_count:
        chat, cnt = max(per_chat_emoji_count.items(), key=lambda kv: kv[1])
        group_most_emojis = {"chat": chat, "count": cnt}

    # longest silence broken by you
    group_longest_silence = None
    for chat, items in per_chat_items.items():
        # sort by datetime
        items_sorted = [it for it in items if it.get("_dt_obj")]
        if not items_sorted:
            continue
        items_sorted.sort(key=lambda x: x["_dt_obj"])
        max_gap = 0
        max_gap_dt = None
        for i in range(1, len(items_sorted)):
            prev = items_sorted[i-1]
            cur = items_sorted[i]
            if cur.get("sender") != user:
                continue
            if not prev.get("_dt_obj") or not cur.get("_dt_obj"):
                continue
            gap = (cur["_dt_obj"] - prev["_dt_obj"]).total_seconds()
            if gap > max_gap:
                max_gap = gap
                max_gap_dt = (prev["_dt_obj"], cur["_dt_obj"])
        if max_gap > 0:
            if not group_longest_silence or max_gap > group_longest_silence["seconds"]:
                group_longest_silence = {"chat": chat, "seconds": max_gap, "start": max_gap_dt[0].isoformat() if max_gap_dt else None, "end": max_gap_dt[1].isoformat() if max_gap_dt else None}

    return {
        "group_most_messages": group_most_messages,
        "group_most_emojis": group_most_emojis,
        "group_longest_silence_broken_by_user": group_longest_silence,
    }

def _word_freq_for_user(msgs, user):
    """
    Build a Counter of word -> count for messages sent by `user`.
    Uses same tokenization / filtering rules as compute_word_stats but returns the full Counter.
    """
    word_re = re.compile(r"[A-Za-z\u00C0-\u017F\u05D0-\u05EA0-9']+", flags=re.UNICODE)
    placeholder_substrings = ("omitted", "deleted", "edited", "media", "message", "audio", "image", "sticker")
    stopwords = {
        "the","and","to","of","in","it","you","i","that","for","on","with","was","is","are","a","an","this","be","have","has","at","by","from","or"
    }
    counts = Counter()
    for m in msgs:
        if m.get("sender") != user:
            continue
        txt = (m.get("message") or "") or ""
        t_low = txt.strip().lower()
        # skip whole-placeholder messages
        if (t_low.startswith("<") and t_low.endswith(">")) or any(sub in t_low for sub in placeholder_substrings):
            continue
        for w in word_re.findall(txt):
            wlow = w.lower()
            if wlow in stopwords or re.fullmatch(r"\d+", wlow) or len(wlow) <= 1:
                continue
            if any(sub in wlow for sub in placeholder_substrings):
                continue
            if wlow.startswith("<") and wlow.endswith(">"):
                continue
            counts[wlow] += 1
    return counts

def _find_hebrew_font():
    """
    Try to find a system TTF that likely contains Hebrew glyphs.
    Returns a font path or None.
    """
    candidates = [
        # common Windows fonts
        r"C:\Windows\Fonts\NotoSansHebrew-Regular.ttf",
        r"C:\Windows\Fonts\NotoSansHebrew-Roboto-Regular.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
        r"C:\Windows\Fonts\SegoeUI.ttf",
        r"C:\Windows\Fonts\Tahoma.ttf",
        r"C:\Windows\Fonts\DejaVuSans.ttf",
        # fallback generic names in Fonts folder
    ]
    # also scan Fonts folder for names containing 'Heb', 'Noto', 'Arial', 'DejaVu'
    fonts_dir = r"C:\Windows\Fonts"
    try:
        if os.path.isdir(fonts_dir):
            for fn in os.listdir(fonts_dir):
                low = fn.lower()
                if any(k in low for k in ("hebrew", "hebrew", "noto", "dejavu", "arial", "tahoma", "segoe", "taff")):
                    candidates.append(os.path.join(fonts_dir, fn))
    except Exception:
        pass

    for p in candidates:
        try:
            if p and os.path.isfile(p):
                return p
        except Exception:
            continue
    return None

def _prepare_freqs_for_wordcloud(freqs: Counter):
    """
    For languages like Hebrew (RTL), transform word keys using bidi.get_display when available.
    Returns a new dict suitable for WordCloud.generate_from_frequencies.
    """
    out = {}
    for w, c in freqs.items():
        key = w
        # if contains Hebrew letters, try bidi transform for visual rendering
        if re.search(r'[\u0590-\u05FF]', w) and get_display:
            try:
                key = get_display(w)
            except Exception:
                key = w
        out[key] = out.get(key, 0) + c
    return out

@app.route("/api/wordcloud.png", methods=["GET"])
def api_wordcloud_png():
    """
    Returns a PNG word cloud image for ?user=...&years=... (years same format as /api/stats)
    Now attempts to use a font containing Hebrew glyphs and bidi reshaping if available.
    """
    if WordCloud is None:
        return jsonify({"error": "server missing Python package 'wordcloud' or its dependencies"}), 500

    user = request.args.get("user")
    if not user:
        return jsonify({"error": "missing user parameter"}), 400

    # load and optionally filter by years (reuse existing helper)
    msgs = _load_all_messages()
    years_param = None
    if request.args.getlist("years"):
        years_param = request.args.getlist("years")
    elif request.args.get("years"):
        years_param = request.args.get("years")
    years_filter = _parse_years_param(years_param)
    if years_filter:
        msgs = [m for m in msgs if m.get("_dt_obj") and m["_dt_obj"].year in years_filter]

    freqs = _word_freq_for_user(msgs, user)
    if not freqs:
        # no words to render
        return jsonify({"error": "no word frequencies found for this user / filter"}), 204

    # transform frequencies for RTL words if necessary
    freqs_for_wc = _prepare_freqs_for_wordcloud(freqs)

    # try to find a font that supports Hebrew; WordCloud will fail silently or render squares without it
    font_path = _find_hebrew_font()

    try:
        wc_kwargs = {"width": 900, "height": 360, "background_color": "white", "prefer_horizontal": 0.9}
        if font_path:
            wc_kwargs["font_path"] = font_path
        wc = WordCloud(**wc_kwargs)
        img = wc.generate_from_frequencies(freqs_for_wc).to_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return Response(buf.getvalue(), mimetype="image/png")
    except Exception as e:
        # helpful debug message; return JSON describing the issue
        return jsonify({
            "error": "failed to generate word cloud",
            "detail": str(e),
            "note": "ensure 'wordcloud' + Pillow are installed and a font with Hebrew glyphs is available on the server (set font_path or place a TTF in C:\\Windows\\Fonts).",
            "font_used": font_path
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
