"""Collect real HKJC race results into the canonical frame.

Where the data comes from
-------------------------
Two public HKJC pages, both fetched over plain HTTP:

``LocalResults.aspx``
    The race results page.  It honours ``RaceDate``/``RaceNo`` and serves
    **historical** meetings (verified back to at least 2016), which is why it
    is the backbone of the collector.  One request per race.

``/en-us/local/information/horse?horseid=...``
    The horse detail page.  Its *race history* table carries the **rating**
    at each past run, plus the horse's country, colour and sex.  One request
    per distinct horse.

The rating cannot be taken from the results page: the column the results page
calls ``Act. Wt.`` is the horse's handicap weight, not its rating, and it is
*not* a monotone function of the rating (a horse rated 39 and a horse rated
60 can both show 135).  So the rating is joined in from the horse history.

What is pre-race and what is not
--------------------------------
This matters more than anything else in the collector, so it is spelled out:

``win_odds`` on the results page is the **final (SP) price**, set after
betting closed.  Betting at the SP is the single easiest way to manufacture
an impressive backtest, so the SP is stored in ``win_odds_final`` and the
betting simulation is pointed at a price that actually existed before the
race.  ``win_odds_open`` is left empty rather than guessed.

The rating in the horse history is the rating that applied **at that run**,
i.e. the rating published on the racecard before the race.  That is the value
a pre-race model would have had.

Politeness and resumability
----------------------------
Requests are cached on disk (gzipped HTML keyed by URL), so an interrupted
run costs nothing to resume and re-parsing never re-downloads.  A small
delay between requests keeps the load on HKJC's servers negligible.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["HKJCCollector", "CollectConfig", "collect_hkjc"]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

RESULTS_URL = (
    "https://racing.hkjc.com/racing/information/english/Racing/LocalResults.aspx"
    "?RaceDate={date}&RaceNo={race_no}"
)
HORSE_URL = "https://racing.hkjc.com/en-us/local/information/horse?horseid={horse_id}"
VET_URL = "https://racing.hkjc.com/en-us/local/information/veterinaryrecord?horseid={horse_id}"

_HORSE_HREF = re.compile(r"/en-us/local/information/horse\?horseid=([A-Za-z0-9_]+)", re.I)
_CLASS_HEADER = re.compile(r"Class\s+([A-Za-z0-9]+)\s*-\s*(\d+)\s*M\s*-\s*\((\d+)-(\d+)\)")
_PRIZE = re.compile(r"HK\$\s*([\d,]+)")
_GOING = re.compile(r"Going\s*:\s*([A-Za-z0-9 /\-]+)")
_COURSE = re.compile(r"Course\s*:\s*([A-Za-z0-9 \"\-]+)")


@dataclass
class CollectConfig:
    """Knobs for a collection run."""

    start: str = "2018-09-01"
    end: str = "2025-12-31"
    target_races: int = 5000
    cache_dir: Path = Path("data/hkjc_cache")
    out_path: Path = Path("data/hkjc.csv")
    horses_path: Path = Path("data/hkjj_horses.csv")
    delay: float = 0.35
    workers: int = 4
    enrich_horses: bool = True
    enrich_veterinary: bool = True
    verbose: bool = True

@dataclass
class _Fetch:
    hits: int = 0
    misses: int = 0
    cached: int = 0
    errors: list[str] = field(default_factory=list)


class HKJCCollector:
    """Fetches and parses HKJC results into the canonical frame."""

    def __init__(self, config: CollectConfig | None = None) -> None:
        self.config = config or CollectConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = _Fetch()
        self._rate_lock = threading.Lock()
        self._last_request = 0.0

    def _throttle(self) -> None:
        """Global rate limit across all worker threads.

        The delay has to be global, not per-thread: four threads each
        sleeping 0.35s is still four requests every 0.35s.
        """
        interval = self.config.delay
        if interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = self._last_request + interval - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last_request = now

    # -- transport -----------------------------------------------------------------

    def _cache_path(self, url: str) -> Path:
        import hashlib

        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.config.cache_dir / f"{key}.html.gz"

    def fetch(self, url: str, *, force: bool = False) -> str | None:
        """Fetch a URL, using the on-disk cache when possible."""
        path = self._cache_path(url)
        if path.exists() and not force:
            self.stats.cached += 1
            try:
                return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
            except Exception:
                pass

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "gzip",
                "Referer": "https://racing.hkjc.com/",
            },
        )
        for attempt in range(3):
            try:
                self._throttle()
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    html = raw.decode("utf-8", errors="replace")
                path.write_bytes(gzip.compress(html.encode("utf-8")))
                self.stats.hits += 1
                return html
            except urllib.error.HTTPError as exc:
                if exc.code in (404, 403):
                    self.stats.misses += 1
                    return None
                self.stats.errors.append(f"{url} -> HTTP {exc.code}")
            except Exception as exc:
                self.stats.errors.append(f"{url} -> {type(exc).__name__}")
            time.sleep(1.5 * (attempt + 1))
        return None

    # -- parsing -------------------------------------------------------------------

    def parse_results(self, html: str, meeting_date: date, race_no: int) -> tuple[dict, list[dict]] | None:
        """Parse one results page into race metadata and runner rows."""
        if not html or "No information" in html:
            return None

        # The tables are parsed once and shared: reading them twice was costing
        # more than the download did.
        tables = pd.read_html(io.StringIO(html))
        results = None
        for t in tables:
            cols = [str(c) for c in t.columns]
            if "Pla." in cols and "Win Odds" in cols:
                results = t
                break
        if results is None:
            return None

        meta = self._parse_race_meta(html, meeting_date, race_no, tables)
        if meta is None:
            return None

        # The horse name is a link carrying the stable horse id.
        horse_ids = self._horse_ids_by_name(html)

        runners = []
        for _, row in results.iterrows():
            name = str(row["Horse"]).strip()
            # Names look like "KASA PAPA (G312)" ??split the brand code off.
            clean = re.sub(r"\s*\([A-Z]\d{3}\)\s*$", "", name).strip()
            runners.append(
                {
                    "horse_no": _to_int(row.get("Horse No.")),
                    "horse_name": clean,
                    "horse_id": horse_ids.get(clean),
                    "jockey_name": str(row.get("Jockey", "")).strip(),
                    "trainer_name": str(row.get("Trainer", "")).strip(),
                    "handicap_weight": _to_float(row.get("Act. Wt.")),
                    "declared_weight": _to_float(row.get("Declar. Horse Wt.")),
                    "draw": _to_int(row.get("Dr.")),
                    "beaten_margin": str(row.get("LBW", "")).strip(),
                    "running_position": str(row.get("Running Position", "")).strip(),
                    "finish_time": str(row.get("Finish Time", "")).strip(),
                    "win_odds_final": _to_float(row.get("Win Odds")),
                    "finish_position": _to_int(row.get("Pla.")),
                }
            )

        meta["num_runners"] = len(runners)
        return meta, runners

    def _parse_race_meta(
        self,
        html: str,
        meeting_date: date,
        race_no: int,
        tables: list[pd.DataFrame] | None = None,
    ) -> dict | None:
        """Read the race header.

        The header is a small table with a label/value shape:

            Class 5 - 1650M - (40-0) | Going : | GOOD
            D'AGUILAR HANDICAP       | Course : | TURF - "A" Course
            HK$ 875,000
        """
        if tables is None:
            tables = pd.read_html(io.StringIO(html))
        header = None
        for t in tables:
            flat = t.astype(str)
            joined = " ".join(str(v) for v in flat.to_numpy().ravel())
            if "Going :" in joined and "Course :" in joined:
                header = flat
                break
        if header is None:
            return None

        cells = [str(c).strip() for c in header.to_numpy().ravel() if str(c) not in {"nan", ""}]

        def value_after(label: str) -> str:
            """The cell following a label cell such as ``Going :``."""
            for i, value in enumerate(cells):
                if value.rstrip(":").strip() == label:
                    return cells[i + 1] if i + 1 < len(cells) else ""
            return ""

        def cell_before(label: str) -> str:
            for i, value in enumerate(cells):
                if value.rstrip(":").strip() == label and i > 0:
                    return cells[i - 1]
            return ""

        going = value_after("Going")
        course = value_after("Course")
        race_name = cell_before("Course")
        prize = None
        class_text = ""
        for value in cells:
            if value.startswith("HK$"):
                prize = float(value.replace("HK$", "").replace(",", "").strip())
            elif re.match(r"^Class\s+[A-Za-z0-9]+\s*-", value):
                class_text = value

        m = _CLASS_HEADER.search(class_text or " ".join(cells))
        if not m:
            return None
        race_class, distance, band_hi, band_lo = m.groups()

        venue = ""
        for t in tables:
            flat = t.astype(str)
            joined = " ".join(str(v) for v in flat.to_numpy().ravel())
            vm = re.search(r"(Happy Valley|Sha Tin)", joined)
            if vm:
                venue = vm.group(1)
                break

        going_code = _going_code(going)
        track_code = _track_code(course)

        return {
            "race_id": f"{meeting_date:%Y%m%d}-{race_no:02d}",
            "meeting_date": meeting_date,
            "venue": venue,
            "race_no": race_no,
            "race_class": race_class,
            "race_name": race_name,
            "distance_m": int(distance),
            "going": going,
            "going_code": going_code,
            "course": course,
            "track_code": track_code,
            "rating_band_low": int(band_lo),
            "rating_band_high": int(band_hi),
            "prize_money": prize,
        }

    def _horse_ids_by_name(self, html: str) -> dict[str, str]:
        """Map horse name -> stable horse id, taken from the name links."""
        import lxml.html

        try:
            doc = lxml.html.fromstring(html)
        except Exception:
            return {}
        mapping: dict[str, str] = {}
        for anchor in doc.xpath("//a[@href]"):
            href = anchor.get("href") or ""
            m = _HORSE_HREF.search(href)
            if not m:
                continue
            name = " ".join(anchor.text_content().split())
            name = re.sub(r"\s*\([A-Z]\d{3}\)\s*$", "", name).strip()
            if name:
                mapping[name] = m.group(1)
        return mapping

    def parse_horse(self, html: str) -> tuple[dict, list[dict]] | None:
        """Parse a horse detail page into horse metadata and its race history."""
        if not html or "No information" in html:
            return None
        tables = pd.read_html(io.StringIO(html))

        # The horse page's tables carry no <thead>: the header labels live in
        # the first *data* row.  The race history is the wide table whose first
        # row starts with "Race Index".
        history = None
        for t in tables:
            if t.shape[0] < 4 or t.shape[1] < 12:
                continue
            first_row = [str(v).strip() for v in t.iloc[0].tolist()]
            if "Race Index" in first_row:
                history = t.copy()
                history.columns = first_row
                history = history.iloc[1:].reset_index(drop=True)
                break
        if history is None:
            return None

        meta = {}
        for t in tables:
            if t.shape[1] != 3:
                continue
            for _, row in t.iterrows():
                label = str(row.iloc[0]).replace(":", "").strip()
                value = str(row.iloc[2]).strip()
                if not label or label == "nan":
                    continue
                if label.startswith("Country of Origin"):
                    # The label is literally "Country of Origin / Age" and the
                    # value is "NZ / 5" ??the country first, the age in years
                    # second.
                    parts = [p.strip() for p in value.split("/") if p.strip()]
                    meta["country"] = parts[0] if parts else ""
                    if len(parts) > 1:
                        meta["age"] = _to_float(parts[1])
                elif label.startswith("Colour / Sex"):
                    parts = [p.strip() for p in value.split("/") if p.strip()]
                    # "Chestnut / Gelding" ??the colour comes first, the sex
                    # last.  Taking the last part is robust to multi-word
                    # colours such as "Dark Bay".
                    meta["colour"] = parts[0] if parts else ""
                    meta["sex"] = parts[-1] if len(parts) > 1 else ""
                elif label.startswith("Import Type"):
                    meta["import_type"] = value
                elif label.startswith("Import Date"):
                    meta["import_date"] = value

        runs = []
        for _, row in history.iterrows():
            raw_date = str(row.get("Date", "")).strip()
            run_date = _parse_hk_date(raw_date)
            if run_date is None:
                continue
            runs.append(
                {
                    "horse_id": None,  # filled in by the caller
                    "run_date": run_date,
                    "race_index": str(row.get("Race Index", "")).strip(),
                    "place": _to_int(row.get("Pla.")),
                    "venue_course": str(row.get("RC/Track/ Course", "")).strip(),
                    "distance_m": _to_int(row.get("Dist.")),
                    "going": str(row.get("G", "")).strip(),
                    "race_class": str(row.get("Race Class", "")).strip(),
                    "draw": _to_int(row.get("Dr.")),
                    "rating": _to_int(row.get("Rtg.")),
                    "trainer_name": str(row.get("Trainer", "")).strip(),
                    "jockey_name": str(row.get("Jockey", "")).strip(),
                    "win_odds_final": _to_float(row.get("Win Odds")),
                    "beaten_margin": str(row.get("LBW", "")).strip(),
                }
            )
        return meta, runs

    # -- veterinary records -------------------------------------------------------

    def parse_veterinary(self, html: str) -> list[dict]:
        """Parse a horse's veterinary record page.

        The page lists one row per veterinary incident: the date of the
        incident, a free-text description, and the date the horse was passed
        on (cleared) to race.  This is the closest thing HKJC publishes to the
        report's 傷患/獸醫報告 column, and it is **pre-race** information: the
        record for an incident on date *d* was public before the horse's next
        run.
        """
        if not html or "No information" in html:
            return []
        try:
            tables = pd.read_html(io.StringIO(html))
        except Exception:
            return []

        records: list[dict] = []
        for t in tables:
            if t.shape[1] < 3 or t.shape[0] < 2:
                continue
            header = [str(c).strip() for c in t.iloc[0].tolist()]
            if "Details" not in header:
                continue
            idx_date = header.index("Date")
            idx_details = header.index("Details")
            idx_passed = header.index("Passed On") if "Passed On" in header else None
            for _, row in t.iloc[1:].iterrows():
                raw_date = str(row.iloc[idx_date]).strip()
                incident = _parse_hk_date(raw_date)
                if incident is None:
                    continue
                passed = None
                if idx_passed is not None:
                    passed = _parse_hk_date(str(row.iloc[idx_passed]).strip())
                records.append(
                    {
                        "vet_date": incident,
                        "vet_text": str(row.iloc[idx_details]).strip(),
                        "vet_passed_on": passed,
                    }
                )
            if records:
                break
        return records

    def collect_veterinary(self, horse_ids: list[str]) -> pd.DataFrame:
        """Fetch every horse's veterinary history."""
        from concurrent.futures import ThreadPoolExecutor

        def fetch(hid):
            html = self.fetch(VET_URL.format(horse_id=hid))
            if html is None:
                return []
            out = []
            for rec in self.parse_veterinary(html):
                rec["horse_id"] = hid
                out.append(rec)
            return out

        rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            for i, recs in enumerate(pool.map(fetch, horse_ids), start=1):
                rows.extend(recs)
                if self.config.verbose and i % 250 == 0:
                    print(f"  veterinary {i}/{len(horse_ids)} horses", flush=True)
        if not rows:
            return pd.DataFrame(columns=["horse_id", "vet_date", "vet_text", "vet_passed_on"])
        out = pd.DataFrame(rows)
        # Normalise the dates to Timestamps so they compare cleanly with the
        # meeting dates further down.
        out["vet_date"] = pd.to_datetime(out["vet_date"])
        out["vet_passed_on"] = pd.to_datetime(out["vet_passed_on"])
        return out

    # -- discovery -----------------------------------------------------------------

    def discover_race_days(self, start: date, end: date) -> list[date]:
        """Find the dates on which races were run.

        A date is a race day when the results page for race 1 contains a
        results table.  That costs one request per calendar day, which is
        cheaper than it sounds (the pages are cached) and far more reliable
        than trying to guess HK's race calendar.

        The days are probed concurrently ??there are thousands of them, and
        doing it one at a time would dominate the whole run.
        """
        from concurrent.futures import ThreadPoolExecutor

        days = []
        current = start
        while current <= end:
            days.append(current)
            current += timedelta(days=1)

        def probe(day: date) -> bool:
            url = RESULTS_URL.format(date=day.strftime("%Y/%m/%d"), race_no=1)
            html = self.fetch(url)
            return bool(html) and "No information" not in html and "Win Odds" in html

        found: list[date] = []
        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            for i, ok in enumerate(pool.map(probe, days), start=1):
                if ok:
                    found.append(days[i - 1])
                if self.config.verbose and i % 200 == 0:
                    print(f"  probed {i}/{len(days)} days, {len(found)} race days", flush=True)
        return found

    def count_races_on(self, meeting_date: date) -> int:
        """How many races were run on a meeting day."""
        html = self.fetch(
            RESULTS_URL.format(date=meeting_date.strftime("%Y/%m/%d"), race_no=1)
        )
        if not html:
            return 0
        numbers = {int(x) for x in re.findall(r"RaceNo=(\d+)", html)}
        numbers |= {int(x) for x in re.findall(r"Race\s+(\d+)\s*\(", html)}
        if not numbers:
            return 1
        return max(numbers)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


#: HKJC going descriptions -> the short codes used in form lines.
GOING_CODES = {
    "GOOD": "G",
    "GOOD TO FIRM": "GF",
    "GOOD TO YIELDING": "GY",
    "GOOD TO YIELDING OUTSIDE": "GYO",
    "YIELDING": "Y",
    "YIELDING TO SOFT": "YS",
    "SLOW": "SL",
    "FAST": "F",
    "WET SLOW": "WS",
    "WET FAST": "WF",
    "WET": "W",
    "HEAVY": "HV",
    "HARD": "H",
    "C3+3": "C3+3",
    "C4+3": "C4+3",
}


def _going_code(going: str) -> str:
    """Map HKJC's going description to its short code."""
    key = re.sub(r"\s+", " ", (going or "").strip().upper())
    if key in GOING_CODES:
        return GOING_CODES[key]
    for name, code in GOING_CODES.items():
        if key.startswith(name):
            return code
    return key[:4] if key else ""


def _track_code(course: str) -> str:
    """Extract the track condition code (``A``, ``C+3``, ...) from the course."""
    m = re.search(r'"([^"]+)"', course or "")
    return m.group(1) if m else ""


def _flat_columns(table: pd.DataFrame) -> list[str] | None:
    """Collapse pandas' multi-level column index into plain names."""
    if not isinstance(table.columns, pd.MultiIndex):
        return [str(c) for c in table.columns]
    out = []
    for col in table.columns:
        parts = [str(p) for p in col if str(p) not in {"Unnamed: 0_level_0"}]
        out.append(parts[-1] if parts else "")
    return out


def _flatten(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``table`` with single-level column names."""
    out = table.copy()
    out.columns = _flat_columns(table) or []
    return out


def _to_int(value) -> int | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text in {"", "nan", "None", "---", "-"}:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip().replace(",", "")
        if text in {"", "nan", "None", "---", "-"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_hk_date(text: str) -> date | None:
    """Parse HKJC's DD/MM/YY (and DD/MM/YYYY) date format."""
    text = text.strip()
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%d-%m-%y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------------------


def collect_races(config: CollectConfig | None = None) -> pd.DataFrame:
    """Phase 1 ??fetch every race result in the window.

    Returns one row per runner, with the race metadata merged in.
    """
    from concurrent.futures import ThreadPoolExecutor

    cfg = config or CollectConfig()
    collector = HKJCCollector(cfg)

    start = datetime.strptime(cfg.start, "%Y-%m-%d").date()
    end = datetime.strptime(cfg.end, "%Y-%m-%d").date()

    print(f"discovering race days between {start} and {end} ...", flush=True)
    race_days = collector.discover_race_days(start, end)
    print(f"found {len(race_days)} race days", flush=True)

    # How many races on each meeting day.
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        counts = list(pool.map(collector.count_races_on, race_days))

    jobs: list[tuple[date, int]] = []
    for day, count in zip(race_days, counts):
        for race_no in range(1, count + 1):
            jobs.append((day, race_no))
            if len(jobs) >= cfg.target_races:
                break
        if len(jobs) >= cfg.target_races:
            break
    print(f"{len(jobs)} races to fetch", flush=True)

    def fetch_one(job):
        day, race_no = job
        url = RESULTS_URL.format(date=day.strftime("%Y/%m/%d"), race_no=race_no)
        html = collector.fetch(url)
        if html is None:
            return None
        return collector.parse_results(html, day, race_no)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        for i, result in enumerate(pool.map(fetch_one, jobs), start=1):
            if result:
                meta, runners = result
                for r in runners:
                    rows.append({**meta, **r})
            if cfg.verbose and i % 250 == 0:
                print(f"  fetched {i}/{len(jobs)} races", flush=True)

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("no races collected")
        return frame

    frame["won"] = (frame["finish_position"] == 1).astype(int)
    # The results page carries the *final* (SP) price.  It is kept under an
    # explicit name so that no betting simulation can pick it up by accident.
    frame["win_odds_final"] = frame["win_odds_final"]
    frame["meeting_date"] = pd.to_datetime(frame["meeting_date"])

    # A conservative stand-in for the post time.  HKJC's first race of a
    # meeting posts at 12:45 (Happy Valley) or 13:00 (Sha Tin) and the rest
    # follow every ~30 minutes, so midday is a safe *lower* bound for every
    # race of the day.  It exists so that the text-leakage guard has something
    # to check the publication time of free text against.
    frame["race_start_time"] = frame["meeting_date"] + pd.Timedelta(hours=12)
    frame = frame.sort_values(["meeting_date", "race_no", "draw"]).reset_index(drop=True)

    print(
        f"collected {frame['race_id'].nunique():,} races / {len(frame):,} runner rows "
        f"({collector.stats.hits} fetched, {collector.stats.cached} cached)"
    )
    return frame


def enrich_horses(
    frame: pd.DataFrame,
    config: CollectConfig | None = None,
) -> pd.DataFrame:
    """Phase 2 ??fetch every horse's page and join the rating back in.

    The results page does not carry the horse's rating, so the rating comes
    from the horse's own race history, where it is the rating that applied at
    that run (i.e. the pre-race rating).
    """
    from concurrent.futures import ThreadPoolExecutor

    cfg = config or CollectConfig()
    collector = HKJCCollector(cfg)

    horse_ids = sorted({h for h in frame["horse_id"].dropna().unique()})
    print(f"fetching {len(horse_ids)} horse pages ...", flush=True)

    def fetch_horse(hid):
        html = collector.fetch(HORSE_URL.format(horse_id=hid))
        if html is None:
            return None
        parsed = collector.parse_horse(html)
        if parsed is None:
            return None
        meta, runs = parsed
        meta["horse_id"] = hid
        for r in runs:
            r["horse_id"] = hid
        return meta, runs

    meta_rows, run_rows = [], []
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        for i, res in enumerate(pool.map(fetch_horse, horse_ids), start=1):
            if res:
                meta, runs = res
                meta_rows.append(meta)
                run_rows.extend(runs)
            if cfg.verbose and i % 250 == 0:
                print(f"  fetched {i}/{len(horse_ids)} horses", flush=True)

    out = frame

    if meta_rows:
        horses = pd.DataFrame(meta_rows)
        cfg.horses_path.parent.mkdir(parents=True, exist_ok=True)
        horses.to_csv(cfg.horses_path, index=False, encoding="utf-8-sig")
        print(f"wrote {len(horses):,} horse records -> {cfg.horses_path}")

        # Fold the horse attributes into the race rows: country, sex, age and
        # import type are all known before the race, and all belong in the
        # feature set.
        keep = ["horse_id"] + [
            c
            for c in ("country", "sex", "colour", "import_type", "import_date", "age")
            if c in horses.columns
        ]
        out = out.merge(
            horses[keep].drop_duplicates(subset="horse_id"), on="horse_id", how="left"
        )
        print(
            "horse attributes joined: "
            + ", ".join(
                f"{c}={out[c].notna().mean():.0%}" for c in keep[1:] if c in out.columns
            ),
            flush=True,
        )

        # The horse page reports the age *today*, and only for horses imported
        # from 2019 on — so the age is recovered two ways:
        #
        #   * from the import date: HK horses are imported at three, and the
        #     published age agrees with "3 + years since import" to within a
        #     year for 97% of horses, so the import date gives the more
        #     precise version;
        #   * from the published age, backing out the elapsed time.
        #
        # Either way the result is the age the horse *was* at the race, which
        # is the pre-race information a model is allowed to use.
        if "age" in out.columns or "import_date" in out.columns:
            race_dates = pd.to_datetime(out["meeting_date"])
            derived = pd.Series(np.nan, index=out.index, dtype=float)

            if "import_date" in out.columns:
                imp = pd.to_datetime(out["import_date"], dayfirst=True, errors="coerce")
                elapsed = (race_dates - imp).dt.days / 365.25
                derived = (3 + np.floor(elapsed)).clip(lower=2).where(imp.notna())

            if "age" in out.columns:
                as_of = race_dates.max() + pd.Timedelta(days=300)
                back = out["age"] - np.floor((as_of - race_dates).dt.days / 365.25)
                derived = derived.where(derived.notna(), back.clip(lower=2))

            out["age_at_race"] = derived
            print(
                f"age_at_race available for {out['age_at_race'].notna().mean():.0%} of rows",
                flush=True,
            )

    if run_rows:
        runs_df = pd.DataFrame(run_rows)
        # A horse runs at most once per meeting, so (horse_id, meeting_date) is
        # a safe join key.
        key = ["horse_id", "meeting_date"]
        runs_df["meeting_date"] = pd.to_datetime(runs_df["run_date"])

        # The rating change since the horse's previous run ??the "Rtg. +/-"
        # figure on the racecard.  Computed from the history itself, so it is
        # available for every run rather than only the latest.
        runs_df = runs_df.sort_values(key).reset_index(drop=True)
        runs_df["rating_change"] = runs_df.groupby("horse_id")["rating"].diff()

        runs_df = runs_df.drop_duplicates(subset=key, keep="first")
        out = out.merge(
            runs_df[key + ["rating", "rating_change", "race_index"]].rename(
                columns={"race_index": "race_index_history"}
            ),
            on=key,
            how="left",
        )
        joined = out["rating"].notna().mean()
        print(f"ratings joined for {joined:.1%} of rows", flush=True)
        if joined < 0.5:
            print(
                "  WARNING: fewer than half the rows got a rating. "
                "The join key may need revisiting.",
                flush=True,
            )

    # --- veterinary records -> pre-race text ------------------------------------
    if cfg.enrich_veterinary and len(horse_ids):
        print(f"fetching veterinary records for {len(horse_ids)} horses ...", flush=True)
        vet = collector.collect_veterinary(horse_ids)
        if not vet.empty:
            # HKJC's veterinary-record endpoint does **not** filter by horse: it
            # returns the same population-level list of vet events for every
            # horse.  Attaching that list to each runner would smuggle a date
            # signal in and call it an injury record, so it is detected and
            # discarded rather than used.
            distinct = vet[["vet_date", "vet_text"]].drop_duplicates()
            if len(distinct) < 50 and len(vet) > 100:
                print(
                    f"  SKIPPED: the veterinary endpoint returned {len(distinct)} distinct "
                    f"events for {len(horse_ids)} horses — it is not filtered by horse, "
                    "so the notes cannot be attributed to a runner.",
                    flush=True,
                )
                vet = pd.DataFrame()
            else:
                vet = vet.sort_values(["horse_id", "vet_date"])
                race_dates = pd.to_datetime(out["meeting_date"])
                texts, days_since, cleared = [], [], []
                by_horse = {h: g for h, g in vet.groupby("horse_id")}
                for hid, rdate in zip(out["horse_id"], race_dates):
                    rec = by_horse.get(hid)
                    if rec is None or rec.empty:
                        texts.append("")
                        days_since.append(np.nan)
                        cleared.append(np.nan)
                        continue
                    prior = rec[rec["vet_date"] < rdate]
                    if prior.empty:
                        texts.append("")
                        days_since.append(np.nan)
                        cleared.append(np.nan)
                        continue
                    latest = prior.iloc[-1]
                    texts.append(str(latest["vet_text"]))
                    days_since.append(float((rdate - latest["vet_date"]).days))
                    passed = latest["vet_passed_on"]
                    # "Passed on" is when the horse was cleared to race.  If that
                    # is after the race date, the horse was not yet cleared.
                    cleared.append(0.0 if (passed is not None and passed > rdate) else 1.0)
                out["vet_report_text"] = texts
                # The guard in features.py checks a text column's timestamp
                # against the race start, so the note carries the date it was
                # published.
                out["vet_timestamp"] = [
                    rdate - pd.Timedelta(days=days) if days == days else pd.NaT
                    for rdate, days in zip(race_dates, days_since)
                ]
                out["vet_days_since"] = days_since
                out["vet_cleared"] = cleared
                with_text = (out["vet_report_text"].astype(str).str.len() > 0).mean()
                print(
                    f"veterinary notes attached to {with_text:.1%} of rows "
                    f"({len(vet):,} records over {vet['horse_id'].nunique():,} horses)",
                    flush=True,
                )
    return out
def collect_hkjc(config: CollectConfig | None = None) -> dict:
    """Collect HKJC races into the canonical frame and write them to disk."""
    cfg = config or CollectConfig()
    frame = collect_races(cfg)
    if frame.empty:
        return {"races": 0, "rows": 0}

    if cfg.enrich_horses:
        frame = enrich_horses(frame, cfg)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.out_path, index=False, encoding="utf-8-sig")
    print(
        f"wrote {len(frame):,} runner rows across {frame['race_id'].nunique():,} races "
        f"-> {cfg.out_path}"
    )
    return {
        "races": int(frame["race_id"].nunique()),
        "rows": int(len(frame)),
        "rated": float(frame["rating"].notna().mean()) if "rating" in frame else float("nan"),
    }
