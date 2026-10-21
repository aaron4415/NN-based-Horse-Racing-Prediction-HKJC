"""Tests for the HKJC collector's parsers.

The fixtures are **synthetic mini-pages** that reproduce the *structure* of the
HKJC pages (the same table shapes, the same header labels) without embedding
any of HKJC's content.  That keeps third-party material out of the repository
while still exercising the parser end to end.
"""

from __future__ import annotations

import sys
from datetime import date

import pytest

from hkjc_nn.collect.hkjc import (
    HKJCCollector,
    _going_code,
    _parse_hk_date,
    _track_code,
)

RESULTS_HTML = """
<html><body>
<table><tr><td>Happy Valley:</td><td></td><td></td></tr></table>
<table>
  <tr><td>Class 5 - 1650M - (40-0)</td><td>Going :</td><td>GOOD</td></tr>
  <tr><td>TEST CUP HANDICAP</td><td>Course :</td><td>TURF - "A" Course</td></tr>
  <tr><td>HK$ 875,000</td><td>Time :</td><td>(28.29)</td></tr>
</table>
<table>
  <thead><tr>
    <td>Pla.</td><td>Horse No.</td><td>Horse</td><td>Jockey</td><td>Trainer</td>
    <td>Act. Wt.</td><td>Declar. Horse Wt.</td><td>Dr.</td><td>LBW</td>
    <td>Running Position</td><td>Finish Time</td><td>Win Odds</td>
  </tr></thead>
  <tbody>
    <tr><td>1</td><td>1</td>
      <td><a href="/en-us/local/information/horse?horseid=HK_2025_L045">FIRST HORSE (L045)</a></td>
      <td>Z Purton</td><td>D J Whyte</td><td>135</td><td>1057</td><td>7</td><td>---</td>
      <td>1 1 1 1</td><td>1:40.50</td><td>4.2</td></tr>
    <tr><td>2</td><td>12</td>
      <td><a href="/en-us/local/information/horse?horseid=HK_2024_K216">SECOND HORSE (K216)</a></td>
      <td>R Kingscote</td><td>D Eustace</td><td>120</td><td>1034</td><td>6</td><td>1-1/4</td>
      <td>3 3 4 2</td><td>1:40.70</td><td>8.7</td></tr>
  </tbody>
</table>
</body></html>
"""

HORSE_HTML = """
<html><body>
<table>
  <tr><td>Country of Origin</td><td>:</td><td>NZ / 5</td></tr>
  <tr><td>Colour / Sex</td><td>:</td><td>Chestnut / Gelding</td></tr>
  <tr><td>Import Type</td><td>:</td><td>PPG</td></tr>
</table>
<table>
  <tr><td>Race Index</td><td>Pla.</td><td>Date</td><td>RC/Track/ Course</td><td>Dist.</td>
      <td>G</td><td>Race Class</td><td>Dr.</td><td>Rtg.</td><td>Trainer</td><td>Jockey</td>
      <td>LBW</td><td>Win Odds</td><td>Act. Wt.</td><td>Running Position</td>
      <td>Finish Time</td><td>Declar. Horse Wt.</td><td>Gear</td><td>Video Replay</td></tr>
  <tr><td>26/27 Season</td><td>26/27 Season</td><td>26/27 Season</td><td>26/27 Season</td>
      <td>26/27 Season</td><td>26/27 Season</td><td>26/27 Season</td><td>26/27 Season</td>
      <td>26/27 Season</td><td>26/27 Season</td><td>26/27 Season</td><td>26/27 Season</td>
      <td>26/27 Season</td><td>26/27 Season</td><td>26/27 Season</td><td>26/27 Season</td>
      <td>26/27 Season</td><td>26/27 Season</td><td>26/27 Season</td></tr>
  <tr><td>099</td><td>01</td><td>14/10/26</td><td>HV / Turf / "A"</td><td>1650</td><td>G</td>
      <td>5</td><td>7</td><td>39</td><td>D J Whyte</td><td>Z Purton</td><td>1-1/4</td><td>4.2</td>
      <td>135</td><td>1 1 1 1</td><td>1.40.50</td><td>1057</td><td>--</td><td></td></tr>
  <tr><td>024</td><td>05</td><td>13/09/26</td><td>ST / Turf / "B"</td><td>1600</td><td>GF</td>
      <td>4</td><td>6</td><td>40</td><td>D J Whyte</td><td>K Teetan</td><td>2-1/4</td><td>11</td>
      <td>116</td><td>3 4 3 5</td><td>1.33.94</td><td>1062</td><td>H-/PC-</td><td></td></tr>
</table>
</body></html>
"""


@pytest.fixture
def collector(tmp_path):
    return HKJCCollector.__new__(HKJCCollector)  # no network, no cache


def test_parse_results_extracts_the_race_header(collector):
    meta, runners = collector.parse_results(RESULTS_HTML, date(2026, 10, 14), 1)

    assert meta["race_id"] == "20261014-01"
    assert meta["race_class"] == "5"
    assert meta["distance_m"] == 1650
    assert meta["going"] == "GOOD"
    assert meta["going_code"] == "G"
    assert meta["course"] == 'TURF - "A" Course'
    assert meta["track_code"] == "A"
    assert meta["race_name"] == "TEST CUP HANDICAP"
    assert meta["prize_money"] == 875000.0
    assert meta["num_runners"] == 2
    assert meta["venue"] == "Happy Valley"


def test_parse_results_extracts_every_runner(collector):
    _, runners = collector.parse_results(RESULTS_HTML, date(2026, 10, 14), 1)

    first = runners[0]
    assert first["horse_name"] == "FIRST HORSE"
    assert first["horse_id"] == "HK_2025_L045"
    assert first["jockey_name"] == "Z Purton"
    assert first["trainer_name"] == "D J Whyte"
    assert first["draw"] == 7
    assert first["finish_position"] == 1
    assert first["win_odds_final"] == pytest.approx(4.2)
    assert first["declared_weight"] == pytest.approx(1057.0)
    assert first["beaten_margin"] == "---"


def test_parse_results_ignores_an_empty_page(collector):
    assert collector.parse_results("<html>No information</html>", date(2026, 10, 14), 1) is None


def test_parse_horse_extracts_metadata_and_history(collector):
    meta, runs = collector.parse_horse(HORSE_HTML)

    assert meta["country"] == "NZ"
    assert meta["colour"] == "Chestnut"
    assert meta["sex"] == "Gelding"
    assert meta["import_type"] == "PPG"

    # The season-divider row must not become a run.
    assert len(runs) == 2
    assert runs[0]["run_date"] == date(2026, 10, 14)
    assert runs[0]["rating"] == 39
    assert runs[0]["draw"] == 7
    assert runs[0]["distance_m"] == 1650
    assert runs[0]["race_class"] == "5"
    assert runs[1]["run_date"] == date(2026, 9, 13)
    assert runs[1]["rating"] == 40


def test_going_and_track_codes():
    assert _going_code("GOOD") == "G"
    assert _going_code("GOOD TO FIRM") == "GF"
    assert _going_code("GOOD TO YIELDING") == "GY"
    assert _going_code("") == ""
    assert _track_code('TURF - "C+3" Course') == "C+3"
    assert _track_code("") == ""


def test_hk_date_parsing():
    assert _parse_hk_date("14/10/26") == date(2026, 10, 14)
    assert _parse_hk_date("13/09/2026") == date(2026, 9, 13)
    assert _parse_hk_date("not a date") is None


def test_odds_alias_maps_the_starting_price():
    """The collector's ``win_odds_final`` must land on the canonical ``win_odds``."""
    from hkjc_nn.data.csv_adapter import COLUMN_ALIASES

    assert "win_odds_final" in COLUMN_ALIASES["win_odds"]
