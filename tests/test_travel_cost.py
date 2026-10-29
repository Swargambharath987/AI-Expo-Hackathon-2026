"""
Lock down the JTR travel-cost math. These numbers are the product's core
promise ("verifiable — judges can check the JTR math"), and this code path
has already had one silent bug (mileage lookup returning $0).
"""

import pytest

from agents.tools import (
    FIRST_LAST_PCT,
    POV_MILEAGE_RATE,
    calculate_travel_cost,
    get_per_diem,
)

# GSA FY2026 rates for Columbus, GA (Fort Moore) from data/gsa_cache.json
MOORE_LODGING, MOORE_MIE, MOORE_FIRST_LAST = 104, 64, 48


def moore(days, mode="POV", miles=370.0, origin=""):
    return calculate_travel_cost("Columbus", "GA", days, mode, miles, origin)


def test_demo_scenario_fort_liberty_to_fort_moore():
    r = moore(5)
    assert r["success"]
    assert r["lodging"]["total"] == 416      # 4 nights x $104
    assert r["meals"]["total"] == 288        # 2 x $48 + 3 x $64
    assert r["mileage"]["total"] == 518      # 370 mi x 2 x $0.70
    assert r["grand_total"] == 1222


def test_lodging_is_one_fewer_night_than_days():
    assert moore(5)["lodging"]["nights"] == 4
    assert moore(1)["lodging"]["nights"] == 0


@pytest.mark.parametrize("days, expected_meals", [
    (1, MOORE_FIRST_LAST),                               # single day counts as first/last
    (2, MOORE_FIRST_LAST * 2),                           # both days are travel days
    (3, MOORE_FIRST_LAST * 2 + MOORE_MIE),
    (10, MOORE_FIRST_LAST * 2 + MOORE_MIE * 8),
])
def test_first_and_last_day_meals_at_75_percent(days, expected_meals):
    assert moore(days)["meals"]["total"] == expected_meals


def test_first_last_rate_is_75_percent_of_mie():
    assert FIRST_LAST_PCT == 0.75
    assert MOORE_FIRST_LAST == MOORE_MIE * FIRST_LAST_PCT


def test_zero_or_negative_days_clamped_to_one():
    assert moore(0)["travel_days"] == 1
    assert moore(-3)["travel_days"] == 1


def test_mileage_rounds_to_cents():
    r = moore(3, miles=123.45)
    assert r["mileage"]["total"] == round(123.45 * 2 * POV_MILEAGE_RATE, 2) == 172.83


def test_known_distance_lookup_both_directions():
    # Regression: the lookup once used underscore-joined keys and always missed
    fwd = calculate_travel_cost("Fort Moore", "GA", 5, "POV", 0, origin="Fort Liberty")
    rev = calculate_travel_cost("Fort Liberty", "NC", 5, "POV", 0, origin="Fort Moore")
    assert fwd["mileage"]["one_way_miles"] == 370
    assert fwd["mileage"]["total"] == 518
    assert rev["mileage"]["total"] == 518


def test_pov_without_distance_reports_missing_mileage():
    r = moore(5, miles=0)
    assert r["mileage"]["total"] == 0
    assert "provide one_way_miles" in r["mileage"]["note"]


@pytest.mark.parametrize("mode", ["PLANE", "RENTAL", "GOVT"])
def test_no_mileage_for_non_pov(mode):
    r = moore(5, mode=mode)
    assert r["mileage"]["total"] == 0
    assert r["grand_total"] == 416 + 288


def test_unknown_city_falls_back_to_standard_conus_rate():
    r = get_per_diem("Nowhereville", "ZZ")
    assert r["match_type"] == "standard_conus_fallback"
    assert (r["lodging"], r["mie"]) == (110, 68)


def test_installation_name_resolves_to_city_rate():
    r = get_per_diem("Fort Moore", "GA")
    assert (r["lodging"], r["mie"]) == (MOORE_LODGING, MOORE_MIE)
