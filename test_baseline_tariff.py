"""FR-05 baseline + tariff arithmetic."""
import json

import pytest

from src.baseline import baseline_energy_kwh
from src.tariff import Tariff, TariffError, load_tariff


def test_baseline_formula_hand_calc():
    assert baseline_energy_kwh(150, 24) == pytest.approx(3.6)
    assert baseline_energy_kwh(2000, 0.5) == pytest.approx(1.0)


@pytest.mark.parametrize("p,d", [(0, 1), (-5, 1), (100, -1), (float("nan"), 1), (100, float("inf"))])
def test_baseline_rejects_invalid_inputs(p, d):
    with pytest.raises(ValueError):
        baseline_energy_kwh(p, d)


def test_flat_tariff():
    t = Tariff("t", "INR", "flat", rate_per_kwh=7.0)
    assert t.monthly_cost(10) == pytest.approx(70.0)
    assert t.cost_of_energy(2.5)[0] == pytest.approx(17.5)


def test_slab_tariff_progressive_hand_calc():
    t = Tariff("t", "INR", "slab", slabs=((100, 3.0), (200, 5.0), (None, 8.0)))
    assert t.monthly_cost(50) == pytest.approx(150)
    assert t.monthly_cost(250) == pytest.approx(100 * 3 + 100 * 5 + 50 * 8)


def test_slab_allocation_adds_up_to_monthly_cost():
    t = Tariff("t", "INR", "slab", slabs=((100, 3.0), (None, 8.0)))
    shares = [60, 90, 50]
    total = sum(shares)
    assert sum(t.cost_of_energy(s, total)[0] for s in shares) == pytest.approx(t.monthly_cost(total))


def test_slab_without_monthly_total_warns():
    t = Tariff("t", "INR", "slab", slabs=((100, 3.0), (None, 8.0)))
    assert t.cost_of_energy(10)[1]


def test_negative_energy_rejected():
    with pytest.raises(TariffError):
        Tariff("t", "INR", "flat", rate_per_kwh=1.0).monthly_cost(-1)


def test_load_tariff_validation(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"name": "x", "currency": "INR", "type": "slab",
                             "slabs": [{"up_to_kwh": 100, "rate_per_kwh": 3}]}))
    with pytest.raises(TariffError, match="open-ended"):
        load_tariff(p)
    assert load_tariff("configs/tariff.json").is_placeholder
