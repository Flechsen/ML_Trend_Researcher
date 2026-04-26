import pytest
from ai_research_agent.budget import Budget, BudgetExceeded, PRICING


def test_pricing_table_has_required_models():
    assert "claude-sonnet-4-6" in PRICING
    assert "claude-haiku-4-5" in PRICING
    assert "text-embedding-3-small" in PRICING


def test_charge_accumulates_cost():
    b = Budget(cap_usd=1.0)
    b.charge("rank", "claude-haiku-4-5", in_tok=1000, out_tok=100)
    # Haiku: 1000 * $1e-6 + 100 * $5e-6 = $0.0015
    assert abs(b.spent - 0.0015) < 1e-9
    assert len(b.calls) == 1
    assert b.calls[0][0] == "rank"


def test_charge_records_per_stage():
    b = Budget(cap_usd=10.0)
    b.charge("rank", "claude-haiku-4-5", 1000, 100)
    b.charge("synthesize", "claude-sonnet-4-6", 1000, 100)
    stages = [c[0] for c in b.calls]
    assert stages == ["rank", "synthesize"]


def test_charge_raises_above_cap():
    b = Budget(cap_usd=0.001)
    with pytest.raises(BudgetExceeded) as ei:
        b.charge("rank", "claude-haiku-4-5", in_tok=1000, out_tok=1000)
    assert "$" in str(ei.value)


def test_charge_records_cost_even_when_raising():
    b = Budget(cap_usd=0.001)
    try:
        b.charge("rank", "claude-haiku-4-5", in_tok=1000, out_tok=1000)
    except BudgetExceeded:
        pass
    # Even after exceeding, the failing call should be recorded for visibility
    assert len(b.calls) == 1
    assert b.spent > 0.001


def test_unknown_model_raises_keyerror():
    b = Budget(cap_usd=1.0)
    with pytest.raises(KeyError):
        b.charge("rank", "claude-not-real", 100, 100)
