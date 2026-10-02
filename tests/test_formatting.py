from errorlens import Condition
from errorlens.utils.formatting import fmt_lift, fmt_num, fmt_p, fmt_pct


def test_p_values_never_round_up_across_thresholds():
    assert fmt_p(0.0497) == "0.049"
    assert fmt_p(0.0012) == "0.0012"
    assert fmt_p(0.0004) == "< 0.001"
    assert fmt_p(0.2) == "0.20"
    assert fmt_p(float("nan")) == "n/a"


def test_condition_values_are_compact_and_exact():
    assert str(Condition("x", "<=", 0.25)) == "x <= 0.25"
    assert str(Condition("x", ">", 30000.0)) == "x > 30,000"
    assert str(Condition("x", "<=", 1234.5)) == "x <= 1,234.5"
    assert str(Condition("flag", "==", True)) == "flag == True"


def test_number_formats():
    assert fmt_pct(0.1234) == "12.3%"
    assert fmt_lift(4.056) == "4.06x"
    assert fmt_num(1234567.0) == "1,234,567"
    assert fmt_num(0.012345) == "0.0123"
