from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from featureengineering import dataset as dataset_module
from featureengineering.dataset import DataRepository
from featureengineering.factor_loader import (
    _filter_non_point_in_time_factors,
    ensure_builtin_factors_loaded,
)
from featureengineering.factors.chip_deep import (
    _chip_daily_metrics,
    _compute_single_date_metrics,
)
from featureengineering.factors.chip_deep import CHIP_FACTOR_SPEC
from featureengineering.factors.indicator_minute import _indicator_all_metrics
from featureengineering.factors.intraday import _intraday_all_metrics
from featureengineering.factors.direct_fields import factor_pe_pb_divergence
from featureengineering.factors.margin_advanced import factor_total_leverage_ratio
from featureengineering.factors.margin_short_deep import factor_short_sell_volume_ratio
from featureengineering.factors.momentum_rebuilt import (
    factor_drawdown_60,
    factor_drawdown_120,
)
from featureengineering.factors.technical_daily import (
    _rolling_extreme_age,
    factor_eom_14,
    factor_rsi_spread_6_14,
)
from featureengineering.factors.technical_pattern import (
    _compute_rsrs_beta,
    factor_donchian_breakout_strength,
)
from featureengineering.factors.price import factor_breakout_60
from featureengineering.factors.price_deep import (
    factor_gap_fill_5d,
    factor_oi_divergence_intensity,
)
from featureengineering.factors.unused_fields_factors import (
    factor_mf_big_order_vol_ratio,
)
from featureengineering.factors.coupling_daily_extra2 import (
    factor_defensive_momentum_combo_60,
    factor_fund_flow_alpha_combo_60,
    factor_liftoff_pulse_combo_20,
    factor_margin_trend_combo_20,
    factor_volume_price_liftoff_20,
)
from featureengineering.registry import DEFAULT_LOOKBACK_DAYS, FACTOR_REGISTRY
from featureengineering.settings import ProjectPaths
from featureengineering.storage import ensure_single_factor_frame
from featureengineering.utils import (
    cross_sectional_rank,
    event_decay,
    stack_date_code,
)


def _paths(tmp_path: Path) -> ProjectPaths:
    source = tmp_path / "source"
    project = tmp_path / "project"
    source.mkdir()
    project.mkdir()
    pool = tmp_path / "codes.txt"
    pool.write_text("000001\n000002\n", encoding="utf-8")
    return ProjectPaths(
        project_root=project,
        source_root=source,
        factor_output_dir=project / "factors",
        manifest_output_dir=project / "manifests",
        target_output_dir=project / "targets",
        stock_pool_file=pool,
    )


def test_feature_registry_obeys_point_in_time_contract() -> None:
    ensure_builtin_factors_loaded()
    feature_specs = [s for s in FACTOR_REGISTRY.values() if s.category != "target"]

    assert feature_specs
    assert all("daily_adj.parquet" not in s.dependencies for s in feature_specs)
    assert DEFAULT_LOOKBACK_DAYS >= 800

    for spec in feature_specs:
        source = inspect.getsource(spec.compute)
        assert "cross_sectional_rank" in source or (
            ".rank(" in source and "Date" in source
        )
        tree = ast.parse(textwrap.dedent(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "shift" and node.args:
                try:
                    periods = ast.literal_eval(node.args[0])
                except (ValueError, TypeError):
                    continue
                assert not isinstance(periods, (int, float)) or periods >= 0
            if node.func.attr == "pct_change":
                fill = [kw for kw in node.keywords if kw.arg == "fill_method"]
                assert fill and isinstance(fill[0].value, ast.Constant)
                assert fill[0].value.value is None


def test_margin_is_mapped_to_exact_next_trading_day(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    daily = pd.DataFrame(
        {
            "trade_date": ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"] * 2,
            "stock_code": ["000001"] * 4 + ["000002"] * 4,
            "close": np.arange(8, dtype=float),
        }
    )
    # 000002 deliberately has no 2024-01-08 observation.  Its Friday value
    # must not be reused on Tuesday; only Tuesday's raw value is available Wed.
    margin = pd.DataFrame(
        {
            "trade_date": [
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-05",
                "2024-01-09",
            ],
            "stock_code": ["000001", "000001", "000001", "000002", "000002"],
            "rzye": [5.0, 8.0, 9.0, 50.0, 90.0],
        }
    )
    daily.to_parquet(paths.source_root / "daily.parquet", index=False)
    margin.to_parquet(paths.source_root / "margin_detail.parquet", index=False)

    dataset_module._ALLOWED_CODES = None
    panel = DataRepository(paths=paths).load_panel(
        "margin_detail.parquet", max_date="20240110"
    )

    assert panel.loc[("20240108", "000001"), "rzye"] == 5.0
    assert panel.loc[("20240109", "000001"), "rzye"] == 8.0
    assert ("20240109", "000002") not in panel.index
    assert panel.loc[("20240110", "000002"), "rzye"] == 90.0


def test_event_decay_is_continuous_before_first_event() -> None:
    idx = pd.MultiIndex.from_product(
        [["20240108", "20240109", "20240110"], ["000001", "000002"]],
        names=["Date", "Code"],
    )
    events = pd.Series(
        [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan], index=idx
    )
    decayed = event_decay(events, half_life=1)

    assert decayed.index.equals(events.index)
    assert decayed.notna().all()
    assert decayed.loc[("20240108", "000001")] == 0.0
    assert decayed.loc[("20240110", "000001")] == 0.5
    assert (decayed.xs("000002", level="Code") == 0.0).all()


class _PanelContext:
    def __init__(self, panels: dict[str, pd.DataFrame]):
        self.panels = panels

    def load(self, name: str) -> pd.DataFrame:
        return self.panels[name].copy()


class _FactorContext:
    def __init__(self, factors: dict[str, pd.Series]):
        self.factors = factors

    def load_factor(self, name: str) -> pd.Series:
        return self.factors[name].copy()


def test_margin_cross_source_factors_use_other_source_at_t() -> None:
    idx = pd.MultiIndex.from_product(
        [["20240109"], ["000001", "000002"]], names=["Date", "Code"]
    )
    margin = pd.DataFrame(
        {"rqmcl": [40.0, 20.0], "rzrqye": [100.0, 50.0]}, index=idx
    )
    daily = pd.DataFrame({"vol": [100.0, 100.0]}, index=idx)
    finance = pd.DataFrame({"total_mv": [1000.0, 1000.0]}, index=idx)
    context = _PanelContext(
        {
            "margin_detail.parquet": margin,
            "daily.parquet": daily,
            "finance.parquet": finance,
        }
    )

    short_ratio = factor_short_sell_volume_ratio(context)
    leverage = factor_total_leverage_ratio(context)

    assert short_ratio.loc[("20240109", "000001")] > short_ratio.loc[("20240109", "000002")]
    # total_leverage_ratio is intentionally negative-ranked (lower leverage is better).
    assert leverage.loc[("20240109", "000001")] < leverage.loc[("20240109", "000002")]


def test_drawdown_factors_rank_shallow_drawdown_higher() -> None:
    dates = pd.bdate_range("2024-01-02", periods=121).strftime("%Y%m%d")
    idx = pd.MultiIndex.from_product(
        [dates, ["000001", "000002"]], names=["Date", "Code"]
    )
    pct_chg = np.zeros(len(idx), dtype=float)
    pct_chg[-1] = -20.0  # final row is 000002 on the final date
    context = _PanelContext({"daily.parquet": pd.DataFrame({"pct_chg": pct_chg}, index=idx)})

    drawdown_60 = factor_drawdown_60(context)
    drawdown_120 = factor_drawdown_120(context)
    final_date = dates[-1]
    assert drawdown_60.loc[(final_date, "000001")] > drawdown_60.loc[(final_date, "000002")]
    assert drawdown_120.loc[(final_date, "000001")] > drawdown_120.loc[(final_date, "000002")]


def test_daily_rsi_uses_returns_as_price_changes() -> None:
    dates = pd.bdate_range("2024-01-02", periods=30).strftime("%Y%m%d")
    codes = ["000001", "000002"]
    idx = pd.MultiIndex.from_product([dates, codes], names=["Date", "Code"])
    returns = pd.DataFrame(
        {
            "000001": [-1.0] * 18 + [0.5] * 6 + [2.0] * 6,
            "000002": [1.0] * 18 + [-0.5] * 6 + [-2.0] * 6,
        },
        index=dates,
    )
    daily = pd.DataFrame(
        {"pct_chg": stack_date_code(returns).reindex(idx)}, index=idx
    )

    actual = factor_rsi_spread_6_14(_PanelContext({"daily.parquet": daily}))
    delta = returns / 100.0

    def expected_rsi(n: int) -> pd.DataFrame:
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100.0 - 100.0 / (1.0 + rs)

    expected = cross_sectional_rank(
        stack_date_code(expected_rsi(6) - expected_rsi(14))
    )
    pd.testing.assert_series_equal(actual, expected)


def test_aroon_extreme_age_stays_inside_rolling_window() -> None:
    dates = pd.bdate_range("2024-01-02", periods=30).strftime("%Y%m%d")
    wide = pd.DataFrame(
        {
            "down": np.arange(30.0, 0.0, -1.0),
            "up": np.arange(1.0, 31.0),
        },
        index=dates,
    )
    high_age = _rolling_extreme_age(wide, window=25, find_max=True)
    low_age = _rolling_extreme_age(wide, window=25, find_max=False)

    assert high_age.iloc[-1].to_dict() == {"down": 24.0, "up": 0.0}
    assert low_age.iloc[-1].to_dict() == {"down": 0.0, "up": 24.0}
    assert high_age.max().max() <= 24.0
    assert low_age.max().max() <= 24.0


def test_eom_rewards_easier_movement_not_higher_volume() -> None:
    dates = pd.bdate_range("2024-01-02", periods=16).strftime("%Y%m%d")
    idx = pd.MultiIndex.from_product(
        [dates, ["000001", "000002"]], names=["Date", "Code"]
    )
    daily = pd.DataFrame(
        {
            "pct_chg": 1.0,
            "close": 10.1,
            "pre_close": 10.0,
            "high": 10.2,
            "low": 10.0,
            "vol": np.tile([100.0, 1000.0], len(dates)),
        },
        index=idx,
    )
    factor = factor_eom_14(_PanelContext({"daily.parquet": daily}))
    final = factor.xs(dates[-1], level="Date")
    assert final["000001"] > final["000002"]


def test_chip_trap_and_profit_intervals_are_not_swapped() -> None:
    assert CHIP_FACTOR_SPEC["chip_deep_trap_ratio"] == ("chip_upper_110", "neg")
    assert CHIP_FACTOR_SPEC["chip_high_float_ratio"] == ("chip_below_90", "neg")


def test_pe_pb_quality_direction_matches_implied_roe() -> None:
    idx = pd.MultiIndex.from_product(
        [["20240109"], ["000001", "000002", "000003"]],
        names=["Date", "Code"],
    )
    finance = pd.DataFrame(
        {
            "pe_ttm": [5.0, 10.0, 20.0],
            "pb": [3.0, 2.0, 1.0],
        },
        index=idx,
    )
    factor = factor_pe_pb_divergence(_PanelContext({"finance.parquet": finance}))
    assert factor.loc[("20240109", "000001")] > factor.loc[("20240109", "000003")]


def test_big_order_volume_ratio_measures_participation_not_net_flow() -> None:
    idx = pd.MultiIndex.from_product(
        [["20240109"], ["000001", "000002"]], names=["Date", "Code"]
    )
    columns = {}
    for size in ("sm", "md", "lg", "elg"):
        columns[f"buy_{size}_vol"] = [10.0, 10.0]
        columns[f"sell_{size}_vol"] = [10.0, 10.0]
    # Equal buy/sell means zero net flow for both stocks, but stock 000001 has
    # substantially greater large-order participation.
    for side in ("buy", "sell"):
        columns[f"{side}_lg_vol"] = [100.0, 10.0]
        columns[f"{side}_elg_vol"] = [100.0, 10.0]
    panel = pd.DataFrame(columns, index=idx)
    factor = factor_mf_big_order_vol_ratio(
        _PanelContext({"main_fund_flow.parquet": panel})
    )
    assert factor.loc[("20240109", "000001")] > factor.loc[("20240109", "000002")]


def test_chip_gini_is_independent_of_cost_location() -> None:
    left = pd.DataFrame(
        {"price": [9.0, 10.0, 11.0], "percent": [80.0, 10.0, 10.0]}
    )
    right = pd.DataFrame(
        {"price": [9.0, 10.0, 11.0], "percent": [10.0, 10.0, 80.0]}
    )
    left_gini = _compute_single_date_metrics(left)["chip_gini"]
    right_gini = _compute_single_date_metrics(right)["chip_gini"]
    assert left_gini == pytest.approx(right_gini)
    assert left_gini > 0.0


def test_rsrs_waits_for_the_declared_regression_window() -> None:
    dates = pd.bdate_range("2024-01-02", periods=18).strftime("%Y%m%d")
    idx = pd.MultiIndex.from_product(
        [dates, ["000001"]], names=["Date", "Code"]
    )
    low = pd.Series(np.arange(1.0, 19.0), index=idx)
    high = 2.0 * low + 1.0
    beta, r2 = _compute_rsrs_beta(high, low, window=18)
    assert beta.iloc[:-1].isna().all()
    assert beta.iloc[-1] == pytest.approx(2.0)
    assert r2.iloc[-1] == pytest.approx(1.0)


def _breakout_daily(periods: int = 65) -> tuple[pd.DataFrame, str]:
    dates = pd.bdate_range("2024-01-02", periods=periods).strftime("%Y%m%d")
    idx = pd.MultiIndex.from_product(
        [dates, ["000001", "000002"]], names=["Date", "Code"]
    )
    frame = pd.DataFrame(
        {
            "pct_chg": 0.0,
            "close": 10.0,
            "pre_close": 10.0,
            "high": 10.1,
            "low": 9.9,
        },
        index=idx,
    )
    final = dates[-1]
    frame.loc[(final, "000001"), ["pct_chg", "close", "high"]] = [10.0, 11.0, 11.1]
    return frame, final


def test_breakout_baselines_exclude_the_current_session() -> None:
    daily, final = _breakout_daily()
    context = _PanelContext({"daily.parquet": daily})
    donchian = factor_donchian_breakout_strength(context)
    breakout = factor_breakout_60(context)
    for factor in (donchian, breakout):
        assert factor.loc[(final, "000001")] > factor.loc[(final, "000002")]


def test_gap_fill_uses_the_gap_anchor_not_plain_five_day_momentum() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10).strftime("%Y%m%d")
    idx = pd.MultiIndex.from_product(
        [dates, ["000001", "000002"]], names=["Date", "Code"]
    )
    daily = pd.DataFrame(
        {
            "pct_chg": 0.0,
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "pre_close": 10.0,
        },
        index=idx,
    )
    event_date = dates[-6]
    for code in ("000001", "000002"):
        daily.loc[(event_date, code), ["pct_chg", "open", "high", "low", "close"]] = [
            8.0, 11.0, 11.2, 10.5, 10.8
        ]
        later = pd.IndexSlice[dates[-5]:dates[-1], code]
        daily.loc[later, ["open", "high", "close", "pre_close"]] = [10.8, 11.0, 10.8, 10.8]
    daily.loc[(dates[-3], "000001"), "low"] = 9.9   # filled
    daily.loc[pd.IndexSlice[dates[-5]:dates[-1], "000002"], "low"] = 10.5

    factor = factor_gap_fill_5d(_PanelContext({"daily.parquet": daily}))
    assert factor.loc[(dates[-1], "000001")] > factor.loc[(dates[-1], "000002")]


def test_gap_reinforcement_respects_gap_direction() -> None:
    idx = pd.MultiIndex.from_product(
        [["20240109"], ["000001", "000002"]], names=["Date", "Code"]
    )
    daily = pd.DataFrame(
        {
            "pre_close": [10.0, 10.0],
            "open": [11.0, 11.0],
            "close": [11.5, 10.5],
        },
        index=idx,
    )
    factor = factor_oi_divergence_intensity(
        _PanelContext({"daily.parquet": daily})
    )
    assert factor.loc[("20240109", "000001")] > factor.loc[("20240109", "000002")]


def test_coupling_factors_follow_registered_rank_directions() -> None:
    idx = pd.MultiIndex.from_product(
        [["20240109"], ["000001", "000002"]], names=["Date", "Code"]
    )
    constant = pd.Series([0.8, 0.8], index=idx)
    favorable = pd.Series([0.9, 0.1], index=idx)
    ctx = _FactorContext(
        {
            "momentum_20": constant,
            "momentum_10": constant,
            "momentum_60": constant,
            "volume_breakout_confirm_20": constant,
            "amount_surge_count_20": constant,
            "margin_net_flow_ratio": constant,
            # These registered factors are already oriented so high means
            # shallow drawdown, low beta, low ulcer, or low correlation.
            "drawdown_60": favorable,
            "beta_60": favorable,
            "ulcer_index_20": constant,
            "mf_net_inflow_5d": constant,
            "corr_market_60": favorable,
        }
    )

    outputs = (
        factor_volume_price_liftoff_20(ctx),
        factor_liftoff_pulse_combo_20(ctx),
        factor_margin_trend_combo_20(ctx),
        factor_defensive_momentum_combo_60(ctx),
        factor_fund_flow_alpha_combo_60(ctx),
    )
    for output in outputs:
        assert output.loc[("20240109", "000001")] > output.loc[("20240109", "000002")]


def test_current_snapshot_mapping_is_not_registered() -> None:
    ensure_builtin_factors_loaded()
    _filter_non_point_in_time_factors()
    forbidden = {
        "sector_mv_rank", "sector_amount_rank",
        "sector_amount_momentum_5d", "industry_relative_momentum_20",
        "pb_industry_adjusted", "ps_ttm_sector_neutral",
    }
    assert forbidden.isdisjoint(FACTOR_REGISTRY)
    assert all(
        "stock_list.parquet" not in spec.dependencies
        for spec in FACTOR_REGISTRY.values()
    )


def test_factor_alignment_uses_explicit_project_paths(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    daily = pd.DataFrame(
        {
            "trade_date": ["2024-01-08", "2024-01-08", "2024-01-09", "2024-01-09"],
            "stock_code": ["000001", "000002", "000001", "000002"],
        }
    )
    daily.to_parquet(paths.source_root / "daily.parquet", index=False)
    idx = pd.MultiIndex.from_tuples(
        [("20240108", "000001"), ("20240108", "000002")],
        names=["Date", "Code"],
    )
    raw = pd.Series([0.25, 0.75], index=idx)

    aligned = ensure_single_factor_frame(raw, "test_factor", paths=paths)

    assert list(aligned.index) == ["20240108", "20240109"]
    assert list(aligned.columns) == ["000001", "000002"]
    assert aligned.loc["20240109", "000001"] == 0.25


def _minute_frame() -> pd.DataFrame:
    times = []
    for day in ("2024-01-08", "2024-01-09"):
        times.extend(pd.date_range(f"{day} 09:30", periods=75, freq="min"))
    n = len(times)
    close = 10.0 + np.arange(n) * 0.01
    return pd.DataFrame(
        {
            "trade_time": times,
            "open": close - 0.01,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "vol": 100.0 + np.arange(n),
            "amount": (100.0 + np.arange(n)) * close,
        }
    )


def test_intraday_metrics_are_invariant_to_parquet_row_order() -> None:
    ordered = _minute_frame()
    shuffled = ordered.sample(frac=1.0, random_state=7).reset_index(drop=True)
    expected = _intraday_all_metrics(ordered)
    actual = _intraday_all_metrics(shuffled)
    pd.testing.assert_frame_equal(actual, expected)


def test_indicator_metrics_are_invariant_to_parquet_row_order() -> None:
    base = _minute_frame()
    n = len(base)
    x = np.linspace(-1.0, 1.0, n)
    indicator = pd.DataFrame(
        {
            "trade_time": base["trade_time"],
            "close": base["close"],
            "dif": x,
            "dea": x * 0.8,
            "macd": x * 0.4,
            "k": 50.0 + x * 10,
            "d": 50.0 + x * 8,
            "j": 50.0 + x * 14,
            "rsi": 50.0 + x * 20,
            "boll_mid": base["close"],
            "boll_upper": base["close"] + 0.5,
            "boll_lower": base["close"] - 0.5,
            "ma5": base["close"] - 0.05,
            "ma10": base["close"] - 0.10,
            "ma20": base["close"] - 0.20,
            "ma30": base["close"] - 0.30,
            "ma60": base["close"] - 0.60,
            "mavol5": 1000.0 + np.arange(n),
            "mavol10": 1100.0 + np.arange(n),
        }
    )
    shuffled = indicator.sample(frac=1.0, random_state=11).reset_index(drop=True)
    expected = _indicator_all_metrics(indicator)
    actual = _indicator_all_metrics(shuffled)
    pd.testing.assert_frame_equal(actual, expected)


def test_minute_metric_engines_are_prefix_invariant() -> None:
    days = pd.bdate_range("2024-01-02", periods=30)
    frames = []
    for day_no, day in enumerate(days):
        frame = _minute_frame().iloc[:75].copy()
        frame["trade_time"] = pd.date_range(
            f"{day:%Y-%m-%d} 09:30", periods=len(frame), freq="min"
        )
        frame["close"] += day_no * 0.03
        frames.append(frame)
    minute = pd.concat(frames, ignore_index=True)
    n = len(minute)
    x = np.linspace(-1.0, 1.0, n)
    indicator = minute.assign(
        dif=x,
        dea=x * 0.8,
        macd=x * 0.4,
        k=50.0 + x * 10,
        d=50.0 + x * 8,
        j=50.0 + x * 14,
        rsi=50.0 + x * 20,
        boll_mid=minute["close"],
        boll_upper=minute["close"] + 0.5,
        boll_lower=minute["close"] - 0.5,
        ma5=minute["close"] - 0.05,
        ma10=minute["close"] - 0.10,
        ma20=minute["close"] - 0.20,
        ma30=minute["close"] - 0.30,
        ma60=minute["close"] - 0.60,
        mavol5=1000.0 + np.arange(n),
        mavol10=1100.0 + np.arange(n),
    )
    cutoff = days[23].strftime("%Y-%m-%d")
    minute_prefix = minute[minute["trade_time"].dt.strftime("%Y-%m-%d") <= cutoff]
    indicator_prefix = indicator[
        indicator["trade_time"].dt.strftime("%Y-%m-%d") <= cutoff
    ]

    intraday_expected = _intraday_all_metrics(minute_prefix)
    intraday_full = _intraday_all_metrics(minute).loc[intraday_expected.index]
    pd.testing.assert_frame_equal(intraday_full, intraday_expected)

    indicator_expected = _indicator_all_metrics(indicator_prefix)
    indicator_full = _indicator_all_metrics(indicator).loc[indicator_expected.index]
    pd.testing.assert_frame_equal(indicator_full, indicator_expected)


def test_chip_metrics_are_sorted_and_order_invariant() -> None:
    rows = []
    for date in ("20240105", "20240108", "20240109"):
        for price, percent in ((9.0, 10.0), (10.0, 50.0), (11.0, 40.0)):
            rows.append((date, price, percent))
    chips = pd.DataFrame(rows, columns=["trade_date", "price", "percent"])
    shuffled = chips.sample(frac=1.0, random_state=13).reset_index(drop=True)
    expected = _chip_daily_metrics(chips)
    actual = _chip_daily_metrics(shuffled)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.index.is_monotonic_increasing


def test_chip_metric_engine_is_prefix_invariant() -> None:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=30).strftime("%Y%m%d")
    for day_no, date in enumerate(dates):
        for price, percent in ((9.0, 10.0), (10.0, 50.0), (11.0, 40.0)):
            rows.append((date, price + day_no * 0.01, percent))
    chips = pd.DataFrame(rows, columns=["trade_date", "price", "percent"])
    prefix = chips[chips["trade_date"] <= dates[23]]
    expected = _chip_daily_metrics(prefix)
    actual = _chip_daily_metrics(chips).loc[expected.index]
    pd.testing.assert_frame_equal(actual, expected)
