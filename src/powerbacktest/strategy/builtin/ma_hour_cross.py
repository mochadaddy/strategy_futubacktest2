"""Daily MA cross signal, executed on hourly bars with close-confirm / next-day reversal."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from powerbacktest.strategy.base import BarContext, PlotSpec, Strategy, StrategyParams
from powerbacktest.strategy.registry import register
from powerbacktest.strategy.tdx import CROSS, EMA, MA, REF


class MAHourCrossParams(StrategyParams):
    short_window: int = Field(20, ge=1, le=500)
    long_window: int = Field(50, ge=2, le=1000)
    ma_type: Literal["sma", "ema"] = "sma"
    turn_bars: int = Field(3, ge=1, le=20)
    bars_per_day: int = Field(4, ge=1, le=24, description="hourly bars per trading day")

    @model_validator(mode="after")
    def _order(self) -> "MAHourCrossParams":
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")
        return self


@register
class MAHourCrossStrategy(Strategy):
    name = "ma_hour_cross"
    title = "Daily MA cross, hourly execution"
    description = (
        "Daily short/long MA cross decides direction; executes intraday on hourly bars, "
        "confirms at the daily close, and reverses at next day's open if the signal vanished."
    )
    Params = MAHourCrossParams
    plots = (
        PlotSpec("ma_short", "price", "line", "MA short (daily)"),
        PlotSpec("ma_long", "price", "line", "MA long (daily)"),
    )
    params: MAHourCrossParams

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._day_signal: dict = {}  # date -> signal recorded at intraday trigger

    def warmup_bars(self) -> int:
        p = self.params
        base = p.long_window * (3 if p.ma_type == "ema" else 1)
        daily_bars_needed = base + 3 * p.turn_bars
        return daily_bars_needed * p.bars_per_day

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        fn = MA if p.ma_type == "sma" else EMA
        day = bars.index.normalize()

        # Daily OHLC built only from hourly bars already seen (causal).
        daily = bars.groupby(day).agg(close=("close", "last"))
        ma_short, ma_long = fn(daily["close"], p.short_window), fn(daily["close"], p.long_window)
        n = p.turn_bars
        up, down = ma_long > REF(ma_long, n), ma_long < REF(ma_long, n)

        # "Live" daily signal (same-day, no shift) -- used to detect intraday trigger.
        live_sig = pd.Series(float("nan"), index=daily.index)
        live_sig[CROSS(ma_short, ma_long) & up] = 1.0
        live_sig[CROSS(ma_long, ma_short) & down] = 0.0
        live_sig = live_sig.ffill()  # state held through the day until it flips

        day_series = pd.Series(day, index=bars.index)
        is_last_bar_of_day = day_series != day_series.shift(-1)

        return pd.DataFrame(
            {
                "ma_short": ma_short.reindex(day).to_numpy(),
                "ma_long": ma_long.reindex(day).to_numpy(),
                "daily_sig_live": live_sig.reindex(day).to_numpy(),
                "is_last_bar_of_day": is_last_bar_of_day.to_numpy(),
            },
            index=bars.index,
        )

    def signals(self, bars: pd.DataFrame, ind: pd.DataFrame) -> pd.Series:
        # Decision fully delegated to on_bar.
        return pd.Series(float("nan"), index=bars.index)

    def on_bar(self, ctx: BarContext) -> float:
        d = ctx.time.normalize()
        live_sig = ctx.indicator("daily_sig_live")
        is_close = bool(ctx.indicator("is_last_bar_of_day"))

        if live_sig != live_sig:  # NaN check, no opinion yet
            return ctx.signal

        triggered = self._day_signal.get(d)

        if triggered is None:
            # First trigger today -> act immediately on the hourly bar.
            self._day_signal[d] = live_sig
            return live_sig

        if is_close:
            if live_sig != triggered:
                # Signal vanished by the daily close -> flip; next_open fill
                # defers execution to the next trading day's open.
                self._day_signal[d] = live_sig
                return live_sig
            return ctx.signal  # confirmed -> keep current position

        return ctx.signal  # mid-day, no new trigger -> hold