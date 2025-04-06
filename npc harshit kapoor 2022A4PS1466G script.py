import logging
import os
from decimal import Decimal
from typing import Dict, List
import inspect

import pandas as pd
import pandas_ta as ta

from pydantic import Field
from hummingbot.client.config.config_data_types import BaseClientModel, ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig


class SmartTrendVolatilityMakerConfig(BaseClientModel):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    exchange: str = Field("binance_paper_trade", client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Exchange to trade on"))
    trading_pair: str = Field("SOL-FDUSD", client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Trading pair to use"))
    order_amount: Decimal = Field(1.0, client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Order amount"))
    order_refresh_time: int = Field(15, client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Order refresh time (in seconds)"))
    candles_interval: str = Field("1m", client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Candlestick interval (e.g., 1m, 5m)"))
    candles_max_records: int = Field(1000, client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Max candle records to fetch"))


class SmartTrendVolatilityMaker(ScriptStrategyBase):
    price_source = PriceType.MidPrice

    @classmethod
    def init_markets(cls, config: SmartTrendVolatilityMakerConfig):
        cls.markets = {config.exchange: {config.trading_pair}}
        cls.candles_config = CandlesConfig(
            connector=config.exchange.replace("_paper_trade", ""),
            trading_pair=config.trading_pair,
            interval=config.candles_interval,
            max_records=config.candles_max_records
        )
        cls.candles = CandlesFactory.get_candle(cls.candles_config)

    def __init__(self, connectors: Dict[str, ConnectorBase], config: SmartTrendVolatilityMakerConfig):
        super().__init__(connectors)
        self.config = config
        self.create_timestamp = 0
        self.candles = self.__class__.candles
        self.candles.start()

    # def on_stop(self):
    #     self.candles.stop()
    

    def on_stop(self):
        if hasattr(self, "candles") and self.candles is not None:
            stop_method = getattr(self.candles, "stop", None)
            if stop_method:
                if inspect.iscoroutinefunction(stop_method):
                    self._safe_ensure_future(stop_method())
                else:
                    stop_method()


    def on_tick(self):
        if not self.ready_to_trade or not self.candles.ready:
            return

        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()

            candles_df = self.get_candles_with_indicators()
            trend = self.compute_trend_state(candles_df)
            volatility = self.compute_volatility_state(candles_df)
            is_retracing, retrace_zone = self.detect_fibonacci_retracement(candles_df)

            ref_price = self.connectors[self.config.exchange].get_price_by_type(self.config.trading_pair, self.price_source)
            mid_price = Decimal(str(ref_price))
            mean_price, bid_spread, ask_spread, buy_size, sell_size = self.adjust_parameters(
                trend, volatility, candles_df, mid_price, is_retracing, retrace_zone
            )

            best_bid = self.connectors[self.config.exchange].get_price(self.config.trading_pair, False)
            best_ask = self.connectors[self.config.exchange].get_price(self.config.trading_pair, True)

            buy_price = min(mean_price * (1 - bid_spread), Decimal(str(best_bid)))
            sell_price = max(mean_price * (1 + ask_spread), Decimal(str(best_ask)))

            proposal = [
                OrderCandidate(self.config.trading_pair, True, OrderType.LIMIT, TradeType.BUY, Decimal(buy_size), buy_price),
                OrderCandidate(self.config.trading_pair, True, OrderType.LIMIT, TradeType.SELL, Decimal(sell_size), sell_price)
            ]
            proposal_adjusted = self.connectors[self.config.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)

            for order in proposal_adjusted:
                self.place_order(self.config.exchange, order)

            self.create_timestamp = self.current_timestamp + self.config.order_refresh_time

    def get_candles_with_indicators(self) -> pd.DataFrame:
        df = self.candles.candles_df.copy()
        df.ta.macd(append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.bbands(length=20, append=True)
        df.ta.natr(length=14, append=True)
        df["volume_sma"] = df["volume"].rolling(window=5).mean()
        return df

    def compute_trend_state(self, df: pd.DataFrame) -> str:
        macd_line = df["MACD_12_26_9"].iloc[-1]
        macd_signal = df["MACDs_12_26_9"].iloc[-1]
        macd_cross = "bullish" if macd_line > macd_signal else "bearish" if macd_line < macd_signal else "none"

        rsi = df["RSI_14"].iloc[-1]
        close = df["close"].iloc[-1]
        lower_bb = df["BBL_20_2.0"].iloc[-1]
        upper_bb = df["BBU_20_2.0"].iloc[-1]
        bb_position = (close - lower_bb) / (upper_bb - lower_bb) if upper_bb > lower_bb else 0.5

        macd_score = 1.0 if macd_cross == "bullish" else 0.0 if macd_cross == "bearish" else 0.5
        rsi_score = 0.0 if rsi < 45 else 1.0 if rsi > 55 else 0.5
        bb_score = 0.0 if bb_position < 0.2 else 1.0 if bb_position > 0.8 else 0.5

        if (macd_cross == "bullish" and rsi_score == 0.0):
            return "bullish"
        elif (macd_cross == "bearish" and rsi_score == 1.0):
            return "bearish"

        trend_score = 0.5 * macd_score + 0.3 * rsi_score + 0.2 * bb_score
        return "bullish" if trend_score >= 0.7 else "bearish" if trend_score <= 0.3 else "neutral"

    def compute_volatility_state(self, df: pd.DataFrame) -> str:
        natr = df["NATR_14"].iloc[-1]
        bb_width = (df["BBU_20_2.0"].iloc[-1] - df["BBL_20_2.0"].iloc[-1]) / df["close"].iloc[-1]
        return "high" if (bb_width + natr) > 0.055 else "low"

    # def detect_fibonacci_retracement(self, df: pd.DataFrame) -> (bool, tuple):
    #     recent = df.tail(30)
    #     high = recent["high"].max()
    #     low = recent["low"].min()
    #     last_price = recent["close"].iloc[-1]
    #     retrace_levels = {
    #         0.618: high - 0.618 * (high - low),
    #         0.5: high - 0.5 * (high - low),
    #         0.382: high - 0.382 * (high - low)
    #     }
    #     for level in retrace_levels.values():
    #         if abs(last_price - level) / last_price < 0.01:
    #             return True, (min(retrace_levels.values()), max(retrace_levels.values()))
    #     return False, (None, None)
    def detect_fibonacci_retracement(self, df: pd.DataFrame) -> (bool, tuple):
    # Step 1: Calculate fractal points
        df["fractal_high"] = df["high"][(df["high"].shift(2) < df["high"]) &
                                        (df["high"].shift(1) < df["high"]) &
                                        (df["high"].shift(-1) < df["high"]) &
                                        (df["high"].shift(-2) < df["high"])]

        df["fractal_low"] = df["low"][(df["low"].shift(2) > df["low"]) &
                                    (df["low"].shift(1) > df["low"]) &
                                    (df["low"].shift(-1) > df["low"]) &
                                    (df["low"].shift(-2) > df["low"])]

        recent = df.tail(100).copy()
        highs = recent.dropna(subset=["fractal_high"])
        lows = recent.dropna(subset=["fractal_low"])

        if len(highs) < 1 or len(lows) < 1:
            return False, (None, None)

        last_high_idx = highs.index[-1]
        last_low_idx = lows.index[-1]

        # Step 2: Ensure fractals are within 20 candles of each other
        candles_apart = abs(last_high_idx - last_low_idx)
        if candles_apart > 20:
            return False, (None, None)

        # Step 3: Determine the closest fractal to now
        current_idx = df.index[-1]
        dist_to_high = abs(current_idx - last_high_idx)
        dist_to_low = abs(current_idx - last_low_idx)

        closest_idx = last_high_idx if dist_to_high < dist_to_low else last_low_idx
        closest_type = "high" if dist_to_high < dist_to_low else "low"

        # Step 4: Ensure the closest fractal is recent (within 20 candles of now)
        if abs(current_idx - closest_idx) > 20:
            return False, (None, None)

        # Step 5: Compute trend at closest fractal index
        trend_window = df.loc[:closest_idx].tail(30)
        past_trend = self.compute_trend_state(trend_window)

        # Step 6: Get current trend
        current_trend = self.compute_trend_state(df.tail(30))

        if past_trend != current_trend:
            return False, (None, None)

        # Step 7: Calculate Fib levels using fractals
        high_price = df.loc[last_high_idx]["fractal_high"]
        low_price = df.loc[last_low_idx]["fractal_low"]

        
        move_high = high_price
        move_low = low_price

        last_price = df["close"].iloc[-1]
        if not (min(move_high, move_low) <= last_price <= max(move_high, move_low)):
            return False, (None, None)
        retrace_levels = {
            0.618: move_high - 0.618 * (move_high - move_low),
            0.5: move_high - 0.5 * (move_high - move_low),
            0.382: move_high - 0.382 * (move_high - move_low)
        }

        for level in retrace_levels.values():
            if abs(last_price - level) / last_price < 0.0002:  # within 0.07% of level
                return True, (min(retrace_levels.values()), max(retrace_levels.values()))

        return False, (None, None)


    def adjust_parameters(self, trend, volatility, df, mid_price, is_retracing, retrace_zone):
        bid_spread = ask_spread = Decimal("0.0007")
        buy_size = sell_size = self.config.order_amount
        mean_price = mid_price

        # volume_spike = df["volume"].iloc[-1] > 2 * df["volume_sma"].iloc[-1]
        # broke_upper = df["close"].iloc[-1] > df["BBU_20_2.0"].iloc[-1]
        # broke_lower = df["close"].iloc[-1] < df["BBL_20_2.0"].iloc[-1]
        volume_spike = False
        broke_upper = False
        broke_lower = False

        # Check the last 15 candles for a BB break with a volume spike
        for i in range(1, 16):
            close = df["close"].iloc[-i]
            upper = df["BBU_20_2.0"].iloc[-i]
            lower = df["BBL_20_2.0"].iloc[-i]
            vol = df["volume"].iloc[-i]
            vol_sma = df["volume_sma"].iloc[-i]

            if trend == "bullish" and close > upper:
                broke_upper = True
                if vol > 2 * vol_sma:
                    volume_spike = True
                break

            elif trend == "bearish" and close < lower:
                broke_lower = True
                if vol > 2 * vol_sma:
                    volume_spike = True
                break

        # Use NATR (normalized ATR in %)
        natr_pct = Decimal(str(df["NATR_14"].iloc[-1])) / Decimal("100")  # Convert percent to decimal
        natr_offset = mid_price * natr_pct
        if trend == "neutral" and volatility == "low":
            buy_size = sell_size = self.config.order_amount * Decimal("1.5")
            
        if trend == "neutral" and volatility == "high":
            buy_size = sell_size = self.config.order_amount * Decimal("0.9")
            bid_spread = ask_spread = Decimal("0.0015")
            

        elif trend == "bullish":
            if volatility == "low":
                bid_spread = ask_spread = Decimal("0.001")
                mean_price = mid_price + (natr_offset * Decimal("0.6"))
                buy_size = self.config.order_amount * Decimal("1.1")
            else:
                bid_spread, ask_spread = (Decimal("0.002"), Decimal("0.004")) if broke_upper and volume_spike else (Decimal("0.004"), Decimal("0.007"))
                mean_price = mid_price + (natr_offset * Decimal("0.9"))
                if is_retracing:
                    mean_price = Decimal(str(df["close"].iloc[-1]))
                    buy_size = self.config.order_amount * Decimal("1.2")
                    sell_size = self.config.order_amount * Decimal("0.8")

        elif trend == "bearish":
            if volatility == "low":
                bid_spread = ask_spread = Decimal("0.001")
                mean_price = mid_price - (natr_offset * Decimal("0.6"))
                sell_size = self.config.order_amount * Decimal("1.1")
            else:
                bid_spread, ask_spread = (Decimal("0.004"), Decimal("0.002")) if broke_lower and volume_spike else (Decimal("0.007"), Decimal("0.004"))
                mean_price = mid_price + (natr_offset * Decimal("0.9"))
                if is_retracing:
                    mean_price = Decimal(str(df["close"].iloc[-1]))
                    sell_size = self.config.order_amount * Decimal("1.2")
                    buy_size = self.config.order_amount * Decimal("0.8")
        base_asset, quote_asset = self.config.trading_pair.split("-")
        connector = self.connectors[self.config.exchange]

        base_balance = connector.get_balance(base_asset)
        quote_balance = connector.get_balance(quote_asset)

        mid_price = Decimal(str(connector.get_price_by_type(self.config.trading_pair, self.price_source)))
        base_value = base_balance * mid_price
        total_value = base_value + quote_balance

        if total_value == 0:
            sol_pct = Decimal("0.5")
        else:
            sol_pct = base_value / total_value

        # Example thresholds
        max_base_pct = Decimal("0.95")
        min_base_pct = Decimal("0.10")

        # Adjust buy/sell sizes to manage risk
        if sol_pct > max_base_pct:
            buy_size = Decimal("0")  # Don't buy more SOL
            sell_size = sell_size* Decimal("1.2")
        elif sol_pct < min_base_pct:
            sell_size = Decimal("0")  # Don't sell more SOL
            buy_size = buy_size * Decimal("1.2")
        
        return mean_price, bid_spread, ask_spread, buy_size, sell_size

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name, order.trading_pair, order.amount, order.order_type, order.price)
        else:
            self.buy(connector_name, order.trading_pair, order.amount, order.order_type, order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(self.config.exchange):
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} at {round(event.price, 2)}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Bot not ready."

        lines = [f"Strategy running on {self.config.trading_pair} ({self.config.exchange})"]

        # Balances
        balance_df = self.get_balance_df()
        lines += ["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")]

        # Orders
        try:
            df = self.active_orders_df()
            lines += ["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")]
        except ValueError:
            lines.append("  No active orders.")

        # Market insights
        try:
            df = self.get_candles_with_indicators()
            trend = self.compute_trend_state(df)
            volatility = self.compute_volatility_state(df)
            is_retracing, retrace_zone = self.detect_fibonacci_retracement(df)

            ref_price = self.connectors[self.config.exchange].get_price_by_type(self.config.trading_pair, self.price_source)
            mid_price = Decimal(str(ref_price))
            mean_price, bid_spread, ask_spread, buy_size, sell_size = self.adjust_parameters(
                trend, volatility, df, mid_price, is_retracing, retrace_zone
            )

            # Compute NATR and BB Width for debug
            natr = df["NATR_14"].iloc[-1]
            bb_width = (df["BBU_20_2.0"].iloc[-1] - df["BBL_20_2.0"].iloc[-1]) / df["close"].iloc[-1] * 100  # in %

            lines += [
                "",
                "  Market Signal Info:",
                f"    Trend: {trend}",
                f"    Volatility: {volatility}",
                f"    Is Retracing: {'Yes' if is_retracing else 'No'}",
                f"    Retrace Zone: {retrace_zone if is_retracing else 'N/A'}",
                f"    Mean Price: {round(mean_price, 4)}",
                f"    Bid Spread: {bid_spread * 100:.2f}%",
                f"    Ask Spread: {ask_spread * 100:.2f}%",
                f"    Buy Size: {buy_size}",
                f"    Sell Size: {sell_size}",
                f"    NATR (14): {natr:.2f}%",
                f"    BB Width: {bb_width:.2f}%"
            ]
        except Exception as e:
            lines.append(f"  [Error in status display: {str(e)}]")

        return "\n".join(lines)

