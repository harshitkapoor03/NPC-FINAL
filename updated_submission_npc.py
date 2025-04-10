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
    order_amount: Decimal = Field(0.1, client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Order amount"))
    order_refresh_time: int = Field(45, client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Order refresh time (in seconds)"))
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

    
    async def on_stop(self):
     self.logger().info("on_stop called.")
     if hasattr(self, "candles") and self.candles is not None:
         stop_method = getattr(self.candles, "stop", None)
         if stop_method:
             self.logger().info(f"Stopping candles. Async: {inspect.iscoroutinefunction(stop_method)}")
             if inspect.iscoroutinefunction(stop_method):
                 await stop_method()
             else:
                 stop_method()
         else:
             self.logger().warning("Candle stop method not found.")
     else:
         self.logger().warning("Candles not set.")


    
    def on_tick(self):
        if not self.ready_to_trade or not self.candles.ready:
            return

        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()

            candles_df = self.get_candles_with_indicators()
            trendscore, trend = self.compute_trend_state(candles_df)
            volatility = self.compute_volatility_state(candles_df)
            is_retracing, retrace_zone = self.detect_fibonacci_retracement(candles_df)
            skip_buy = skip_sell = False
            # Volatility-based throttling (BB Width)
            bb_width = (candles_df["BBU_20_2.0"].iloc[-1] - candles_df["BBL_20_2.0"].iloc[-1]) / candles_df["close"].iloc[-1]
            if bb_width < Decimal("0.001"):
                skip_buy=skip_sell=True

            ref_price = self.connectors[self.config.exchange].get_price_by_type(self.config.trading_pair, self.price_source)
            mid_price = Decimal(str(ref_price))
            mean_price, bid_spread, ask_spread, buy_size, sell_size = self.adjust_parameters(
                trend, trendscore, volatility, candles_df, mid_price, is_retracing, retrace_zone
            )

            best_bid = self.connectors[self.config.exchange].get_price(self.config.trading_pair, False)
            best_ask = self.connectors[self.config.exchange].get_price(self.config.trading_pair, True)

            # Avoid crossing the book
            buy_price_candidate = mean_price * (1 - bid_spread)
            sell_price_candidate = mean_price * (1 + ask_spread)

            buy_price = min(buy_price_candidate, Decimal(str(best_bid)) * Decimal("0.9999"))
            sell_price = max(sell_price_candidate, Decimal(str(best_ask)) * Decimal("1.0001"))

            # Minimum expected return filter
            actual_spread = (sell_price - buy_price) / mid_price
            min_expected_spread = Decimal("0.002")  # 0.2%
            if actual_spread < min_expected_spread:
                skip_buy=skip_sell=True

            # Trend filter: skip bad-side trades
            rsi = candles_df["RSI_14"].iloc[-1]
            if skip_buy==False:
             skip_buy = trend == "bearish" and rsi < 30
            if skip_sell==False:
             skip_sell = trend == "bullish" and rsi > 60

            proposal = []
            if not skip_buy:
                proposal.append(OrderCandidate(self.config.trading_pair, True, OrderType.LIMIT, TradeType.BUY, Decimal(buy_size), buy_price))
            if not skip_sell:
                proposal.append(OrderCandidate(self.config.trading_pair, True, OrderType.LIMIT, TradeType.SELL, Decimal(sell_size), sell_price))

            if proposal:
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


    def compute_trend_state(self, df: pd.DataFrame) -> tuple[float, str]:
        macd_line = df["MACD_12_26_9"].iloc[-1]
        macd_signal = df["MACDs_12_26_9"].iloc[-1]
        macd_line_prev = df["MACD_12_26_9"].iloc[-4]
        macd_signal_prev = df["MACDs_12_26_9"].iloc[-4]
        if macd_line<0 and macd_signal<0 and macd_line_prev<0 and macd_signal_prev<0:
           if macd_line>macd_signal and macd_line_prev<macd_signal_prev:
              return 0.7, "bullish"
        if macd_line>0 and macd_signal>0 and macd_line_prev>0 and macd_signal_prev>0:
           if macd_line<macd_signal and macd_line_prev>macd_signal_prev:
              return 0.3, "bearish"
        
        macd_diff = macd_line - macd_signal

    # Consider a neutral zone when MACD difference is small
        if abs(macd_diff) < 0.03:
           macd_cross = "neutral"
           macd_score = 0.5
        elif macd_diff > 0:
           macd_cross = "bullish"
           macd_score = 1.0
        else:
           macd_cross = "bearish"
           macd_score = 0.0
        rsi = df["RSI_14"].iloc[-1]
        close = df["close"].iloc[-1]
        lower_bb = df["BBL_20_2.0"].iloc[-1]
        upper_bb = df["BBU_20_2.0"].iloc[-1]
        bb_position = (close - lower_bb) / (upper_bb - lower_bb) if upper_bb > lower_bb else 0.5

        
        rsi_score = 0.0 if rsi < 25 else 0.375 if rsi < 50 else 0.625 if rsi < 75 else 1.0
        bb_score = 0.0 if bb_position < 0.2 else 1.0 if bb_position > 0.8 else 0.5

        if macd_cross == "bullish" and rsi_score == 0.0:
           return 0.7, "bullish"
        elif macd_cross == "bearish" and rsi_score == 1.0:
           return 0.3, "bearish"

        trend_score = 0.5 * macd_score + 0.3 * rsi_score + 0.2 * bb_score

        if trend_score >= 0.65:
           return trend_score, "bullish"
        elif trend_score <= 0.35:
           return trend_score, "bearish"
        else:
           return trend_score, "neutral"

    def compute_volatility_state(self, df: pd.DataFrame) -> str:
        natr = df["NATR_14"].iloc[-1]
        bb_width = ((df["BBU_20_2.0"].iloc[-1] - df["BBL_20_2.0"].iloc[-1]) / df["close"].iloc[-1])*100
        return "high" if (bb_width + natr) > 1.0 else "low"

    
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
        if candles_apart > 30 or candles_apart<8:
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
        ts1,past_trend = self.compute_trend_state(trend_window)

        # Step 6: Get current trend
        ts2,current_trend = self.compute_trend_state(df.tail(30))

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


    def adjust_parameters(self, trend, trendscore,volatility, df, mid_price, is_retracing, retrace_zone):
        bid_spread = ask_spread = Decimal("0.0015")
        buy_size = sell_size = self.config.order_amount
        mean_price = mid_price

        
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
            bid_spread = ask_spread = Decimal("0.0025")


        elif trend == "bullish":
            if volatility == "low":
                bid_spread = ask_spread = Decimal("0.0015")
                bid_spread = bid_spread * Decimal(1 - 0.1 * trendscore)  # trend_score: 0.7-1.0
                ask_spread = ask_spread * Decimal(1 + 0.1 * trendscore)
                mean_price = mid_price + (natr_offset * Decimal("0.4")*Decimal(trendscore))
                buy_size = self.config.order_amount * Decimal("1.1")
            else:
                bid_spread, ask_spread = (Decimal("0.0025"), Decimal("0.0045")) if broke_upper and volume_spike else (Decimal("0.0075"), Decimal("0.0045"))
                mean_price = mid_price + (natr_offset * Decimal("0.7")*Decimal(trendscore))
                if is_retracing:
                    mean_price = Decimal(str(df["close"].iloc[-1]))
                    buy_size = self.config.order_amount * Decimal("1.2")
                    sell_size = self.config.order_amount * Decimal("0.8")

        elif trend == "bearish":
            if volatility == "low":
                j=1-trendscore
                bid_spread = ask_spread = Decimal("0.0015")
                bid_spread = bid_spread * Decimal(1 - 0.1 * j)  # trend_score: 0.7-1.0
                ask_spread = ask_spread * Decimal(1 + 0.1 * j)
                mean_price = mid_price - (natr_offset * Decimal("0.4")*Decimal(j))
                sell_size = self.config.order_amount * Decimal("1.1")
            else:
                j=1-trendscore
                bid_spread, ask_spread = (Decimal("0.0045"), Decimal("0.0025")) if broke_lower and volume_spike else (Decimal("0.0045"), Decimal("0.0075"))
                mean_price = mid_price + (natr_offset * Decimal("0.7")*Decimal(j))
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
            trendscore,trend = self.compute_trend_state(df)
            volatility = self.compute_volatility_state(df)
            is_retracing, retrace_zone = self.detect_fibonacci_retracement(df)

            ref_price = self.connectors[self.config.exchange].get_price_by_type(self.config.trading_pair, self.price_source)
            mid_price = Decimal(str(ref_price))
            mean_price, bid_spread, ask_spread, buy_size, sell_size = self.adjust_parameters(
                trend,trendscore, volatility, df, mid_price, is_retracing, retrace_zone
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
                f"    Mid Price: {round(mid_price,4)}",
                f"    Mean Price: {round(mean_price, 4)}",
                f"    Bid Spread: {bid_spread * 100:.2f}%",
                f"    Ask Spread: {ask_spread * 100:.2f}%",
                f"    Buy Size: {buy_size}",
                f"    Sell Size: {sell_size}",
                f"    NATR (14): {natr:.2f}%",
                f"    BB Width: {bb_width:.2f}%",
                f"    trendscore: {trendscore:.2f}",
            ]
        except Exception as e:
            lines.append(f"  [Error in status display: {str(e)}]")

        return "\n".join(lines)
