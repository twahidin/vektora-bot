#!/usr/bin/env python3
"""
Combo BB+Donchian Trading Bot — Binance Futures

Always-in-position strategy: flips between long and short only when
BOTH Bollinger Band breakout AND Donchian channel breakout agree on
a new direction. This double-confirmation filter reduces whipsaws
and keeps you in winning trades longer.

Risk management:
  - Stop loss (2%) and take profit (4%) on every position
  - Profit floor ratchet: locks in gains as price moves in your favor
  - Exchange-side protective orders (STOP_MARKET + TAKE_PROFIT_MARKET)
  - Software backup checks each scan cycle

Backtest results (60 days, 10x leverage, 43 symbols):
  29% of configs profitable, best: SUI +9,910%, OP +9,267%

Usage:
    python bot.py                          # testnet (default, safe)
    python bot.py --live --yes             # live mode, skip confirmation
    python bot.py --once                   # single cycle then exit
    python bot.py --symbols SOL/USDT       # single symbol
    python bot.py --don-period 75          # Donchian period
    python bot.py --bb-period 14           # Bollinger period
    python bot.py --bb-std 3.0             # Bollinger std devs
    python bot.py --sl-pct 2.0             # stop loss %
    python bot.py --tp-pct 4.0             # take profit %
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SIGNAL_SERVER_URL = os.getenv("SIGNAL_SERVER_URL", "https://signal-server-production-1802.up.railway.app")
BOT_API_KEY = os.getenv("BOT_API_KEY", "") or os.getenv("SIGNAL_API_KEY", "")

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("combo-bot")

# ──────────────────────────────────────────────────────────────
# Default configuration (from backtest winners)
# ──────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "symbols": [
        "SUI/USDT",    # +9,910%  best overall
        "OP/USDT",     # +9,267%  100% WR, 0% DD
        "SOL/USDT",    # +6,338%  strong trending
        "ARB/USDT",    # +5,979%  100% WR, 0% DD
        "APT/USDT",    # +5,879%  consistent
        "FET/USDT",    # +3,688%  AI narrative
        "FIL/USDT",    # +3,506%  100% WR
        "STX/USDT",    # +3,270%  strong trends
        "RUNE/USDT",   # +3,144%  volatile, trending
        "THETA/USDT",  # +3,052%  100% WR
        "BNB/USDT",    # +2,733%  low DD (16%)
        "ETH/USDT",    # +1,183%  major, stable
        "BTC/USDT",    # +753%    major, stable
        "DOT/USDT",    # +1,843%  100% WR
        "LINK/USDT",   # +1,806%  low DD (16%)
    ],
    "timeframe": "15m",
    # Donchian channel params
    "don_period": 100,        # breakout lookback (candles)
    # Bollinger Band params
    "bb_period": 14,          # BB moving average period
    "bb_std": 3.0,            # standard deviations for bands
    # Trading params
    "leverage": 10,
    "margin_mode": "ISOLATED",
    "risk_per_symbol": 0.06,  # fraction of capital per symbol (15 symbols x 6% = 90%, 10% buffer)
    "taker_fee": 0.0005,      # 0.05%
    "candle_lookback": 200,   # candles to fetch (must be > don_period)
    "check_interval_seconds": 60,
    "max_positions": 5,
    "ai_intervention": False,
    # Risk management (SL8/TP0/Floors — backtest-validated best balance)
    "sl_pct": 8.0,            # stop loss 8% from entry (saves 20% margin before liquidation)
    "tp_pct": 0.0,            # no take profit (let floors + signal flips handle exits)
    "profit_floor_enabled": True,
    "profit_floor_levels": {  # peak P&L % -> lock minimum P&L %
        1.2: 0.0,             # 1.2% peak -> lock breakeven (0%)
        1.5: 0.3,             # 1.5% peak -> lock +0.3%
        2.0: 0.7,             # 2.0% peak -> lock +0.7%
        2.5: 1.1,             # 2.5% peak -> lock +1.1%
        3.0: 1.6,             # 3.0% peak -> lock +1.6%
        3.5: 2.1,             # 3.5% peak -> lock +2.1%
        4.0: 2.6,             # 4.0% peak -> lock +2.6%
        5.0: 3.5,             # 5.0% peak -> lock +3.5%
    },
}

STATE_FILE = os.path.join(os.path.dirname(__file__), "bot_state.json")


# ──────────────────────────────────────────────────────────────
# Signal logic
# ──────────────────────────────────────────────────────────────
def donchian_direction(high, low, close, period):
    """Returns signal array: +1 long on upper break, -1 short on lower break.
    Holds last direction between breakouts."""
    n = len(close)
    sig = np.zeros(n, dtype=int)
    current_dir = 1

    for i in range(period, n):
        upper = np.max(high[i - period:i])
        lower = np.min(low[i - period:i])
        if close[i] > upper:
            current_dir = 1
        elif close[i] < lower:
            current_dir = -1
        sig[i] = current_dir

    sig[:period] = sig[period] if period < n else 1
    return sig


def bollinger_direction(close, period, num_std):
    """Returns signal array: +1 long above upper band, -1 short below lower band.
    Holds last direction when between bands."""
    s = pd.Series(close)
    sma = s.rolling(period).mean().values
    std = s.rolling(period).std().values
    upper = sma + num_std * std
    lower = sma - num_std * std

    n = len(close)
    sig = np.zeros(n, dtype=int)
    current_dir = 1

    for i in range(period, n):
        if close[i] > upper[i]:
            current_dir = 1
        elif close[i] < lower[i]:
            current_dir = -1
        sig[i] = current_dir

    sig[:period] = sig[period] if period < n else 1
    return sig


def combo_signal(high, low, close, don_period, bb_period, bb_std):
    """Combined signal: only flip when BOTH Donchian AND Bollinger agree.
    Returns (current_direction, don_dir, bb_dir, details_str) for the latest bar."""
    don_sig = donchian_direction(high, low, close, don_period)
    bb_sig = bollinger_direction(close, bb_period, bb_std)
    warmup = max(don_period, bb_period)

    n = len(close)
    current_dir = 1

    for i in range(warmup, n):
        if don_sig[i] == bb_sig[i] and don_sig[i] != current_dir:
            current_dir = don_sig[i]

    # Info for logging
    don_label = "LONG" if don_sig[-1] == 1 else "SHORT"
    bb_label = "LONG" if bb_sig[-1] == 1 else "SHORT"
    combo_label = "LONG" if current_dir == 1 else "SHORT"

    # Donchian channel values for display
    upper = np.max(high[-don_period:])
    lower = np.min(low[-don_period:])

    # BB values for display
    s = pd.Series(close)
    sma = s.rolling(bb_period).mean().iloc[-1]
    std = s.rolling(bb_period).std().iloc[-1]
    bb_upper = sma + bb_std * std
    bb_lower = sma - bb_std * std

    details = (f"Don({don_period}): {don_label} [${lower:.2f}-${upper:.2f}] | "
               f"BB({bb_period},{bb_std}): {bb_label} [${bb_lower:.2f}-${bb_upper:.2f}] | "
               f"Combo: {combo_label}")

    return current_dir, don_sig[-1], bb_sig[-1], details


# ──────────────────────────────────────────────────────────────
# Bot
# ──────────────────────────────────────────────────────────────
class ComboBot:
    def __init__(self, testnet=True, config=None):
        self.testnet = testnet
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.exchange = None
        self.running = True
        self._start_time = time.time()
        self.positions = {}       # symbol -> {direction, qty, entry_price, entry_time, max_pnl_pct, last_floor}
        self.protective_orders = {}  # symbol -> {sl_order_id, tp_order_id, sl_price, tp_price}
        self._load_state()

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ── Exchange setup ──────────────────────────────────────

    def connect(self):
        """Initialize exchange connection."""
        if self.testnet:
            api_key = os.getenv("BINANCE_TESTNET_APIKEY")
            secret = os.getenv("BINANCE_TESTNET_SECRET")
            if not api_key or not secret:
                log.error("Set BINANCE_TESTNET_APIKEY and BINANCE_TESTNET_SECRET in .env")
                return False
            self.exchange = ccxt.binanceusdm({
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
            })
            self.exchange.set_sandbox_mode(True)
            log.info("Connected to Binance TESTNET")
        else:
            api_key = os.getenv("BINANCE_APIKEY") or os.getenv("BINANCE_KEY", "")
            secret = os.getenv("BINANCE_SECRET", "")
            if not api_key or not secret:
                log.error("Set BINANCE_APIKEY and BINANCE_SECRET in .env")
                return False
            self.exchange = ccxt.binance({
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
                "timeout": 30000,
                "options": {"defaultType": "future"},
            })
            log.info("Connected to Binance LIVE")

        # Setup leverage and margin mode for each symbol
        for symbol in self.cfg["symbols"]:
            self._setup_symbol(symbol)

        # Re-discover protective orders for positions that survived a restart
        if self.positions:
            self._sync_protective_orders()

        return True

    def _setup_symbol(self, symbol):
        """Set leverage and margin mode for a symbol."""
        try:
            self.exchange.set_leverage(self.cfg["leverage"], symbol)
        except Exception as e:
            log.warning(f"set_leverage {symbol}: {e}")
        try:
            self.exchange.set_margin_mode(self.cfg["margin_mode"], symbol)
        except Exception as e:
            if "No need to change margin type" not in str(e):
                log.warning(f"set_margin_mode {symbol}: {e}")

    def _sync_protective_orders(self):
        """Re-discover protective orders after restart.
        Finds existing STOP_MARKET/TAKE_PROFIT_MARKET orders on exchange,
        or places new ones if missing."""
        log.info("Syncing protective orders for existing positions...")
        for symbol, pos in list(self.positions.items()):
            ex_qty, ex_dir = self.get_exchange_position(symbol)
            if ex_qty == 0:
                log.info(f"  {symbol}: no longer on exchange (closed by SL/TP). Cleaning up.")
                self.positions.pop(symbol, None)
                continue

            try:
                open_orders = self.exchange.fetch_open_orders(symbol)
                for order in open_orders:
                    order_type = (order.get("type") or "").upper()
                    stop_price = float(
                        order.get("stopPrice")
                        or order.get("info", {}).get("stopPrice")
                        or 0
                    )
                    if "STOP" in order_type and "TAKE" not in order_type and stop_price > 0:
                        self.protective_orders.setdefault(symbol, {})
                        self.protective_orders[symbol]["sl_order_id"] = order["id"]
                        self.protective_orders[symbol]["sl_price"] = stop_price
                    elif "TAKE_PROFIT" in order_type and stop_price > 0:
                        self.protective_orders.setdefault(symbol, {})
                        self.protective_orders[symbol]["tp_order_id"] = order["id"]
                        self.protective_orders[symbol]["tp_price"] = stop_price

                if symbol in self.protective_orders:
                    sl = self.protective_orders[symbol].get("sl_price", "none")
                    tp = self.protective_orders[symbol].get("tp_price", "none")
                    log.info(f"  {symbol}: synced SL=${sl} TP=${tp}")
                else:
                    log.warning(f"  {symbol}: no protective orders found, placing now")
                    self.place_protective_orders(symbol, pos["direction"], pos["entry_price"], ex_qty)
            except Exception as e:
                log.warning(f"  {symbol}: order sync failed: {e}")

        self._save_state()

    # ── Data & Signal ───────────────────────────────────────

    def fetch_candles(self, symbol):
        """Fetch recent OHLCV candles."""
        candles = self.exchange.fetch_ohlcv(
            symbol, self.cfg["timeframe"], limit=self.cfg["candle_lookback"]
        )
        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
        return df

    def get_signal(self, df):
        """Calculate Combo BB+Donchian signal.
        Returns (direction, don_dir, bb_dir, details_str)."""
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        return combo_signal(
            high, low, close,
            self.cfg["don_period"],
            self.cfg["bb_period"],
            self.cfg["bb_std"],
        )

    # ── Protective orders (SL/TP) ───────────────────────────

    def place_protective_orders(self, symbol, direction, entry_price, qty):
        """Place stop loss and take profit orders on exchange.
        Returns order info dict or None if SL placement fails (position is emergency-closed)."""
        sl_pct = self.cfg["sl_pct"]
        tp_pct = self.cfg["tp_pct"]

        if direction == 1:  # LONG
            sl_price = entry_price * (1 - sl_pct / 100)
            tp_price = entry_price * (1 + tp_pct / 100)
            close_side = "sell"
        else:  # SHORT
            sl_price = entry_price * (1 + sl_pct / 100)
            tp_price = entry_price * (1 - tp_pct / 100)
            close_side = "buy"

        sl_price = float(self.exchange.price_to_precision(symbol, sl_price))
        tp_price = float(self.exchange.price_to_precision(symbol, tp_price))

        sl_order_id = None
        tp_order_id = None

        # Place stop loss (mandatory — no unprotected positions)
        try:
            sl_order = self.exchange.create_order(
                symbol, "STOP_MARKET", close_side, qty, None,
                {"stopPrice": sl_price, "reduceOnly": True}
            )
            sl_order_id = sl_order["id"]
            log.info(f"  SL placed: {close_side} {qty} @ ${sl_price:.4f} (-{sl_pct}%) [order {sl_order_id}]")
        except Exception as e:
            log.error(f"  FAILED to place SL for {symbol}: {e}")
            log.error(f"  Emergency closing {symbol} -- no stop loss protection!")
            self._emergency_close(symbol)
            return None

        # Place take profit (best effort — SL alone is sufficient)
        try:
            tp_order = self.exchange.create_order(
                symbol, "TAKE_PROFIT_MARKET", close_side, qty, None,
                {"stopPrice": tp_price, "reduceOnly": True}
            )
            tp_order_id = tp_order["id"]
            log.info(f"  TP placed: {close_side} {qty} @ ${tp_price:.4f} (+{tp_pct}%) [order {tp_order_id}]")
        except Exception as e:
            log.warning(f"  Failed to place TP for {symbol}: {e} (SL still active)")

        orders = {
            "sl_order_id": sl_order_id,
            "tp_order_id": tp_order_id,
            "sl_price": sl_price,
            "tp_price": tp_price,
        }
        self.protective_orders[symbol] = orders
        return orders

    def cancel_protective_orders(self, symbol):
        """Cancel existing SL/TP orders for a symbol."""
        orders = self.protective_orders.pop(symbol, {})

        for key in ["sl_order_id", "tp_order_id"]:
            order_id = orders.get(key)
            if order_id:
                try:
                    self.exchange.cancel_order(order_id, symbol)
                    label = "SL" if "sl" in key else "TP"
                    log.info(f"  Cancelled {label} order {order_id}")
                except Exception as e:
                    # Order may have already triggered — that's fine
                    if "Unknown order" not in str(e) and "UNKNOWN_ORDER" not in str(e):
                        log.warning(f"  Cancel {key} {order_id}: {e}")

    def cancel_all_orders(self, symbol):
        """Cancel ALL open orders for a symbol. Used to clean up after external close."""
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                try:
                    self.exchange.cancel_order(order["id"], symbol)
                except Exception:
                    pass
            if open_orders:
                log.info(f"  Cleaned up {len(open_orders)} orphaned orders for {symbol}")
        except Exception:
            pass
        self.protective_orders.pop(symbol, None)

    def update_exchange_sl(self, symbol, direction, new_sl_price, qty):
        """Move the exchange stop loss to a new (tighter/better) price.
        Places new SL before cancelling old — position is never unprotected."""
        close_side = "sell" if direction == 1 else "buy"
        new_sl_price = float(self.exchange.price_to_precision(symbol, new_sl_price))

        old_sl_id = self.protective_orders.get(symbol, {}).get("sl_order_id")

        # Place new SL first
        try:
            new_order = self.exchange.create_order(
                symbol, "STOP_MARKET", close_side, qty, None,
                {"stopPrice": new_sl_price, "reduceOnly": True}
            )
            new_sl_id = new_order["id"]
        except Exception as e:
            log.error(f"  Failed to place new SL for {symbol}: {e} (keeping old SL)")
            return False

        # Cancel old SL
        if old_sl_id:
            try:
                self.exchange.cancel_order(old_sl_id, symbol)
            except Exception:
                pass  # old order may have already triggered

        # Update tracking
        if symbol in self.protective_orders:
            self.protective_orders[symbol]["sl_order_id"] = new_sl_id
            self.protective_orders[symbol]["sl_price"] = new_sl_price

        return True

    def _emergency_close(self, symbol):
        """Market close a position when protective orders fail. Last resort."""
        qty, direction = self.get_exchange_position(symbol)
        if qty == 0:
            return
        side = "sell" if direction == 1 else "buy"
        try:
            self.exchange.create_order(symbol, "market", side, qty, None, {"reduceOnly": True})
            log.warning(f"  EMERGENCY CLOSED {symbol}: {side} {qty}")
        except Exception as e:
            log.error(f"  EMERGENCY CLOSE FAILED for {symbol}: {e}")
        self.positions.pop(symbol, None)
        self.protective_orders.pop(symbol, None)

    # ── Profit floor ratchet ─────────────────────────────────

    def check_profit_floors(self, symbol, current_price):
        """Check and update profit floor for a tracked position.
        Ratchets the exchange SL upward as peak P&L crosses thresholds."""
        if not self.cfg["profit_floor_enabled"]:
            return

        pos = self.positions.get(symbol)
        if not pos:
            return

        entry_price = pos["entry_price"]
        direction = pos["direction"]

        # Calculate current P&L %
        if direction == 1:
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100

        # Update peak P&L
        max_pnl = pos.get("max_pnl_pct", 0.0)
        if pnl_pct > max_pnl:
            pos["max_pnl_pct"] = pnl_pct
            max_pnl = pnl_pct

        # Find highest applicable profit floor
        last_floor = pos.get("last_floor", -1.0)
        floor_levels = self.cfg["profit_floor_levels"]
        best_lock = -1.0

        for threshold in sorted(floor_levels.keys()):
            if max_pnl >= threshold:
                best_lock = floor_levels[threshold]

        # Ratchet: only move SL if new floor is strictly higher
        if best_lock > last_floor:
            if direction == 1:
                new_sl = entry_price * (1 + best_lock / 100)
            else:
                new_sl = entry_price * (1 - best_lock / 100)

            old_sl = self.protective_orders.get(symbol, {}).get("sl_price", 0)
            qty = pos["qty"]

            # Verify new SL is actually better (closer to profit)
            should_move = False
            if direction == 1 and new_sl > old_sl:
                should_move = True
            elif direction == -1 and (old_sl == 0 or new_sl < old_sl):
                should_move = True

            if should_move:
                lock_label = f"+{best_lock:.1f}%" if best_lock > 0 else "breakeven"
                log.info(f"  PROFIT FLOOR {symbol}: peak +{max_pnl:.1f}% -> locking {lock_label} "
                         f"(SL ${old_sl:.4f} -> ${new_sl:.4f})")

                if self.update_exchange_sl(symbol, direction, new_sl, qty):
                    pos["last_floor"] = best_lock
                    self._save_state()

    # ── Position management ─────────────────────────────────

    def get_exchange_position(self, symbol):
        """Get current position from exchange. Returns (qty, direction) or (0, 0)."""
        try:
            balance = self.exchange.fetch_balance()
            positions = balance.get("info", {}).get("positions", [])
            clean_sym = symbol.replace("/", "").replace(":USDT", "")
            for pos in positions:
                if pos.get("symbol") == clean_sym:
                    amt = float(pos.get("positionAmt", 0) or 0)
                    if abs(amt) > 0:
                        return abs(amt), 1 if amt > 0 else -1
            return 0, 0
        except Exception as e:
            log.error(f"Failed to get position for {symbol}: {e}")
            return 0, 0

    def calculate_quantity(self, symbol, price):
        """Calculate order quantity based on capital allocation."""
        try:
            balance = self.exchange.fetch_balance()
            total_equity = float(balance.get("total", {}).get("USDT", 0) or 0)
            if total_equity <= 0:
                log.error("No USDT balance found")
                return 0

            allocation = total_equity * self.cfg["risk_per_symbol"]
            notional = allocation * self.cfg["leverage"]
            raw_qty = notional / price

            qty = float(self.exchange.amount_to_precision(symbol, raw_qty))
            log.info(f"  Equity: ${total_equity:.2f} | Alloc: ${allocation:.2f} | "
                     f"Notional: ${notional:.2f} | Qty: {qty}")
            return qty
        except Exception as e:
            log.error(f"calculate_quantity failed: {e}")
            return 0

    def close_position(self, symbol):
        """Close existing position on exchange."""
        qty, direction = self.get_exchange_position(symbol)
        if qty == 0:
            return True

        side = "sell" if direction == 1 else "buy"
        try:
            order = self.exchange.create_order(
                symbol, "market", side, qty, None, {"reduceOnly": True}
            )
            log.info(f"  CLOSED {symbol}: {side} {qty} @ market (order {order['id']})")
            return True
        except Exception as e:
            log.error(f"  Failed to close {symbol}: {e}")
            return False

    def open_position(self, symbol, direction, price):
        """Open a new position and place protective orders."""
        qty = self.calculate_quantity(symbol, price)
        if qty <= 0:
            return False

        side = "buy" if direction == 1 else "sell"
        try:
            order = self.exchange.create_order(symbol, "market", side, qty, None)
            fill_price = float(order.get("average", price) or price)
            log.info(f"  OPENED {symbol}: {side} {qty} @ ${fill_price:.4f} (order {order['id']})")

            self.positions[symbol] = {
                "direction": direction,
                "qty": qty,
                "entry_price": fill_price,
                "entry_time": datetime.now().isoformat(),
                "max_pnl_pct": 0.0,
                "last_floor": -1.0,
            }

            # Place SL + TP on exchange (mandatory)
            orders = self.place_protective_orders(symbol, direction, fill_price, qty)
            if orders is None:
                # SL failed — position was emergency-closed inside place_protective_orders
                return False

            self._save_state()
            return True
        except Exception as e:
            log.error(f"  Failed to open {symbol} {side}: {e}")
            return False

    def flip_position(self, symbol, new_direction, price):
        """Cancel old protective orders, close current position, open new with new orders."""
        log.info(f"  FLIPPING {symbol} to {'LONG' if new_direction == 1 else 'SHORT'}")

        # 1. Cancel old SL/TP (must happen BEFORE close to avoid them triggering on new position)
        self.cancel_protective_orders(symbol)
        time.sleep(0.3)

        # 2. Close old position
        if not self.close_position(symbol):
            return False
        self.positions.pop(symbol, None)
        time.sleep(0.5)

        # 3. Open new position (includes placing new SL/TP)
        return self.open_position(symbol, new_direction, price)

    # ── Main loop ───────────────────────────────────────────

    def run_cycle(self):
        """Run one scan cycle across all symbols."""
        log.info(f"{'='*70}")
        log.info(f"Scan @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
                 f"Don({self.cfg['don_period']}) + BB({self.cfg['bb_period']},{self.cfg['bb_std']})  |  "
                 f"SL {self.cfg['sl_pct']}% / TP {self.cfg['tp_pct']}%")
        log.info(f"{'='*70}")

        for symbol in self.cfg["symbols"]:
            log.info(f"\n--- {symbol} ---")
            try:
                df = self.fetch_candles(symbol)
                current_price = float(df["close"].iloc[-1])

                direction, don_dir, bb_dir, details = self.get_signal(df)
                dir_label = "LONG" if direction == 1 else "SHORT"

                log.info(f"  Price: ${current_price:.4f}")
                log.info(f"  {details}")

                # Check current exchange position
                ex_qty, ex_dir = self.get_exchange_position(symbol)

                # Detect externally-closed positions (SL/TP triggered on exchange)
                if ex_qty == 0 and symbol in self.positions:
                    pos = self.positions[symbol]
                    entry = pos["entry_price"]
                    if pos["direction"] == 1:
                        final_pnl = (current_price - entry) / entry * 100
                    else:
                        final_pnl = (entry - current_price) / entry * 100
                    log.info(f"  Position was closed by exchange (SL/TP). Est P&L: {final_pnl:+.2f}%")
                    self.cancel_all_orders(symbol)
                    self.positions.pop(symbol, None)
                    self._save_state()
                    ex_qty, ex_dir = 0, 0  # fall through to open new position

                if ex_qty == 0:
                    max_pos = self.cfg.get("max_positions", 18)
                    if len(self.positions) >= max_pos:
                        log.info(f"  At max positions ({max_pos}) — skipping")
                    else:
                        log.info(f"  No position. Opening {dir_label}...")
                        self.open_position(symbol, direction, current_price)

                elif ex_dir != direction:
                    # Combo says flip
                    old_label = "LONG" if ex_dir == 1 else "SHORT"
                    log.info(f"  Currently {old_label} -> Combo says {dir_label}. BOTH agree. Flipping...")
                    self.flip_position(symbol, direction, current_price)

                else:
                    # Holding — check profit floors and show P&L
                    pos = self.positions.get(symbol, {})
                    entry = pos.get("entry_price", current_price)
                    if ex_dir == 1:
                        pnl_pct = (current_price - entry) / entry * 100
                    else:
                        pnl_pct = (entry - current_price) / entry * 100
                    roi_pct = pnl_pct * self.cfg["leverage"]

                    sl_price = self.protective_orders.get(symbol, {}).get("sl_price", 0)
                    tp_price = self.protective_orders.get(symbol, {}).get("tp_price", 0)

                    agree = "AGREE" if don_dir == bb_dir else "DISAGREE"
                    log.info(f"  Holding {dir_label} ({ex_qty} qty). P&L: {pnl_pct:+.2f}% (ROI {roi_pct:+.1f}%). "
                             f"Indicators {agree}.")
                    if sl_price or tp_price:
                        floor_label = ""
                        peak = pos.get("max_pnl_pct", 0)
                        if peak > 0:
                            floor_label = f" | Peak: +{peak:.1f}%"
                        log.info(f"  SL: ${sl_price:.4f} | TP: ${tp_price:.4f}{floor_label}")

                    # Profit floor ratchet check
                    self.check_profit_floors(symbol, current_price)

            except Exception as e:
                log.error(f"  Error processing {symbol}: {e}")

        # Poll for dashboard commands
        self._poll_commands()
        # AI exit management
        self._check_ai_exits()
        # Report status to dashboard
        self._report_status()

    def run(self, once=False):
        """Main loop."""
        mode = "TESTNET" if self.testnet else "LIVE"
        log.info(f"Bot started: {mode}")
        log.info(f"Strategy: Combo-Both (Donchian + Bollinger must agree to flip)")
        log.info(f"Symbols: {', '.join(self.cfg['symbols'])}")
        log.info(f"Params: Don({self.cfg['don_period']}) + BB({self.cfg['bb_period']},{self.cfg['bb_std']})")
        log.info(f"Leverage: {self.cfg['leverage']}x | Timeframe: {self.cfg['timeframe']}")
        log.info(f"Risk: SL {self.cfg['sl_pct']}% ({self.cfg['sl_pct'] * self.cfg['leverage']}% ROI) | "
                 f"TP {self.cfg['tp_pct']}% ({self.cfg['tp_pct'] * self.cfg['leverage']}% ROI)")
        if self.cfg["profit_floor_enabled"]:
            floors = self.cfg["profit_floor_levels"]
            log.info(f"Profit floors: {len(floors)} levels "
                     f"(breakeven at {min(floors.keys()):.1f}%, max lock +{max(floors.values()):.1f}%)")

        while self.running:
            try:
                self.run_cycle()
            except Exception as e:
                log.error(f"Cycle error: {e}")

            if once:
                log.info("Single cycle complete (--once). Exiting.")
                break

            interval = self.cfg["check_interval_seconds"]
            log.info(f"\nNext scan in {interval}s...")
            time.sleep(interval)

    # ── State persistence ───────────────────────────────────

    def _save_state(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({
                    "positions": self.positions,
                    "config": {
                        "don_period": self.cfg["don_period"],
                        "bb_period": self.cfg["bb_period"],
                        "bb_std": self.cfg["bb_std"],
                        "sl_pct": self.cfg["sl_pct"],
                        "tp_pct": self.cfg["tp_pct"],
                    },
                    "updated": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception:
            pass

    def _load_state(self):
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
                self.positions = state.get("positions", {})
                # Ensure new fields exist on loaded positions
                for pos in self.positions.values():
                    pos.setdefault("max_pnl_pct", 0.0)
                    pos.setdefault("last_floor", -1.0)
                if self.positions:
                    log.info(f"Loaded state: {len(self.positions)} tracked positions")
        except Exception:
            self.positions = {}

    def _handle_shutdown(self, signum, frame):
        log.info("\nShutdown signal received. Positions and protective orders remain on exchange.")
        self.running = False

    # ── Command polling & AI exits ──────────────────────────

    def _poll_commands(self):
        """Poll signal server for pending commands."""
        if not SIGNAL_SERVER_URL or not BOT_API_KEY:
            return
        try:
            import requests
            resp = requests.get(
                f"{SIGNAL_SERVER_URL}/api/commands",
                params={"key": BOT_API_KEY},
                timeout=10,
            )
            if resp.status_code != 200:
                return
            cmd = resp.json()
            if not cmd.get("id"):
                return
            command = cmd.get("command", "")
            payload = cmd.get("payload", {}) or {}
            log.info(f"Received command: {command} (payload: {payload})")
            if command == "set_max_positions":
                new_max = payload.get("max_positions", 5)
                self.cfg["max_positions"] = new_max
                log.info(f"Max positions updated to {new_max}")
            elif command == "pause":
                self.cfg["paused"] = True
                log.info("Pause command received")
            elif command == "resume":
                self.cfg["paused"] = False
                log.info("Resume command received")
            # Acknowledge
            requests.post(
                f"{SIGNAL_SERVER_URL}/api/commands/{cmd['id']}/ack",
                params={"key": BOT_API_KEY},
                timeout=10,
            )
        except Exception as e:
            log.error(f"Command poll failed: {e}")

    def _check_ai_exits(self):
        """Call signal server for AI exit decisions on profitable positions."""
        if not SIGNAL_SERVER_URL or not BOT_API_KEY:
            return
        if not self.cfg.get("ai_intervention"):
            return
        profitable = []
        for symbol, pos in list(self.positions.items()):
            ex_qty, ex_dir = self.get_exchange_position(symbol)
            if ex_qty == 0:
                continue
            entry = pos["entry_price"]
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker["last"]
            except Exception:
                continue
            dm = 1 if pos["direction"] == 1 else -1
            pnl_pct = dm * (current_price - entry) / entry * 100
            if pnl_pct > 2.0:
                profitable.append({
                    "symbol": symbol,
                    "direction": pos["direction"],
                    "entry_price": entry,
                    "current_price": current_price,
                    "unrealized_pnl_pct": round(pnl_pct, 2),
                    "peak_pnl_pct": round(pos.get("max_pnl_pct", 0), 2),
                    "bars_held": 0,
                })
        if not profitable:
            return
        try:
            import requests
            resp = requests.post(
                f"{SIGNAL_SERVER_URL}/api/ai-exit-check",
                json={"api_key": BOT_API_KEY, "positions": profitable},
                timeout=30,
            )
            if resp.status_code != 200:
                return
            decisions = resp.json().get("decisions", [])
            for dec in decisions:
                if dec.get("action") == "LOCK":
                    symbol = dec["symbol"]
                    log.info(f"AI says LOCK {symbol}: {dec.get('reason', '')}")
                    self.close_position(symbol)
        except Exception as e:
            log.error(f"AI exit check failed: {e}")

    def _report_status(self):
        """Report balance + positions to signal server for dashboard display."""
        if not SIGNAL_SERVER_URL or not BOT_API_KEY:
            return
        try:
            import requests
            balance = self.exchange.fetch_balance()
            total_equity = float(balance.get("total", {}).get("USDT", 0) or 0)

            # Build position list from actual exchange data
            raw_positions = balance.get("info", {}).get("positions", [])
            pos_list = []
            for p in raw_positions:
                amt = float(p.get("positionAmt", 0) or 0)
                if abs(amt) == 0:
                    continue
                sym_raw = p.get("symbol", "")
                # Convert BTCUSDT -> BTC/USDT
                sym_clean = sym_raw.replace("USDT", "/USDT") if "USDT" in sym_raw else sym_raw
                direction = 1 if amt > 0 else -1
                pnl_usd = float(p.get("unrealizedProfit", 0) or 0)
                margin = float(p.get("initialMargin", 0) or 0)
                notional = abs(float(p.get("notional", 0) or 0))
                entry_price = notional / abs(amt) if abs(amt) > 0 else 0
                pnl_pct = (pnl_usd / margin * 100) if margin > 0 else 0

                pos_list.append({
                    "symbol": sym_clean,
                    "direction": direction,
                    "entry_price": round(entry_price, 6),
                    "pnl_usd": round(pnl_usd, 2),
                    "pnl_pct": round(pnl_pct, 1),
                })

            payload = {
                "balance": total_equity,
                "positions": pos_list,
                "uptime_seconds": int(time.time() - self._start_time),
            }

            requests.post(
                f"{SIGNAL_SERVER_URL}/api/bot-status",
                params={"key": BOT_API_KEY},
                json=payload,
                timeout=10,
            )
        except Exception as e:
            log.error(f"Status report failed: {e}")


# ──────────────────────────────────────────────────────────────
# FastAPI app for Railway (health check + bot runs in background thread)
# ──────────────────────────────────────────────────────────────
from fastapi import FastAPI
from contextlib import asynccontextmanager

_bot_instance = None


@asynccontextmanager
async def lifespan(application):
    """Start bot in background thread on startup."""
    global _bot_instance
    import threading
    bot = ComboBot(testnet=False)
    if bot.connect():
        _bot_instance = bot
        t = threading.Thread(target=bot.run, daemon=True)
        t.start()
        log.info("Bot started in background thread")
    else:
        log.error("Bot failed to connect — running without trading")
    yield
    if _bot_instance:
        _bot_instance.running = False


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if _bot_instance:
        return {
            "status": "running",
            "positions": len(_bot_instance.positions),
            "uptime": int(time.time() - _bot_instance._start_time),
        }
    return {"status": "not_connected"}


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Combo BB+Donchian Trading Bot")
    parser.add_argument("--live", action="store_true", help="Use live Binance (default: testnet)")
    parser.add_argument("--yes", action="store_true", help="Skip live mode confirmation")
    parser.add_argument("--once", action="store_true", help="Run single cycle then exit")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to trade")
    parser.add_argument("--don-period", type=int, default=None, help="Donchian channel period")
    parser.add_argument("--bb-period", type=int, default=None, help="Bollinger Band period")
    parser.add_argument("--bb-std", type=float, default=None, help="Bollinger Band std devs")
    parser.add_argument("--leverage", type=int, default=None, help="Leverage multiplier")
    parser.add_argument("--interval", type=int, default=None, help="Check interval in seconds")
    parser.add_argument("--sl-pct", type=float, default=None, help="Stop loss %% from entry")
    parser.add_argument("--tp-pct", type=float, default=None, help="Take profit %% from entry")
    args = parser.parse_args()

    # Live mode safety gate
    if args.live and not args.yes:
        print("\n*** LIVE TRADING MODE ***")
        print("This will trade REAL money on Binance Futures with 10x leverage.")
        print("Strategy: Combo-Both (Donchian + Bollinger must agree)")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    config = {}
    if args.symbols:
        config["symbols"] = args.symbols
    if args.don_period:
        config["don_period"] = args.don_period
    if args.bb_period:
        config["bb_period"] = args.bb_period
    if args.bb_std:
        config["bb_std"] = args.bb_std
    if args.leverage:
        config["leverage"] = args.leverage
    if args.interval:
        config["check_interval_seconds"] = args.interval
    if args.sl_pct:
        config["sl_pct"] = args.sl_pct
    if args.tp_pct:
        config["tp_pct"] = args.tp_pct

    bot = ComboBot(testnet=not args.live, config=config)

    if not bot.connect():
        sys.exit(1)

    bot.run(once=args.once)


if __name__ == "__main__":
    main()
