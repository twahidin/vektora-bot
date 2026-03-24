#!/usr/bin/env python3
"""
Vektora Client Bot — Signal Consumer + Trade Executor

Connects to the Vektora signal server via WebSocket, receives Combo BB+Donchian
direction flips, and executes Binance futures trades via the Vektora Binance proxy.
Risk management: SL + profit floor ratchet.

Designed for self-service deployment on Railway.
"""

import asyncio
import hmac
import json
import logging
import os
import stat
import time
from datetime import datetime

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("vektora-bot")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
STATE_FILE = os.path.join(DATA_DIR, "bot_state.json")
CREDS_FILE = os.path.join(DATA_DIR, "credentials.json")

# ──────────────────────────────────────────────────────────────
# Trading config (hardcoded — matches signal server)
# ──────────────────────────────────────────────────────────────
SYMBOLS = [
    "SUI/USDT", "OP/USDT", "SOL/USDT", "ARB/USDT", "APT/USDT",
    "FET/USDT", "FIL/USDT", "STX/USDT", "RUNE/USDT", "THETA/USDT",
    "BNB/USDT", "ETH/USDT", "BTC/USDT", "DOT/USDT", "LINK/USDT",
    "SAND/USDT", "XLM/USDT", "LTC/USDT",
]
LEVERAGE = 10
SL_PCT = 8.0
RISK_PER_TRADE_PCT = 5.0
MAX_POSITIONS = 15
MIN_HOLD_MINUTES = 60
FLOOR_LEVELS = {}  # No floors — ride signal flips, SL as safety net only

SIGNAL_SERVER_URL = os.environ.get("SIGNAL_SERVER_URL", "wss://signal-server-production-1802.up.railway.app")
PROXY_URL = os.environ.get("PROXY_URL", "")  # fetched from signal server if empty
PROXY_KEY = os.environ.get("PROXY_KEY", "")  # fetched from signal server if empty

# Optional Telegram alerts (set both to enable)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


class TelegramAlerts:
    """Lightweight Telegram alert sender. Only active if env vars are set."""

    def __init__(self):
        self.enabled = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
        self._url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage" if self.enabled else ""
        if self.enabled:
            log.info("Telegram alerts enabled")

    async def _send(self, text: str):
        if not self.enabled:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(self._url, json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": text,
                    "parse_mode": "HTML",
                })
        except Exception as e:
            log.warning(f"Telegram alert failed: {e}")

    async def position_opened(self, symbol: str, direction: int, price: float, sl_price: float, qty: float, notional: float):
        dir_label = "LONG" if direction == 1 else "SHORT"
        emoji = "\U0001f7e2" if direction == 1 else "\U0001f534"
        await self._send(
            f"{emoji} <b>{symbol} {dir_label}</b>\n"
            f"Entry: ${price:,.4f}\n"
            f"SL: ${sl_price:,.4f}\n"
            f"Size: {qty} (${notional:,.2f})"
        )

    async def position_closed(self, symbol: str, direction: int, entry_price: float, close_price: float, pnl_pct: float, pnl_dollar: float, reason: str):
        dir_label = "LONG" if direction == 1 else "SHORT"
        emoji = "\u2705" if pnl_pct >= 0 else "\u274C"
        reason_label = reason.replace("_", " ").title()
        await self._send(
            f"{emoji} <b>{symbol} {dir_label} Closed</b>\n"
            f"Entry: ${entry_price:,.4f} -> ${close_price:,.4f}\n"
            f"P&L: {pnl_pct:+.2f}% (${pnl_dollar:+,.2f})\n"
            f"Reason: {reason_label}"
        )

    async def bot_started(self, positions: int, balance: float):
        await self._send(
            f"\U0001f680 <b>Vektora Bot Started</b>\n"
            f"Positions: {positions}\n"
            f"Balance: ${balance:,.2f}"
        )

START_TIME = time.time()


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def normalize_symbol(symbol: str) -> str:
    """Strip :USDT suffix. BTC/USDT:USDT -> BTC/USDT"""
    return symbol.split(":")[0]


def binance_symbol(symbol: str) -> str:
    """BTC/USDT -> BTCUSDT (Binance raw format)."""
    return symbol.replace("/", "")


# Binance Futures precision per symbol: (qty_decimals, price_decimals)
_PRECISION = {
    "SUI/USDT": (1, 4), "OP/USDT": (1, 4), "SOL/USDT": (2, 2),
    "ARB/USDT": (1, 4), "APT/USDT": (2, 4), "FET/USDT": (1, 4),
    "FIL/USDT": (1, 4), "STX/USDT": (1, 4), "RUNE/USDT": (1, 4),
    "THETA/USDT": (1, 4), "BNB/USDT": (3, 2), "ETH/USDT": (3, 2),
    "BTC/USDT": (3, 1), "DOT/USDT": (1, 3), "LINK/USDT": (2, 3),
    "SAND/USDT": (0, 5), "XLM/USDT": (0, 5), "LTC/USDT": (3, 2),
}


def _round_qty(symbol: str, qty: float) -> float:
    """Round quantity to Binance's required step size for the symbol."""
    decimals = _PRECISION.get(symbol, (3, 4))[0]
    return round(qty, decimals)


def _round_price(symbol: str, price: float) -> float:
    """Round price to Binance's required tick size for the symbol."""
    decimals = _PRECISION.get(symbol, (3, 4))[1]
    return round(price, decimals)


# ──────────────────────────────────────────────────────────────
# Binance Proxy Client
# ──────────────────────────────────────────────────────────────
class BinanceProxy:
    """Thin HTTP client for the Vektora Binance proxy."""

    def __init__(self, api_key: str, secret: str, testnet: bool = False):
        self.api_key = api_key
        self.secret = secret
        self.testnet = testnet
        self.client = httpx.AsyncClient(timeout=15.0)

    def _headers(self) -> dict:
        h = {
            "X-Proxy-Key": PROXY_KEY,
            "X-Binance-Key": self.api_key,
            "X-Binance-Secret": self.secret,
        }
        if self.testnet:
            h["X-Binance-Testnet"] = "true"
        return h

    async def get_balance(self) -> float:
        """Get USDT futures balance."""
        resp = await self.client.get(
            f"{PROXY_URL}/v1/balance", headers=self._headers()
        )
        if resp.status_code != 200:
            log.error(f"Balance request failed: {resp.status_code} {resp.text[:200]}")
            return 0.0
        data = resp.json()
        for asset in data:
            if asset.get("asset") == "USDT":
                return float(asset.get("balance", 0) or 0)
        return 0.0

    async def get_position(self, symbol: str) -> tuple[float, int]:
        """Get position for symbol. Returns (qty, direction), (0, 0) if flat,
        or (-1, 0) on API errors so callers don't mistake errors for closed positions."""
        bsym = binance_symbol(symbol)
        try:
            resp = await self.client.get(
                f"{PROXY_URL}/v1/position",
                headers=self._headers(),
                params={"symbol": bsym},
            )
        except Exception as e:
            log.error(f"get_position({symbol}) network error: {e}")
            return -1.0, 0
        if resp.status_code != 200:
            log.error(f"get_position({symbol}) HTTP {resp.status_code}")
            return -1.0, 0
        data = resp.json()
        for pos in data:
            if pos.get("symbol") == bsym:
                amt = float(pos.get("positionAmt", 0) or 0)
                if abs(amt) > 0:
                    return abs(amt), 1 if amt > 0 else -1
        return 0.0, 0

    async def set_leverage(self, symbol: str, leverage: int):
        """Set leverage for a symbol."""
        bsym = binance_symbol(symbol)
        resp = await self.client.post(
            f"{PROXY_URL}/v1/leverage",
            headers=self._headers(),
            json={"symbol": bsym, "leverage": leverage},
        )
        if resp.status_code != 200:
            log.warning(f"set_leverage {symbol}: {resp.text[:100]}")

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED"):
        """Set margin type for a symbol."""
        bsym = binance_symbol(symbol)
        resp = await self.client.post(
            f"{PROXY_URL}/v1/marginType",
            headers=self._headers(),
            json={"symbol": bsym, "marginType": margin_type},
        )
        if resp.status_code != 200:
            text = resp.text
            if "No need to change margin type" not in text:
                log.warning(f"set_margin_type {symbol}: {text[:100]}")

    async def place_market_order(self, symbol: str, side: str, qty: float) -> dict:
        """Place a market order. Returns order response."""
        bsym = binance_symbol(symbol)
        resp = await self.client.post(
            f"{PROXY_URL}/v1/order",
            headers=self._headers(),
            json={
                "symbol": bsym,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": str(qty),
            },
        )
        if resp.status_code != 200:
            raise Exception(f"Market order failed: {resp.status_code} {resp.text[:200]}")
        return resp.json()

    async def place_stop_market(
        self, symbol: str, side: str, qty: float, stop_price: float
    ) -> dict:
        """Place a stop-market (SL) order via the Algo Order endpoint."""
        bsym = binance_symbol(symbol)
        resp = await self.client.post(
            f"{PROXY_URL}/v1/algoOrder",
            headers=self._headers(),
            json={
                "symbol": bsym,
                "side": side.upper(),
                "type": "STOP_MARKET",
                "quantity": str(qty),
                "triggerPrice": str(stop_price),
            },
        )
        if resp.status_code != 200:
            raise Exception(f"Stop order failed: {resp.status_code} {resp.text[:200]}")
        return resp.json()

    async def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol."""
        bsym = binance_symbol(symbol)
        # Get all open orders and cancel individually
        resp = await self.client.get(
            f"{PROXY_URL}/v1/openOrders",
            headers=self._headers(),
            params={"symbol": bsym},
        )
        if resp.status_code == 200:
            orders = resp.json()
            for order in orders:
                order_id = order.get("orderId", "")
                if order_id:
                    await self.client.delete(
                        f"{PROXY_URL}/v1/order",
                        headers=self._headers(),
                        params={"symbol": bsym, "orderId": str(order_id)},
                    )

    async def close(self):
        await self.client.aclose()


# ──────────────────────────────────────────────────────────────
# Client Bot
# ──────────────────────────────────────────────────────────────
class ClientBot:
    def __init__(self):
        self.proxy: BinanceProxy | None = None
        self.signal_api_key: str = ""
        self.running = False
        self.status = "waiting_for_setup"
        self.positions: dict = {}
        self.protective_orders: dict = {}
        self.session_pnl = 0.0
        self.consecutive_losses: dict = {}
        self._ws_task: asyncio.Task | None = None
        self.alerts = TelegramAlerts()

    # ── Setup ─────────────────────────────────────────────────

    async def _fetch_proxy_config(self, signal_api_key: str):
        """Fetch proxy URL/key from signal server using the signal API key."""
        global PROXY_URL, PROXY_KEY
        if PROXY_URL and PROXY_KEY:
            return  # already set via env vars
        http_url = SIGNAL_SERVER_URL.replace("wss://", "https://").replace("ws://", "http://")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{http_url}/api/proxy-config",
                    params={"key": signal_api_key},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    PROXY_URL = data["proxy_url"]
                    PROXY_KEY = data["proxy_key"]
                    whitelist_ip = data.get("whitelist_ip", "")
                    log.info(f"Proxy config fetched from signal server")
                    if whitelist_ip:
                        log.info(f"*** IMPORTANT: Whitelist this IP on your Binance API key: {whitelist_ip} ***")
                else:
                    log.error(f"Failed to fetch proxy config: {resp.status_code}")
        except Exception as e:
            log.error(f"Failed to fetch proxy config: {e}")

    async def configure(
        self, binance_api_key: str, binance_secret: str,
        signal_api_key: str, testnet: bool = False,
    ):
        """Configure and start the bot."""
        self.signal_api_key = signal_api_key
        await self._fetch_proxy_config(signal_api_key)
        if not PROXY_URL or not PROXY_KEY:
            raise ValueError("Proxy config unavailable — check signal API key")
        self.proxy = BinanceProxy(binance_api_key, binance_secret, testnet)

        # Save credentials to persistent volume
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CREDS_FILE, "w") as f:
            json.dump({
                "binance_api_key": binance_api_key,
                "binance_secret": binance_secret,
                "signal_api_key": signal_api_key,
                "testnet": testnet,
            }, f)
        os.chmod(CREDS_FILE, stat.S_IRUSR | stat.S_IWUSR)

        # Setup leverage + margin mode
        for symbol in SYMBOLS:
            await self.proxy.set_leverage(symbol, LEVERAGE)
            await self.proxy.set_margin_type(symbol)

        self._load_state()
        await self._start()

    async def _start(self):
        """Start the WS connection loop."""
        self.running = True
        self.status = "running"
        self._ws_task = asyncio.create_task(self._connect_signal_server())
        log.info("Bot started")

        # Startup Telegram alert
        try:
            balance = await self.proxy.get_balance() if self.proxy else 0
        except Exception:
            balance = 0
        await self.alerts.bot_started(len(self.positions), balance)

    async def _try_load_credentials(self) -> bool:
        """Load saved credentials on startup."""
        try:
            if os.path.exists(CREDS_FILE):
                with open(CREDS_FILE, "r") as f:
                    creds = json.load(f)
                self.signal_api_key = creds["signal_api_key"]
                await self._fetch_proxy_config(self.signal_api_key)
                if not PROXY_URL or not PROXY_KEY:
                    log.error("Proxy config unavailable on restart")
                    return False
                self.proxy = BinanceProxy(
                    creds["binance_api_key"],
                    creds["binance_secret"],
                    creds.get("testnet", False),
                )
                self._load_state()
                return True
        except Exception as e:
            log.error(f"Failed to load credentials: {e}")
        return False

    # ── Signal handling ───────────────────────────────────────

    async def handle_signal(self, event: dict):
        """Handle a direction flip signal."""
        raw_symbol = event["symbol"]
        symbol = normalize_symbol(raw_symbol)

        if symbol not in SYMBOLS:
            return

        direction = event["direction"]
        price = event["price"]
        sl_price = event["sl_price"]
        dir_label = "LONG" if direction == 1 else "SHORT"

        log.info(f"SIGNAL: {symbol} -> {dir_label} @ ${price:.4f} (SL ${sl_price:.4f})")

        existing = self.positions.get(symbol)

        if existing and existing["direction"] == direction:
            log.info(f"  Already {dir_label} on {symbol}, skipping")
            return

        # Anti-whipsaw: enforce minimum hold time
        if existing:
            try:
                entry_time = datetime.fromisoformat(existing["entry_time"])
                held_minutes = (datetime.now() - entry_time).total_seconds() / 60
                if held_minutes < MIN_HOLD_MINUTES:
                    log.warning(
                        f"  {symbol}: WHIPSAW BLOCKED — held only {held_minutes:.0f}m "
                        f"(min {MIN_HOLD_MINUTES}m)"
                    )
                    return
            except (ValueError, KeyError):
                pass

            await self._close_position(symbol, price, "signal_flip")

        await self._open_position(symbol, direction, price, sl_price)

    async def handle_snapshot(self, snapshot: dict):
        """Handle a 60s snapshot — check SL hits and ratchet profit floors."""
        symbols_data = snapshot.get("symbols", {})

        for symbol, pos in list(self.positions.items()):
            data = symbols_data.get(symbol) or symbols_data.get(f"{symbol}:USDT")
            if not data:
                continue

            current_price = data["price"]

            # Check if position was closed on exchange (SL hit)
            if self.proxy:
                ex_qty, _ = await self.proxy.get_position(symbol)
                if ex_qty < 0:
                    log.warning(f"  {symbol}: exchange API error, skipping position check")
                    continue
                if ex_qty == 0:
                    reason = "floor_stop" if pos.get("last_floor", -1) >= 0 else "stop_loss"
                    sl_price = self.protective_orders.get(symbol, {}).get("sl_price", 0)
                    close_price = sl_price if sl_price > 0 else current_price
                    log.info(f"  {symbol}: position closed on exchange ({reason}) @ ${close_price:.4f}")
                    self._record_close(symbol, close_price, reason)
                    continue

            # Check direction mismatch (missed flip during disconnect)
            signal_dir = data.get("direction")
            if signal_dir is not None and signal_dir != pos["direction"]:
                try:
                    entry_time = datetime.fromisoformat(pos["entry_time"])
                    held_minutes = (datetime.now() - entry_time).total_seconds() / 60
                    if held_minutes < MIN_HOLD_MINUTES:
                        continue
                except (ValueError, KeyError):
                    pass
                log.warning(f"  {symbol}: direction mismatch — closing and reopening")
                await self._close_position(symbol, current_price, "missed_flip")
                sl_pct = SL_PCT / 100
                new_sl = current_price * (1 - sl_pct) if signal_dir == 1 else current_price * (1 + sl_pct)
                new_sl = _round_price(symbol, new_sl)
                await self._open_position(symbol, signal_dir, current_price, new_sl)
                continue

            # Profit floor ratchet
            await self._check_profit_floors(symbol, current_price)

    # ── Trade execution ───────────────────────────────────────

    async def _open_position(self, symbol: str, direction: int, price: float, sl_price: float):
        """Open a new position via the proxy."""
        if len(self.positions) >= MAX_POSITIONS:
            log.warning(f"  Max positions ({MAX_POSITIONS}) reached, skipping {symbol}")
            return

        if not self.proxy:
            return

        # Check for existing exchange position
        ex_qty, _ = await self.proxy.get_position(symbol)
        if ex_qty > 0:
            log.warning(f"  {symbol}: existing position on exchange — skipping")
            return

        try:
            balance = await self.proxy.get_balance()
            if balance <= 0:
                log.error(f"  No USDT balance to open {symbol}")
                return

            allocation = balance * (RISK_PER_TRADE_PCT / 100)
            notional = allocation * LEVERAGE
            raw_qty = notional / price
            # Round quantity to Binance's required precision per symbol
            qty = _round_qty(symbol, raw_qty)

            if qty <= 0:
                log.error(f"  Calculated qty is 0 for {symbol}")
                return

            log.info(f"  Balance: ${balance:.2f} | Alloc: ${allocation:.2f} | Qty: {qty}")

            # Market entry
            side = "BUY" if direction == 1 else "SELL"
            order = await self.proxy.place_market_order(symbol, side, qty)
            fill_price = float(order.get("avgPrice", price) or price)
            price = fill_price
            log.info(f"  OPENED {symbol}: {side} {qty} @ ${fill_price:.4f}")

            # Place SL (round to tick size)
            close_side = "SELL" if direction == 1 else "BUY"
            sl_price = _round_price(symbol, sl_price)
            try:
                await self.proxy.place_stop_market(symbol, close_side, qty, sl_price)
                self.protective_orders[symbol] = {"sl_price": sl_price}
                log.info(f"  SL placed @ ${sl_price:.4f}")
            except Exception as e:
                log.error(f"  SL placement failed for {symbol}: {e}")
                log.error(f"  EMERGENCY CLOSING {symbol}")
                try:
                    await self.proxy.place_market_order(symbol, close_side, qty)
                except Exception as e2:
                    log.error(f"  EMERGENCY CLOSE FAILED: {e2}")
                return

        except Exception as e:
            log.error(f"  Failed to open {symbol}: {e}")
            return

        self.positions[symbol] = {
            "direction": direction,
            "qty": qty,
            "entry_price": price,
            "sl_price": sl_price,
            "entry_time": datetime.now().isoformat(),
            "max_pnl_pct": 0.0,
            "last_floor": -1.0,
        }
        self._save_state()
        await self.alerts.position_opened(symbol, direction, price, sl_price, qty, qty * price)

    async def _close_position(self, symbol: str, close_price: float, reason: str):
        """Close an existing position via the proxy."""
        pos = self.positions.get(symbol)
        if not pos or not self.proxy:
            return

        direction = pos["direction"]

        # Cancel all orders
        await self.proxy.cancel_all_orders(symbol)

        # Check actual exchange qty
        ex_qty, _ = await self.proxy.get_position(symbol)
        if ex_qty < 0:
            log.warning(f"  {symbol}: exchange API error during close, using local qty")
            ex_qty = pos["qty"]
        elif ex_qty == 0:
            log.info(f"  {symbol}: already closed on exchange")
            self._record_close(symbol, close_price, reason)
            return

        close_side = "SELL" if direction == 1 else "BUY"
        try:
            order = await self.proxy.place_market_order(symbol, close_side, ex_qty)
            fill_price = float(order.get("avgPrice", close_price) or close_price)
            close_price = fill_price
            log.info(f"  CLOSED {symbol}: {close_side} {ex_qty} @ ${fill_price:.4f}")
        except Exception as e:
            log.error(f"  Failed to close {symbol}: {e}")
            return

        self._record_close(symbol, close_price, reason)

    def _record_close(self, symbol: str, close_price: float, reason: str):
        """Update local state for a closed position."""
        pos = self.positions.pop(symbol, None)
        self.protective_orders.pop(symbol, None)
        if not pos:
            return

        entry_price = pos["entry_price"]
        direction = pos["direction"]

        if direction == 1:
            pnl_pct = (close_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - close_price) / entry_price * 100

        roi_pct = pnl_pct * LEVERAGE
        notional = pos["qty"] * entry_price
        pnl_dollar = (pnl_pct / 100) * notional

        self.session_pnl += pnl_dollar

        if pnl_pct < 0:
            self.consecutive_losses[symbol] = self.consecutive_losses.get(symbol, 0) + 1
        else:
            self.consecutive_losses[symbol] = 0

        dir_label = "LONG" if direction == 1 else "SHORT"
        log.info(
            f"  {symbol} {dir_label}: {pnl_pct:+.2f}% (ROI {roi_pct:+.1f}%) "
            f"${pnl_dollar:+,.2f} [{reason}]"
        )

        self._save_state()

        # Fire-and-forget Telegram alert
        asyncio.create_task(
            self.alerts.position_closed(symbol, direction, entry_price, close_price, pnl_pct, pnl_dollar, reason)
        )

    # ── Profit floor ratchet ──────────────────────────────────

    async def _check_profit_floors(self, symbol: str, current_price: float):
        """Check and ratchet profit floor for a position."""
        pos = self.positions.get(symbol)
        if not pos or not self.proxy:
            return

        entry_price = pos["entry_price"]
        direction = pos["direction"]
        if entry_price <= 0:
            return

        if direction == 1:
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100

        max_pnl = max(pos.get("max_pnl_pct", 0.0), pnl_pct)
        pos["max_pnl_pct"] = max_pnl

        last_floor = pos.get("last_floor", -1.0)
        best_lock = -1.0
        for threshold in sorted(FLOOR_LEVELS.keys()):
            if max_pnl >= threshold:
                best_lock = FLOOR_LEVELS[threshold]

        if best_lock <= last_floor:
            return

        if direction == 1:
            new_sl = entry_price * (1 + best_lock / 100)
        else:
            new_sl = entry_price * (1 - best_lock / 100)

        old_sl = self.protective_orders.get(symbol, {}).get("sl_price", 0)

        should_move = False
        if direction == 1 and new_sl > old_sl:
            should_move = True
        elif direction == -1 and (old_sl == 0 or new_sl < old_sl):
            should_move = True

        if not should_move:
            return

        lock_label = f"+{best_lock:.1f}%" if best_lock > 0 else "breakeven"
        log.info(
            f"  FLOOR RATCHET {symbol}: peak +{max_pnl:.1f}% -> lock {lock_label} "
            f"(SL ${old_sl:.4f} -> ${new_sl:.4f})"
        )

        # Place new SL before cancelling old
        close_side = "SELL" if direction == 1 else "BUY"
        qty = pos["qty"]
        try:
            await self.proxy.place_stop_market(symbol, close_side, qty, new_sl)
        except Exception as e:
            log.error(f"  Failed to place new SL for {symbol}: {e}")
            return

        # Cancel old orders
        await self.proxy.cancel_all_orders(symbol)

        # Re-place the new SL (cancel_all removed it)
        try:
            await self.proxy.place_stop_market(symbol, close_side, qty, new_sl)
            self.protective_orders[symbol] = {"sl_price": new_sl}
        except Exception as e:
            log.error(f"  Failed to re-place SL for {symbol} after cleanup: {e}")
            return

        pos["last_floor"] = best_lock
        self._save_state()

    # ── WebSocket client ──────────────────────────────────────

    async def _connect_signal_server(self):
        """Connect to signal server with auto-reconnect."""
        ws_base = SIGNAL_SERVER_URL.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_base}/ws?key={self.signal_api_key}"

        reconnect_delay = 5
        max_delay = 60

        while self.running:
            try:
                log.info("Connecting to signal server...")
                async with websockets.connect(ws_url, ping_interval=30, ping_timeout=30) as ws:
                    log.info("Connected to signal server")

                    async for raw in ws:
                        reconnect_delay = 5  # reset only after receiving data
                        if not self.running:
                            break
                        try:
                            msg = json.loads(raw)
                            msg_type = msg.get("type")
                            if msg_type == "signal":
                                await self.handle_signal(msg)
                            elif msg_type == "snapshot":
                                await self.handle_snapshot(msg)
                        except json.JSONDecodeError:
                            log.warning(f"Invalid JSON: {raw[:100]}")
                        except Exception as e:
                            log.error(f"Error handling message: {e}")

            except websockets.exceptions.ConnectionClosed as e:
                log.warning(f"WebSocket closed: code={e.code} reason={e.reason}")
                if e.code == 4029:
                    reconnect_delay = 65  # server says wait 60s, add buffer
            except (ConnectionRefusedError, OSError) as e:
                log.warning(f"Connection failed: {type(e).__name__}")
            except Exception as e:
                log.error(f"WebSocket error: {type(e).__name__}")

            if not self.running:
                break

            log.info(f"Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_delay)

    # ── State persistence ─────────────────────────────────────

    def _save_state(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({
                    "positions": self.positions,
                    "protective_orders": self.protective_orders,
                    "session_pnl": self.session_pnl,
                    "consecutive_losses": self.consecutive_losses,
                    "updated": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save state: {e}")

    def _load_state(self):
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
                self.positions = state.get("positions", {})
                self.protective_orders = state.get("protective_orders", {})
                self.consecutive_losses = state.get("consecutive_losses", {})
                for pos in self.positions.values():
                    pos.setdefault("max_pnl_pct", 0.0)
                    pos.setdefault("last_floor", -1.0)
                if self.positions:
                    log.info(f"Loaded state: {len(self.positions)} tracked positions")
        except Exception as e:
            log.error(f"Failed to load state: {e}")
            self.positions = {}

    def get_status(self) -> dict:
        """Return current bot status for API."""
        return {
            "status": self.status,
            "uptime_seconds": round(time.time() - START_TIME),
            "positions": len(self.positions),
            "session_pnl": round(self.session_pnl, 2),
            "open_positions": {
                sym: {
                    "direction": "LONG" if p["direction"] == 1 else "SHORT",
                    "entry_price": p["entry_price"],
                    "qty": p["qty"],
                    "entry_time": p["entry_time"],
                    "sl_price": self.protective_orders.get(sym, {}).get("sl_price", 0),
                    "max_pnl_pct": round(p.get("max_pnl_pct", 0), 2),
                    "floor_lock": p.get("last_floor", -1),
                }
                for sym, p in self.positions.items()
            },
        }


# ──────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────
bot = ClientBot()
app = FastAPI(title="Vektora Trading Bot", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vektora.trade"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Setup token — loaded from SETUP_TOKEN env var or persisted credentials
_setup_token: str | None = os.environ.get("SETUP_TOKEN")


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Auto-start from env vars or saved credentials."""
    global _setup_token

    # Priority 1: env vars (Railway template sets these)
    env_binance_key = os.environ.get("BINANCE_KEY", "")
    env_binance_secret = os.environ.get("BINANCE_SECRET", "")
    env_signal_key = os.environ.get("SIGNAL_API_KEY", "")

    if env_binance_key and env_binance_secret and env_signal_key:
        log.info("Auto-configuring from environment variables")
        try:
            await bot.configure(env_binance_key, env_binance_secret, env_signal_key)
        except Exception as e:
            log.error(f"Auto-configure from env vars failed: {e}")

    # Priority 2: saved credentials from previous deploy
    elif await bot._try_load_credentials():
        try:
            if os.path.exists(CREDS_FILE):
                with open(CREDS_FILE, "r") as f:
                    creds = json.load(f)
                _setup_token = creds.get("setup_token")
        except Exception:
            pass
        await bot._start()

    yield


app.router.lifespan_context = lifespan


@app.get("/health")
async def health():
    return {
        "status": bot.status,
        "uptime_seconds": round(time.time() - START_TIME),
        "positions": len(bot.positions),
        "service": "vektora-bot",
    }


@app.post("/api/configure")
async def configure(request: Request):
    """Configure the bot with Binance + signal API credentials."""
    global _setup_token

    body = await request.json()
    token = body.get("setup_token", "")
    binance_api_key = body.get("binance_api_key", "")
    binance_secret = body.get("binance_secret", "")
    signal_api_key = body.get("signal_api_key", "")
    testnet = body.get("testnet", False)

    if not token:
        raise HTTPException(status_code=400, detail="setup_token is required")
    if not binance_api_key or not binance_secret:
        raise HTTPException(status_code=400, detail="Binance credentials required")
    if not signal_api_key:
        raise HTTPException(status_code=400, detail="signal_api_key required")

    if _setup_token is None:
        raise HTTPException(status_code=503, detail="SETUP_TOKEN env var not set")
    if not hmac.compare_digest(token, _setup_token):
        raise HTTPException(status_code=401, detail="Invalid setup token")

    # Save token alongside credentials
    if bot.running:
        bot.running = False
        if bot._ws_task:
            bot._ws_task.cancel()
            try:
                await bot._ws_task
            except asyncio.CancelledError:
                pass

    await bot.configure(binance_api_key, binance_secret, signal_api_key, testnet)

    # Persist the setup token
    try:
        with open(CREDS_FILE, "r") as f:
            creds = json.load(f)
        creds["setup_token"] = token
        with open(CREDS_FILE, "w") as f:
            json.dump(creds, f)
    except Exception:
        pass

    return {"status": "configured", "message": "Bot is now running"}


@app.get("/api/status")
async def status(request: Request):
    """Get bot status — requires setup token."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        token = request.query_params.get("token", "")

    if not _setup_token or not hmac.compare_digest(token, _setup_token):
        raise HTTPException(status_code=401, detail="Invalid token")

    return bot.get_status()
