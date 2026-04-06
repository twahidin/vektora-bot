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
MIN_HOLD_MINUTES = 0
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
    "ARB/USDT": (1, 4), "APT/USDT": (1, 4), "FET/USDT": (1, 4),
    "FIL/USDT": (1, 4), "STX/USDT": (1, 4), "RUNE/USDT": (1, 4),
    "THETA/USDT": (1, 4), "BNB/USDT": (2, 2), "ETH/USDT": (3, 2),
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
                "algoType": "CONDITIONAL",
                "symbol": bsym,
                "side": side.upper(),
                "type": "STOP_MARKET",
                "quantity": str(qty),
                "triggerPrice": str(stop_price),
                "reduceOnly": "true",
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
        self.active_symbols: list[str] = list(SYMBOLS)
        self._syncing = False
        self.bot_state = "running"  # running, paused_holding, paused_closed
        self.last_prices: dict[str, float] = {}
        self.session_pnl = 0.0
        self.consecutive_losses: dict = {}
        self._ws_task: asyncio.Task | None = None
        self._agent_task: asyncio.Task | None = None
        self.alerts = TelegramAlerts()
        self._events: list[dict] = []
        self._last_snapshot_symbols: dict = {}  # for agent: indicator state per symbol
        self._signal_history: list[dict] = []   # for agent: recent signal flips with timestamps
        self._last_balance: float = 0.0         # for agent: last known wallet balance

    def _log_event(self, event_type: str, symbol: str, message: str):
        """Log a bot activity event for the dashboard."""
        self._events.append({
            "type": event_type,
            "symbol": symbol,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self._events) > 50:
            self._events = self._events[-50:]

    def _add_pending_flip(self, symbol: str, direction: int, price: float,
                          sl_price: float, adx: float, bb_w: float, zone: str):
        """Queue a blocked flip for AI agent evaluation."""
        if not hasattr(self, '_pending_flips'):
            self._pending_flips = {}
        self._pending_flips[symbol] = {
            "signal_direction": direction,
            "signal_price": price,
            "sl_price": sl_price,
            "blocked_at": time.time(),
            "adx": adx,
            "bb_width": bb_w,
            "zone": zone,
            "consecutive_losses": self.consecutive_losses.get(symbol, 0),
        }
        log.info(f"  {symbol}: added to pending_flips (zone={zone}, ADX={adx:.0f}, BBW={bb_w:.1f}%)")

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

    async def _fetch_symbols(self):
        """Fetch selected symbols from signal server."""
        if not self.signal_api_key:
            return
        http_url = SIGNAL_SERVER_URL.replace("wss://", "https://").replace("ws://", "http://")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{http_url}/api/symbols",
                    params={"key": self.signal_api_key},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self.active_symbols = [normalize_symbol(s) for s in data["symbols"]]
                    log.info(f"Symbol preferences loaded: {len(self.active_symbols)} of {len(data['available'])} active")
                else:
                    log.warning(f"Failed to fetch symbols: {resp.status_code}")
        except Exception as e:
            log.warning(f"Failed to fetch symbols: {e}")

    async def _symbol_refresh_loop(self):
        """Re-fetch symbol preferences every 5 minutes."""
        while self.running:
            await asyncio.sleep(300)
            await self._fetch_symbols()

    async def _cleanup_orphaned_orders(self):
        """Cancel ALL conditional orders across all symbols on startup.
        Prevents orphaned SL orders from accumulating across restarts."""
        if not self.proxy:
            return
        log.info("Cleanup: cancelling all conditional orders across all symbols...")
        cancelled = 0
        for symbol in SYMBOLS:
            symbol = normalize_symbol(symbol)
            try:
                await self.proxy.cancel_all_orders(symbol)
                cancelled += 1
            except Exception as e:
                log.warning(f"Cleanup: failed to cancel orders for {symbol}: {e}")
        log.info(f"Cleanup: cancelled orders for {cancelled} symbols")

    async def _sync_positions_with_signals(self):
        """Sync Binance positions with signal server directions on startup.

        Checks every active symbol against the signal server snapshot:
        0. Cancel ALL orphaned conditional orders first
        1. Wrong direction on Binance → close and reopen correctly
        2. Missing position (flat but signal is active) → open position
        3. Stale local state (tracked but not on Binance) → clean up

        Only touches the user's Binance account via proxy. Does not affect
        the signal server's paper trader (separate system).

        Sets self._syncing = True to prevent race conditions with WS handlers.
        """
        if not self.signal_api_key or not self.proxy:
            return

        self._syncing = True

        # If bot is paused, don't sync — it would reopen positions
        if self.bot_state != "running":
            log.info(f"Sync: skipping — bot is {self.bot_state}")
            self._syncing = False
            return

        # Step 0: Cancel all orphaned orders before sync
        await self._cleanup_orphaned_orders()
        http_url = SIGNAL_SERVER_URL.replace("wss://", "https://").replace("ws://", "http://")

        # Fetch current signal directions from server snapshot
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{http_url}/api/snapshot",
                    params={"key": self.signal_api_key},
                )
                if resp.status_code != 200:
                    log.warning(f"Sync: failed to fetch snapshot ({resp.status_code})")
                    self._syncing = False
                    return
                snapshot = resp.json()
        except Exception as e:
            log.warning(f"Sync: failed to fetch snapshot: {e}")
            self._syncing = False
            return

        symbols_data = snapshot.get("symbols", {})
        if not symbols_data:
            log.info("Sync: no snapshot data available yet")
            self._syncing = False
            return

        log.info(f"Sync: checking {len(self.active_symbols)} symbols against signal server...")
        fixed = 0

        for symbol in self.active_symbols:
            # Normalize symbol to short format (SUI/USDT, not SUI/USDT:USDT)
            symbol = normalize_symbol(symbol)

            # Get signal direction (try ccxt futures format first)
            sig_data = symbols_data.get(f"{symbol}:USDT") or symbols_data.get(symbol)
            if not sig_data:
                continue
            signal_dir = sig_data.get("direction")
            if signal_dir is None:
                continue
            signal_price = sig_data.get("price", 0)

            # Validate price before any trading actions
            if not signal_price or signal_price <= 0:
                log.warning(f"  Sync: {symbol} has invalid price ({signal_price}), skipping")
                continue

            # Get Binance position
            ex_qty, ex_dir = await self.proxy.get_position(symbol)
            if ex_qty < 0:
                continue  # API error, skip

            tracked = self.positions.get(symbol)

            if ex_qty > 0 and ex_dir == signal_dir:
                # Correct direction — ensure it's tracked locally
                if not tracked:
                    log.info(f"  Sync: {symbol} on Binance matches signal, adding to local state")
                    dir_label = "LONG" if signal_dir == 1 else "SHORT"
                    self.positions[symbol] = {
                        "direction": signal_dir,
                        "qty": ex_qty,
                        "entry_price": signal_price,
                        "sl_price": 0,
                        "entry_time": datetime.now().isoformat(),
                        "max_pnl_pct": 0.0,
                        "last_floor": -1.0,
                    }

            elif ex_qty > 0 and ex_dir != signal_dir:
                # WRONG DIRECTION — close and reopen
                dir_label = "LONG" if ex_dir == 1 else "SHORT"
                new_label = "LONG" if signal_dir == 1 else "SHORT"
                log.warning(f"  Sync: {symbol} is {dir_label} on Binance but signal says {new_label} — fixing")

                # Cancel existing orders
                await self.proxy.cancel_all_orders(symbol)

                # Close wrong-direction position
                close_side = "SELL" if ex_dir == 1 else "BUY"
                try:
                    order = await self.proxy.place_market_order(symbol, close_side, ex_qty)
                    fill_price = float(order.get("avgPrice", signal_price) or signal_price)
                    log.info(f"  Sync: closed {symbol} {dir_label} @ ${fill_price:.4f}")
                except Exception as e:
                    log.error(f"  Sync: failed to close {symbol}: {e}")
                    continue

                # Remove from internal state
                self.positions.pop(symbol, None)
                self.protective_orders.pop(symbol, None)
                self._log_event("close", symbol, f"Sync: closed {dir_label} (wrong direction)")

                # Open in correct direction
                sl_pct = SL_PCT / 100
                new_sl = signal_price * (1 - sl_pct) if signal_dir == 1 else signal_price * (1 + sl_pct)
                new_sl = _round_price(symbol, new_sl)
                await self._open_position(symbol, signal_dir, signal_price, new_sl)
                fixed += 1

            elif ex_qty == 0 and tracked:
                # Bot thinks it has a position but Binance doesn't — clean up
                log.warning(f"  Sync: {symbol} tracked locally but not on Binance — cleaning up")
                self.positions.pop(symbol, None)
                self.protective_orders.pop(symbol, None)

            elif ex_qty == 0 and not tracked:
                # NO POSITION — open in signal direction (if under max positions)
                if len(self.positions) >= MAX_POSITIONS:
                    continue
                dir_label = "LONG" if signal_dir == 1 else "SHORT"
                log.info(f"  Sync: {symbol} has no position, signal says {dir_label} — opening")
                sl_pct = SL_PCT / 100
                new_sl = signal_price * (1 - sl_pct) if signal_dir == 1 else signal_price * (1 + sl_pct)
                new_sl = _round_price(symbol, new_sl)
                await self._open_position(symbol, signal_dir, signal_price, new_sl)
                fixed += 1

        self._save_state()
        self._syncing = False
        if fixed > 0:
            log.info(f"Sync: fixed {fixed} position(s)")
        else:
            log.info("Sync: all positions aligned")

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
        await self._fetch_symbols()

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
        await self._sync_positions_with_signals()
        await self._start()

    async def _start(self):
        """Start the WS connection loop, status reporting, and AI agent."""
        self.running = True
        self.status = "running"
        self._ws_task = asyncio.create_task(self._connect_signal_server())
        self._status_task = asyncio.create_task(self._status_report_loop())
        self._symbol_refresh_task = asyncio.create_task(self._symbol_refresh_loop())
        self._periodic_sync_task = asyncio.create_task(self._periodic_sync_loop())
        self._command_poll_task = asyncio.create_task(self._poll_commands())

        # Start AI peak agent (calls signal server for evaluation — no API key needed)
        from bot_agent import BotPeakAgent
        self._bot_agent = BotPeakAgent(self)
        self._agent_task = asyncio.create_task(self._bot_agent.run())
        log.info("AI peak agent started (server-side evaluation)")

        log.info("Bot started")

    async def _periodic_sync_loop(self):
        """Re-sync positions with signal server every 15 minutes as a safety net."""
        while self.running:
            await asyncio.sleep(900)  # 15 minutes
            try:
                log.info("Periodic sync: verifying positions match signal server...")
                await self._sync_positions_with_signals()
            except Exception as e:
                log.warning(f"Periodic sync failed: {e}")

    async def _poll_commands(self):
        """Poll for pending commands every 30 seconds."""
        http_url = SIGNAL_SERVER_URL.replace("wss://", "https://").replace("ws://", "http://")
        while self.running:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{http_url}/api/commands",
                        params={"key": self.signal_api_key},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("id") and data.get("command"):
                            await self._execute_command(data["id"], data["command"])
            except Exception as e:
                log.error(f"Command poll failed: {e}")
            await asyncio.sleep(30)

    async def _execute_command(self, cmd_id: int, command: str):
        """Execute a bot command."""
        log.info(f"Executing command: {command} (id={cmd_id})")
        try:
            if command == "pause":
                self.bot_state = "paused_holding"
                log.info("Bot paused (holding positions)")
            elif command == "pause_close_all":
                self.bot_state = "paused_closed"
                await self._close_all_positions()
                log.info("Bot paused (all positions closed)")
            elif command == "resume":
                self.bot_state = "running"
                log.info("Bot resumed")
            elif command.startswith("close_symbol|"):
                # Peak agent: close a specific symbol
                symbol = normalize_symbol(command.split("|", 1)[1])
                await self._agent_close_symbol(symbol)
            elif command.startswith("skip_symbol|"):
                # Peak agent: close and skip a symbol for N minutes
                parts = command.split("|")
                symbol = normalize_symbol(parts[1]) if len(parts) > 1 else ""
                duration = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 30
                await self._agent_skip_symbol(symbol, duration)

            # Acknowledge command
            http_url = SIGNAL_SERVER_URL.replace("wss://", "https://").replace("ws://", "http://")
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{http_url}/api/commands/{cmd_id}/ack",
                    params={"key": self.signal_api_key},
                )
        except Exception as e:
            log.error(f"Command execution failed: {e}")

    async def _close_all_positions(self):
        """Close all open Binance positions."""
        try:
            for symbol in list(self.positions.keys()):
                pos = self.positions.get(symbol)
                if not pos:
                    continue
                direction = pos["direction"]
                dir_label = "LONG" if direction == 1 else "SHORT"
                log.info(f"Closing {symbol} ({dir_label})")
                # Use current entry price as close price estimate;
                # _close_position will get the actual fill price from exchange
                await self._close_position(symbol, pos["entry_price"], "command_close_all")
        except Exception as e:
            log.error(f"Close all positions failed: {e}")

    async def _agent_close_symbol(self, symbol: str):
        """Close a position on behalf of the peak agent (profit-taking).
        Blocks re-entry until next signal flip by adding to _agent_paused."""
        pos = self.positions.get(symbol)
        if not pos:
            log.info(f"Peak agent close_symbol: {symbol} — no position held, ignoring")
            return
        dir_label = "LONG" if pos["direction"] == 1 else "SHORT"
        log.info(f"Peak agent: closing {symbol} ({dir_label}) — profit-taking at peak")
        await self._close_position(symbol, pos["entry_price"], "peak_agent")
        # Block re-entry until next signal flip (mirrors engine.paused_symbols)
        if not hasattr(self, "_agent_paused"):
            self._agent_paused = set()
        self._agent_paused.add(symbol)
        log.info(f"Peak agent: {symbol} paused until next signal flip")
        await self.alerts.position_closed(
            symbol, pos["direction"], pos["entry_price"], pos["entry_price"],
            0.0, 0.0, "peak_agent"
        )

    async def _agent_skip_symbol(self, symbol: str, duration_minutes: int):
        """Close a position and block re-entry for N minutes (whipsaw protection)."""
        pos = self.positions.get(symbol)
        if pos:
            dir_label = "LONG" if pos["direction"] == 1 else "SHORT"
            log.info(f"Peak agent: skipping {symbol} ({dir_label}) for {duration_minutes}m — whipsaw protection")
            await self._close_position(symbol, pos["entry_price"], "peak_agent_skip")
        else:
            log.info(f"Peak agent skip_symbol: {symbol} — no position, blocking re-entry for {duration_minutes}m")

        # Block re-entry using a time-based skip
        skip_until = time.time() + duration_minutes * 60
        if not hasattr(self, "_skip_until"):
            self._skip_until = {}
        self._skip_until[symbol] = skip_until
        log.info(f"Peak agent: {symbol} blocked for {duration_minutes}m")

    async def _status_report_loop(self):
        """Report status to signal server every 60 seconds."""
        # Startup Telegram alert
        try:
            balance = await self.proxy.get_balance() if self.proxy else 0
        except Exception:
            balance = 0
        await self.alerts.bot_started(len(self.positions), balance)

        await asyncio.sleep(10)  # first report after 10s
        while self.running:
            try:
                await self._report_status()
                log.info("Status report sent to dashboard")
            except Exception as e:
                log.warning(f"Status report failed: {e}")
            await asyncio.sleep(60)

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
                await self._fetch_symbols()
                self._load_state()
                await self._sync_positions_with_signals()
                return True
        except Exception as e:
            log.error(f"Failed to load credentials: {e}")
        return False

    # ── Signal handling ───────────────────────────────────────

    async def handle_signal(self, event: dict):
        """Handle a direction flip signal."""
        if self.bot_state != "running":
            log.info(f"Ignoring signal (bot state: {self.bot_state})")
            return

        if self._syncing:
            return  # sync in progress, ignore signals until done

        raw_symbol = event["symbol"]
        symbol = normalize_symbol(raw_symbol)

        if symbol not in self.active_symbols:
            return

        direction = event["direction"]
        price = event["price"]
        sl_price = event["sl_price"]
        dir_label = "LONG" if direction == 1 else "SHORT"

        log.info(f"SIGNAL: {symbol} -> {dir_label} @ ${price:.4f} (SL ${sl_price:.4f})")

        # Track signal for agent's whipsaw detection
        self._signal_history.append({"symbol": symbol, "ts": time.time(), "direction": direction})
        if len(self._signal_history) > 200:
            self._signal_history = self._signal_history[-200:]

        existing = self.positions.get(symbol)

        if existing and existing["direction"] == direction:
            log.info(f"  Already {dir_label} on {symbol}, skipping")
            return

        # ── Consolidation gate: block flips in choppy markets ──
        snap = {}
        if hasattr(self, '_last_snapshot_symbols'):
            snap = self._last_snapshot_symbols.get(f"{symbol}:USDT") or \
                   self._last_snapshot_symbols.get(symbol) or {}
        adx = snap.get("adx", 50)          # default trending (safe)
        bb_w = snap.get("bb_width", 5.0)   # default wide (safe)

        # Decay consecutive losses after 1 hour
        if not hasattr(self, '_last_loss_ts'):
            self._last_loss_ts = {}
        last_loss = self._last_loss_ts.get(symbol, 0)
        if last_loss > 0 and (time.time() - last_loss) > 3600:
            self.consecutive_losses[symbol] = 0

        # Classify zone
        if adx > 25 and bb_w > 2.5:
            zone = "trending"
        elif adx < 20 or bb_w < 2.0:
            zone = "consolidating"
        else:
            zone = "grey"

        # Block logic
        if zone == "consolidating":
            self._add_pending_flip(symbol, direction, price, sl_price, adx, bb_w, zone)
            self._log_event("whipsaw_blocked", symbol,
                f"Flip to {dir_label} blocked — consolidating (ADX={adx:.0f}, BBW={bb_w:.1f}%)")
            log.info(f"  {symbol}: CONSOLIDATION BLOCK — ADX={adx:.0f}, BBW={bb_w:.1f}%, queued for agent")
            return

        if zone == "grey" and self.consecutive_losses.get(symbol, 0) >= 2:
            losses = self.consecutive_losses[symbol]
            self._add_pending_flip(symbol, direction, price, sl_price, adx, bb_w, zone)
            self._log_event("whipsaw_blocked", symbol,
                f"Flip to {dir_label} blocked — grey zone + {losses} consecutive losses")
            log.info(f"  {symbol}: GREY ZONE BLOCK — {losses} losses, ADX={adx:.0f}, BBW={bb_w:.1f}%, queued for agent")
            return

        # ── Trending or grey with <2 losses: execute normally ──
        if existing:
            await self._close_position(symbol, price, "signal_flip")

        # Check agent pause (CLOSE action — clears on signal flip)
        if hasattr(self, "_agent_paused") and symbol in self._agent_paused:
            self._agent_paused.discard(symbol)
            log.info(f"  {symbol}: agent pause cleared — signal flipped, allowing re-entry")

        # Check agent skip (SKIP action — time-based, ignores flips)
        if hasattr(self, "_skip_until"):
            skip_until = self._skip_until.get(symbol, 0)
            if skip_until > 0 and time.time() < skip_until:
                remaining = int((skip_until - time.time()) / 60)
                log.info(f"  {symbol}: SKIPPED by agent — {remaining}m remaining, not opening")
                return

        await self._open_position(symbol, direction, price, sl_price)

    async def handle_snapshot(self, snapshot: dict):
        """Handle a 60s snapshot — check SL hits and ratchet profit floors."""
        if self._syncing:
            return  # sync in progress, ignore snapshots until done

        symbols_data = snapshot.get("symbols", {})

        # Store full snapshot for agent (indicator state per symbol)
        self._last_snapshot_symbols = symbols_data

        # Update last known prices for unrealized P&L calculation
        for sym_key, sym_data in symbols_data.items():
            clean_sym = sym_key.replace(":USDT", "")
            if "price" in sym_data:
                self.last_prices[clean_sym] = sym_data["price"]

        # Don't process positions if bot is paused
        if self.bot_state != "running":
            return

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
                    # Check if bot initiated this close (recently closed set)
                    if hasattr(self, '_recently_closed') and symbol in self._recently_closed:
                        self._recently_closed.discard(symbol)
                        self.positions.pop(symbol, None)
                        self.protective_orders.pop(symbol, None)
                        continue
                    # Not bot-initiated — external close (manual, liquidation, etc.)
                    reason = "external_close"
                    sl_price = self.protective_orders.get(symbol, {}).get("sl_price", 0)
                    close_price = sl_price if sl_price > 0 else current_price
                    log.info(f"  {symbol}: position closed externally @ ${close_price:.4f}")
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
        # Check agent pause (profit-taking — clears on signal flip)
        if hasattr(self, "_agent_paused") and symbol in self._agent_paused:
            log.info(f"  {symbol}: blocked by peak agent (paused until next flip)")
            return
        # Check agent skip (whipsaw — time-based)
        if hasattr(self, "_skip_until"):
            skip_until = self._skip_until.get(symbol, 0)
            if skip_until > 0 and time.time() < skip_until:
                remaining = int((skip_until - time.time()) / 60)
                log.info(f"  {symbol}: blocked by peak agent skip ({remaining}m remaining)")
                return
            elif skip_until > 0:
                # Skip expired — clear it
                del self._skip_until[symbol]

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
        dir_label = "LONG" if direction == 1 else "SHORT"
        self._log_event("open", symbol, f"Opened {dir_label} @ ${price:,.4f}")

    async def _close_position(self, symbol: str, close_price: float, reason: str):
        """Close an existing position via the proxy."""
        # Track bot-initiated closes so snapshot handler doesn't re-process
        if not hasattr(self, '_recently_closed'):
            self._recently_closed = set()
        self._recently_closed.add(symbol)
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

        # Guard against zero/invalid prices
        if close_price <= 0 or entry_price <= 0:
            log.warning(f"  {symbol}: skipping P&L calc — invalid prices (entry={entry_price}, close={close_price})")
            self._save_state()
            dir_label = "LONG" if direction == 1 else "SHORT"
            self._log_event("close", symbol, f"Closed {dir_label} — price unavailable")
            return

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
            if not hasattr(self, '_last_loss_ts'):
                self._last_loss_ts = {}
            self._last_loss_ts[symbol] = time.time()
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

        reason_label = reason.replace("_", " ").title()
        # Map close reasons to specific event types for dashboard icons
        event_type_map = {
            "peak_agent": "agent_close", "bot_agent_peak": "agent_close",
            "peak_agent_skip": "agent_skip", "bot_agent_skip": "agent_skip",
            "signal_flip": "signal_flip", "missed_flip": "signal_flip",
            "external_close": "external_close",
        }
        event_type = event_type_map.get(reason, "close")
        self._log_event(event_type, symbol, f"Closed {dir_label} {pnl_pct:+.2f}% (${pnl_dollar:+,.2f}) — {reason_label}")

        # Report trade close to signal server (fire-and-forget)
        try:
            entry_time = datetime.fromisoformat(pos.get("entry_time", datetime.now().isoformat()))
            duration = int((datetime.now() - entry_time).total_seconds())
        except Exception:
            duration = 0
        asyncio.create_task(
            self._report_trade_close(
                symbol, direction, entry_price, close_price,
                pnl_pct, pnl_dollar, reason, duration,
            )
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

    # ── Status reporting (for customer dashboard) ────────────

    async def _report_status(self):
        """Report bot status to signal server for the customer dashboard."""
        if not self.signal_api_key or not self.proxy:
            return
        http_url = SIGNAL_SERVER_URL.replace("wss://", "https://").replace("ws://", "http://")
        try:
            # Fetch positions directly from Binance (always accurate)
            positions = []
            for symbol in self.active_symbols:
                try:
                    qty, direction = await self.proxy.get_position(symbol)
                    if qty > 0:
                        pos_data = self.positions.get(symbol, {})
                        entry_price = pos_data.get("entry_price", 0)
                        # Calculate unrealized P&L from current price
                        pnl_usd = 0.0
                        pnl_pct = 0.0
                        if entry_price > 0:
                            current_price = self.last_prices.get(symbol, entry_price)
                            if direction == 1:
                                pnl_pct = (current_price - entry_price) / entry_price * 100
                            else:
                                pnl_pct = (entry_price - current_price) / entry_price * 100
                            notional = qty * entry_price
                            pnl_usd = (pnl_pct / 100) * notional  # notional already includes leverage
                        positions.append({
                            "symbol": symbol,
                            "direction": direction,
                            "entry_price": entry_price,
                            "pnl_usd": round(pnl_usd, 2),
                            "pnl_pct": round(pnl_pct, 2),
                        })
                except Exception:
                    pass

            # Get total wallet balance (includes unrealized P&L)
            balance = 0
            try:
                resp = await self.proxy.client.get(
                    f"{PROXY_URL}/v1/balance", headers=self.proxy._headers()
                )
                if resp.status_code == 200:
                    for asset in resp.json():
                        if asset.get("asset") == "USDT":
                            balance = float(asset.get("balance", 0) or 0)
                            break
            except Exception:
                pass
            # Store for agent
            if balance > 0:
                self._last_balance = balance

            # Recent trades (last 20)
            recent = self._get_recent_trades(20)

            payload = {
                "balance": round(balance, 2),
                "positions": positions,
                "recent_trades": recent,
                "uptime_seconds": int(time.time() - START_TIME),
                "bot_state": self.bot_state,
                "events": list(self._events),
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{http_url}/api/bot-status",
                    params={"key": self.signal_api_key},
                    json=payload,
                )
        except Exception as e:
            log.warning(f"Status report error: {e}")

    async def _report_trade_close(self, symbol: str, direction: int,
                                   entry_price: float, exit_price: float,
                                   pnl_pct: float, pnl_usd: float,
                                   reason: str, duration_seconds: int):
        """Report a closed trade to signal server for analytics."""
        if not self.signal_api_key:
            return
        http_url = SIGNAL_SERVER_URL.replace("wss://", "https://").replace("ws://", "http://")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{http_url}/api/trade-close",
                    params={"key": self.signal_api_key},
                    json={
                        "symbol": symbol,
                        "direction": direction,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_usd": round(pnl_usd, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "close_reason": reason,
                        "duration_seconds": duration_seconds,
                    },
                )
        except Exception as e:
            log.warning(f"Failed to report trade close: {e}")

    def _get_recent_trades(self, limit: int = 20) -> list[dict]:
        """Get recent closed trades from state."""
        state = self._load_state_dict()
        trades = state.get("trades", [])
        return trades[-limit:] if trades else []

    def _load_state_dict(self) -> dict:
        """Load raw state dict from disk."""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

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
            "bot_state": self.bot_state,
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
